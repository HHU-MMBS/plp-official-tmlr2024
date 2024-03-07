"""
Evaluation for things that include some for of assignment,
i.e., knn accuracy and clustering performance
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_mutual_info_score as adjusted_nmi, \
    adjusted_rand_score as adjusted_rand_index
from torch import nn
from tqdm import tqdm


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    """
    KNN classifier with cosine similarity and temperature scaling
    """
    if isinstance(train_labels, np.ndarray):
        train_labels = torch.from_numpy(train_labels)
    
    if isinstance(test_labels, np.ndarray):
        test_labels = torch.from_numpy(test_labels)
    
    try:
        train_features = train_features.cuda()
        test_features = test_features.cuda()
        train_labels = train_labels.cuda()
    except:
        print("Featues don't fit the GPU, ignoring KNN similarity evaluation")
        return 0,0    
    
    train_features = nn.functional.normalize(train_features, dim=1, p=2)
    test_features = nn.functional.normalize(test_features, dim=1, p=2)

    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    if train_features.shape[-1] > 100000:
        imgs_per_chunk = 32
    else:
        imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in tqdm(range(0, num_test_images, imgs_per_chunk)):
        # get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]
        features = features.to(train_features.device)

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)

        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()

        temp = torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            )

        probs = torch.sum(temp,1)
        _, predictions = probs.sort(1, True)
        
        targets = targets.cpu()
        predictions = predictions.cpu()
        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


def compute_metrics(targets, preds, min_samples_per_class=5, superclass_mapping=None, print_results=True):
    """
    copied from https://github.com/elad-amrani/self-classifier/blob/39351384277d4541a0b7525a66770cacdd8f12c3/src/cls_eval.py#L406
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    val_nmi = nmi(targets, preds)
    val_adjusted_nmi = adjusted_nmi(targets, preds)
    val_adjusted_rand_index = adjusted_rand_index(targets, preds)

    # compute accuracy
    num_classes = max(targets.max(), preds.max()) + 1
    count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for ii in range(preds.shape[0]):
        count_matrix[preds[ii], targets[ii]] += 1
    reassignment = np.dstack(linear_sum_assignment(count_matrix.max() - count_matrix))[0]

    if len(np.unique(preds)) > len(np.unique(targets)):  # if using over-clustering, append remaining clusters to best option
        for cls_idx in np.unique(preds):
            if reassignment[cls_idx, 1] not in targets:
                reassignment[cls_idx, 1] = count_matrix[cls_idx].argmax()

    if superclass_mapping is not None:
        count_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for ii in range(preds.shape[0]):
            count_matrix[preds[ii], superclass_mapping[targets[ii]]] += 1
        for ii in range(len(reassignment[:, 1])):
            reassignment[ii, 1] = superclass_mapping[reassignment[ii, 1]]
    acc = count_matrix[reassignment[:, 0], reassignment[:, 1]].sum().astype(np.float32) / preds.shape[0]

    if print_results:
        print('=> number of samples: {}'.format(len(targets)))
        print('=> number of unique assignments: {}'.format(len(set(preds))))
        print('=> NMI: {:.3f}%'.format(val_nmi * 100.0))
        print('=> Adjusted NMI: {:.3f}%'.format(val_adjusted_nmi * 100.0))
        print('=> Adjusted Rand-Index: {:.3f}%'.format(val_adjusted_rand_index * 100.0))
        print('=> Accuracy: {:.3f}%'.format(acc * 100.0))

    # extract max accuracy classes
    num_samples_per_class = count_matrix[reassignment[:, 0], :].sum(axis=1)
    acc_per_class = np.where(num_samples_per_class >= min_samples_per_class,
                             count_matrix[reassignment[:, 0], reassignment[:, 1]] / num_samples_per_class, 0)
    max_acc_classes = np.argsort(acc_per_class)[::-1]
    acc_per_class = acc_per_class[max_acc_classes]
    num_samples_per_class = num_samples_per_class[max_acc_classes]

    return acc * 100, val_nmi * 100.0, val_adjusted_nmi * 100.0, val_adjusted_rand_index * 100.0
