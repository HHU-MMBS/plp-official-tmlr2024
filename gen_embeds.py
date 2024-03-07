"""
Generate embeddings for a given dataset and model.
"""
import os
from os.path import exists
import json
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.backends import cudnn
from tqdm import tqdm

from eval.classification import knn_classifier
from loaders import get_dataset, get_num_classes
from loaders.datasets import check_dataset, get_embeds_path
from model_builders import load_backbone, load_embeds
from utils import load_pretrained_weights


@torch.no_grad()
def compute_same_class_embeds(embeds, labels):
    same_class_embeds = []
    length = []
    all_indices = torch.arange(len(embeds))
    for i in range(len(embeds)):
        # computes the indices of the same class and get the indices of the same class
        indices = all_indices[labels == labels[i]]
        same_class_embeds.append(indices)
        
        if len(embeds[labels == labels[i]]) not in length:
            length.append(len(embeds[labels == labels[i]]))
    return same_class_embeds

@torch.no_grad()
def compute_embedding(model, loader, keys=["label", "uq_idx", "label_split"]):
    batch_info = len(next(iter(loader)))
    if batch_info == 2:
        embeds = []
        labels = []
        for images, label in tqdm(loader):
            images = images.cuda()
            image_features = model(images).float()
            embeds.append(image_features.cpu())
            labels.append(label)
        return torch.cat(embeds).squeeze_(), torch.cat(labels)
    else:
        assert (batch_info-1) <= len(keys), f"Number of keys {len(keys)} must match number of batch info {batch_info-1}"
        embeds = []
        other_info_dict = {}
        #initialize dict
        for i in range(batch_info-1):
            other_info_dict[keys[i]] = []
        
        for datapoint in tqdm(loader):
            images, other_info = datapoint[0], datapoint[1:]
            images = images.cuda()
            image_features = model(images).float()
            embeds.append(image_features.cpu())
            for c,value in enumerate(other_info):
                other_info_dict[keys[c]].append(value)
        #concatenate all tensors in dict
        for key in other_info_dict.keys():
            other_info_dict[key] = torch.cat(other_info_dict[key]).cpu()
        return torch.cat(embeds), other_info_dict

@torch.no_grad()
def compute_neighbors(embedding, k=50, step_size=32, device='cuda'):
    embedding = embedding.to(device)
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    num_embeds = embedding.shape[0]
    if num_embeds <= 8*1e4:
        dists = embedding @ embedding.permute(1, 0)
        # exclude self-similarity
        dists.fill_diagonal_(-torch.inf)
        return dists.topk(k, dim=-1).indices
    else:
        topk_knn_ids = []
        print(f"Chunk-wise implementation step = {step_size}")
        # num_embeds // num_chunks
        embedding = embedding.cuda()
        for idx in tqdm(range(0, num_embeds, step_size)):
            idx_next_chunk = min((idx + step_size), num_embeds)
            features = embedding[idx : idx_next_chunk, :]
            # calculate the dot product dist
            dists_chunk = torch.mm(features, embedding.T)
            dists_chunk.fill_diagonal_(-torch.inf)
            _, indices = dists_chunk.topk(k, dim=-1)
            topk_knn_ids.append(indices)
        return torch.cat(topk_knn_ids)
    
        
def get_outpath(arch, dataset, datapath=get_embeds_path()):
    datapath = Path(datapath).expanduser().resolve()
    arch = arch.replace('/', '_')
    dataset = dataset.replace('/', '_')
    return datapath / f'{dataset}'/ f'{arch}'

def save_embeds(outpath, embeddings, label, test_str):
    torch.save(embeddings, outpath / f'embeddings{test_str}.pt')
    torch.save(embeddings.mean(dim=0), outpath / f'mean{test_str}.pt')
    torch.save(embeddings.std(dim=0), outpath / f'std{test_str}.pt')
    torch.save(label, outpath / f'label{test_str}.pt')    

def get_nn(args, preprocess, model, test=False, gen_split="all"):
    print("Computing Embeddings....")
    dset = get_dataset(args.dataset, datapath=args.datapath, train=not test, transform=preprocess, download=True, gen_split=gen_split)
    dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=16)
    embeddings, labels = compute_embedding(model, dataloader)    
    return embeddings, labels


def compute_stats(outpath):
    for test in True, False:
        test_str = '-test' if test else ''
        embeddings = torch.load(outpath / f'embeddings{test_str}.pt', map_location='cpu')
        torch.save(embeddings.mean(dim=0), outpath / f'mean{test_str}.pt')
        torch.save(embeddings.std(dim=0), outpath / f'std{test_str}.pt')

def main(args):
    cudnn.benchmark = True
    cudnn.deterministic = True
    model, preprocess = load_backbone(args.arch)
    
    model = model.cuda()
    model.eval()
    datasets = args.dataset
    for dataset in datasets:
        args.dataset = dataset
        print('Computing embeddings for', dataset)
        compute_emb_for_dset(args, model, preprocess)
    # empty gpu memory
    model.cpu()
    del model


def compute_emb_for_dset(args, model, preprocess):
    outpath = get_outpath(args.arch, args.dataset)
    outpath.mkdir(parents=True, exist_ok=True)
    if args.stats_only:
        compute_stats(outpath)
        return
    num_classes = get_num_classes(args.dataset)
    
    embs = {}
    labels = {}
    embeds_exist = exists(outpath / f'embeddings.pt') and exists(outpath / f'embeddings-test.pt')
    if embeds_exist and not args.overwrite and "Gen" not in args.dataset:
        print(f'\n\n Embeddings for {args.dataset} with {args.arch} already exist. Loading...')
        for test in [False, True] if not args.test_only else [True]:
            test_str = '-test' if test else ''
            embs[test] = torch.load(outpath / f'embeddings{test_str}.pt', map_location='cpu')
            labels[test] = torch.load(outpath / f'label{test_str}.pt', map_location='cpu')
    else:
        for test in [False, True] if not args.test_only else [True]:
            embeddings, labels_curr = get_nn(args, preprocess, model, test, gen_split="all")
            test_str = '-test' if test else ''
            save_embeds(outpath, embeddings, labels_curr, test_str)
            embs[test] = embeddings
            labels[test] = labels_curr
            # extra knn computation for Gen datasets - NOT saved!
            if test and "Gen" in args.dataset:
                orig_test_emb, orig_labels_test = get_nn(args, preprocess, model, test=True, gen_split="test")
        
    if not args.no_compute_knn:
        k = 50 if args.k is None else args.k
        neighbors = compute_neighbors(embs[False], k, args.knn_step)
        torch.save(neighbors, outpath / f'knn.pt')
    
    if "Gen" in args.dataset and isinstance(labels[False], dict):
        print("Keys in label dict found:", labels[False].keys())
        train_labels_split = labels[False]["label_split"]
        labels_train = labels[False]["label"]
        print("Computing true positive pairs for LABELED train set...")
        labeled_data, valid_labels = embs[False][train_labels_split], labels_train[train_labels_split]
        print("Length of labeled data:", len(labeled_data), len(valid_labels))
        assert  len(labeled_data) == len(valid_labels), "Length of labeled data and labels must be the same"
        same_class_embeds = compute_same_class_embeds(labeled_data, valid_labels)
        torch.save(same_class_embeds, outpath / f'true_pos_knn_lab.pt')
        print("Computing KNN pairs for UNLABELED train split...")
        unlabeled_data_embeds = embs[False][~train_labels_split]
        unlab_neighbors = compute_neighbors(unlabeled_data_embeds, k=50, device='cuda').cpu()
        torch.save(unlab_neighbors, outpath / f'unlab_knn.pt')
        # hacky way to compute the knn for the original/official test split
        test_features = orig_test_emb
        labels_test = orig_labels_test["label"]
    else:
        labels_train = labels[False]
        labels_test = labels[True]
        test_features=embs[True]
        
    if not args.no_eval_knn and not args.test_only:
        print('Computing KNN accuracy')
        # train_features, train_labels, test_features, test_labels, k, T, num_classes)
        top1, top5 = knn_classifier(
            train_features=embs[False],
            train_labels=labels_train,
            test_features=test_features,
            test_labels=labels_test,
            k=args.classifier_k,
            T=args.temperature,
            num_classes=num_classes)
        
        print(f'Top-1 accuracy: {top1}, Top-5 accuracy: {top5}')
        with open(outpath / 'accuracy.json', 'w') as f:
            json.dump({'top1': top1, 'top5': top5}, f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default=['CIFAR100'], type=check_dataset, nargs='+')
    parser.add_argument('--arch', default='openclip_ViT-B-16/openai')
    parser.add_argument('--outpath', type=Path, default=Path(get_embeds_path()))
    parser.add_argument('--temperature', default=0.02, type=float,
                        help='Temperature used in the voting coefficient')
    parser.add_argument('--classifier-k', default=20, type=int, help='Numbers of neighbors to use in the classifier')
    parser.add_argument('-k', type=int, default=50, help='total NNs to compute. Default: num images / num classes')
    parser.add_argument('--vit_image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--knn_step', type=int, default=64)
    parser.add_argument('--datapath', default=None, type=str)
    parser.add_argument('--no_eval_knn', action='store_true', help='Do not evaluate k-nn accuracy', default=False)
    parser.add_argument('--stats_only', action='store_true',
                        help='Only compute the mean and std of the dataset for precomputed embeddings')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing embeddings')
    parser.add_argument('--test_only', action='store_true', help='Only compute on test set', default=False)
    parser.add_argument('--no_compute_knn', action='store_true', help='Only compute on test set', default=False)
    # Using tuned/modified weights
    parser.add_argument('--weights', default=None, type=str)
    parser.add_argument('--ckpt_key', default="model", type=str)
    parser.add_argument('--prefix', default=None, type=str)
    main(parser.parse_args())
