""" Various stuff that is used during evaluation """
import itertools
from collections import OrderedDict

import torch

from loaders.datasets import get_class_names
from model_builders.openclip import build_openclip_text
from utils import _dict_to_kv_pair

import argparse
import itertools
import json
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import eval.classification
import utils
from eval.binary_metrics import auroc_score, fpr95_score
from eval.classification import knn_classifier
from eval.features import FeatureExtractionPipeline
from eval.ood_scores import *
from loaders import get_num_classes, check_dataset, get_embeds_path, get_ood, get_default_path
from model_builders import load_embeds
from scipy.optimize import linear_sum_assignment as linear_assignment

@torch.no_grad()
def clip_label_embeddings(arch, dataset, context_prompts=None, device="cpu", logit_scale=True):
    """
    Computes the embeddings of the sentence + all labels in the dataset
    Returns the (optionally scaled) embeddings and the visual projection matrix if available
    Scaled means, L2 normalized and multiplied by the logit scale
    Context_prompts: prompts to use, must have a single {c} placeholder. Default: "an image of a {c}"
    """
    if context_prompts is None:
        context_prompts = [
        "a photo of a {c}.",
        "a blurry photo of a {c}.",
        "a photo of many {c}.",
        "a photo of the large {c}.",
        "a photo of the small {c}."]
    
    if isinstance(context_prompts, str):
        context_prompts = [context_prompts]

    model, tokenizer, proj = build_openclip_text(arch)
    class_names = get_class_names(dataset)
    tokens = [tokenizer([p.format(c=name) for name in class_names]).to(device) for p in context_prompts]
    model.to(device)
    proj = proj.to(device)
    model.eval()
    embeddings = [model.encode_text(t, normalize=True) for t in tokens]
    embeddings = torch.stack(embeddings, dim=0).mean(dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    if logit_scale:
        embeddings *= model.logit_scale.exp()
    return embeddings, proj


def flatten_result(result, prefix='') -> OrderedDict:
    if isinstance(result, dict):
        kv_pairs = [list(flatten_result(v, prefix=k).items()) for k, v in result.items()]
        if prefix:
            prefix += '_'
        kv_pairs = [(prefix+k, v) for k, v in itertools.chain(*kv_pairs)]
        return OrderedDict(kv_pairs)
    if isinstance(result, list):
        result = [_dict_to_kv_pair(flatten_result(x), prefix) for x in result]
        return OrderedDict(result)
    return {prefix: result}


def pprint_dict(d):
    d = flatten_result(d)
    offset = max([len(k) for k in d]) + 2
    for k, v in d.items():
        print(f"{k+':':<{offset}} {v:.2f}")



def eval_ood(args, epoch, test_features, test_features_ood, train_features, train_labels, writer=None):
    # ood evaluation
    result = OrderedDict(
            epoch = epoch,
            auroc = OrderedDict(),
            fpr95 = OrderedDict())
    num_in = int(test_features.shape[0])
    num_out = int(test_features_ood.shape[0])
    labels = torch.cat((torch.ones(num_in), torch.zeros(num_out))).numpy()
    all_scores = compute_all_ood_scores(args, test_features, test_features_ood, train_features, train_labels)
    for method, scores in all_scores:
        auroc = auroc_score(labels, scores) * 100
        fpr95 = fpr95_score(labels, scores) * 100
        if isinstance(method, str):
            method_name = method
            result["auroc"][method_name] = auroc
            result["fpr95"][method_name] = fpr95
            if writer is not None:
                writer.add_scalar(f'eval-auroc/{method_name}', auroc, global_step=epoch)
                writer.add_scalar(f'eval-fpr95/{method_name}', fpr95, global_step=epoch)
        else:
            method_name, params = method
            aurocs = result["auroc"].get(method_name, [])
            aurocs.append(OrderedDict(params, auroc=auroc))
            result["auroc"][method_name] = aurocs
            fpr95s = result["fpr95"].get(method_name, [])
            fpr95s.append(OrderedDict(params, fpr95=fpr95))
            result["fpr95"][method_name] = fpr95s
            if writer is not None:
                suffix = '/'.join(str(p) for p in params.values())
                writer.add_scalar(f'eval-auroc/{method_name}/{suffix}', auroc, global_step=epoch)
                writer.add_scalar(f'eval-fpr95/{method_name}/{suffix}', fpr95, global_step=epoch)
    return result

def compute_all_ood_scores(args, test_features, test_features_ood, train_features, train_labels):
    """
    Computes all relevant OOD scores.
    Returns a list of (method, scores) tuples, where method is either the method name (str)
    or a tuple (method_name, params) where params is an OrderedDict of parameters.
    """
    all_scores = []
    num_classes = get_num_classes(args.dataset)
    if args.eval_ood_maha:
        maha_dist = MahaDist(
            train_embeds_in=train_features,
            train_labels_in=train_labels,
            test_embeds_in=test_features,
            num_classes=num_classes)
        maha_scores = maha_dist(test_features_ood)
        all_scores.append(("Mahalanobis", maha_scores))
        maha_scores_relative = maha_dist(test_features_ood, relative=True)
        all_scores.append(("Mahalanobis_relative", maha_scores_relative))

    if args.eval_ood_knn:
        for k, metric in itertools.product(args.ood_knn_ks, args.ood_knn_metrics):
            try:
                train_features = train_features.cuda()
                scores_in = OOD_classifier_knn(train_features, test_features, k, args, metric=metric)
                scores_out = OOD_classifier_knn(train_features, test_features_ood, k, args, metric=metric)
                if torch.is_tensor(scores_in) and torch.is_tensor(scores_out):
                    scores = torch.cat((scores_in, scores_out))
                    params = OrderedDict(k=k, metric=metric)
                    all_scores.append((("knn", params), scores))
            except:
                continue
            
    if args.eval_ood_norm:
        for norm in args.ood_norms:
            scores = OOD_cls_max_val(test_features, test_features_ood, norm=norm)
            params = OrderedDict(norm=norm)
            all_scores.append((("max", params), scores))
    if args.eval_ood_logits:
        for score_fn in (free_energy, msp, l1_norm):
            for temp in args.ood_logits_temp:
                scores_in = score_fn(test_features, temp)
                scores_out = score_fn(test_features_ood, temp)
                scores = torch.cat((scores_in, scores_out))
                params = OrderedDict(temp=temp)
                all_scores.append(((score_fn.__name__, params), scores))
    return all_scores


def eval_knn_acc(args, epoch, result, test_features, train_features, train_labels, val_labels, writer=None):
    result["knn_acc"] = []
    for k in args.nb_knn:
        top1, top5 = knn_classifier(train_features, train_labels, test_features, val_labels, k, args.temperature,
                                    num_classes=get_num_classes(args.dataset))

        result["knn_acc"].append(OrderedDict(k=k, top=1, acc=top1))
        result["knn_acc"].append(OrderedDict(k=k, top=5, acc=top5))
    if writer is not None:
        # Only write last k-NN result to tensorboard
        writer.add_scalar('eval-accuracy/Top1', top1, global_step=epoch)
        writer.add_scalar('eval-accuracy/Top5', top5, global_step=epoch)


def eval_cluster(epoch, result, test_features, val_labels, writer=None, train_metrics=False, metric_key='cluster'):
    # Cluster performance test
    pred_labels = test_features.argmax(dim=-1)
    pred_labels = pred_labels.cpu().numpy()
    cluster_acc, nmi, anmi, ari = eval.classification.compute_metrics(val_labels, pred_labels,
                                                                       min_samples_per_class=5, print_results=False)
    if writer is not None:
        writer.add_scalar('eval-cluster/acc', cluster_acc, global_step=epoch)
        writer.add_scalar('eval-cluster/nmi', nmi, global_step=epoch)
        writer.add_scalar('eval-cluster/anmi', anmi, global_step=epoch)
        writer.add_scalar('eval-cluster/ari', ari, global_step=epoch)
    
    result[metric_key] = OrderedDict(acc=cluster_acc, nmi=nmi, anmi=anmi, ari=ari)
    
    if train_metrics:
        # Cluster performance train
        pred_labels = test_features.argmax(dim=-1)
        pred_labels = pred_labels.cpu().numpy()
        cluster_acc, nmi, anmi, ari = eval.classification.compute_metrics(val_labels, pred_labels,
                                                                        min_samples_per_class=5, print_results=False)
        if writer is not None:
            writer.add_scalar('eval-cluster/acc-train', cluster_acc, global_step=epoch)
            writer.add_scalar('eval-cluster/nmi-train', nmi, global_step=epoch)
            writer.add_scalar('eval-cluster/anmi-train', anmi, global_step=epoch)
            writer.add_scalar('eval-cluster/ari-train', ari, global_step=epoch)
        result[metric_key] = OrderedDict(acc_train=cluster_acc, nmi_train=nmi, anmi_train=anmi, ari_train=ari)



def _print_results(args, d):
    if args.eval_ood:
        print('Auroc')
        pprint_dict(d['auroc'])

def load_tensorboard_loss(path):
    if type(path) is str:
        path = Path(path)
    tag = 'Train loss epoch'
    event_acc = EventAccumulator(str(next(path.glob('event*'))))
    event_acc.Reload()
    if tag in event_acc.Tags()['scalars']:
        return pd.DataFrame([{'Epoch': ev.step, 'loss': ev.value}
                             for ev in event_acc.Scalars(tag)]).set_index('Epoch')
    # Multihead case
    dfs = []
    for p in path.rglob('Train loss*/event*'):
        event_acc = EventAccumulator(str(p))
        event_acc.Reload()
        dfs.append(pd.DataFrame([{'Epoch': ev.step, 'loss': ev.value}
                                 for ev in event_acc.Scalars(tag)]).set_index('Epoch'))
    df = pd.concat(dfs)
    return df.groupby('Epoch').min()

def create_summary(df):
    try:
        oods = df["out_dist"].unique() 
    except:
        oods = df["out_dist"].T.unique() 
    best_rows = []
    for ood in oods:
        df_ood = df[df['out_dist'].isin([ood])]
        for col_name in df.columns:
            if "auroc" in col_name or "cluster_acc" in col_name or "cluster_ari" in col_name or "knn_acc_k=20_top=1" in col_name:
                idx = df_ood[col_name].idxmax()
            elif "train_loss" in col_name or "loss_train" in col_name:
                idx = df_ood[col_name].idxmin()
            else:
                continue
            # Append idx if not already in best_rows
            if idx not in best_rows:
                best_rows.append(idx)
    return df.iloc[best_rows]

def get_best_ckpts(ckpt_dir):
    """Evaluates only the last ckpt and the one with the lowest loss
    """
    df = load_tensorboard_loss(ckpt_dir)
    checkpoint_list = sorted(ckpt_dir.glob('*.pth'), key=get_checkpoint_number)
    #drops last one (np.inf)
    ckpt_ids = [get_checkpoint_number(ckpt) for ckpt in checkpoint_list][:-1]
    # append max index from last epoch
    ckpt_ids.append(df.last_valid_index())
    df = df.iloc[ckpt_ids]
    # get last ckpt and lowest loss
    lowest_loss_idx = df["loss"].idxmin()
    
    if lowest_loss_idx != df.last_valid_index():
        checkpoint_list = list( checkpoint_list[i] for i in [ckpt_ids.index(lowest_loss_idx), -1] )
        df = df.iloc[[ckpt_ids.index(lowest_loss_idx),-1]]
    else:
        df = df.iloc[[-1]]
        checkpoint_list = [checkpoint_list[-1]]
    return df, checkpoint_list

def get_checkpoint_number(ckpt: Path):
    # Get the epoch number from the checkpoint name
    numbers = re.findall(r'\d+', ckpt.name)
    if numbers:
        return int(numbers[-1])
    return np.inf


@torch.no_grad()
def eval_features(args, epoch, test_features, test_features_ood, train_features, train_labels, test_labels, writer=None):
    """
    Evaluation function used for backbone evaluations only! 
    KNN acc, cluster acc, OOD detection in one function
    """
    result = OrderedDict(
        epoch = epoch,
        auroc = OrderedDict(),
        fpr95 = OrderedDict(),
        cluster = OrderedDict(),
        knn_acc = OrderedDict())

    if args.eval_cluster: 
        eval_cluster(epoch, result, test_features, test_labels, writer)  

    if args.eval_knn_acc:
        eval_knn_acc(args, epoch, result, test_features, train_features, train_labels, test_labels, writer)

    if args.eval_ood:
        # TODO test backbone eval 
        result_ood = eval_ood(args, epoch, test_features, test_features_ood, train_features, train_labels, writer)
        result.update(result_ood)
    return result

def eval_backbone(args, writer):
    datapath = get_embeds_path()
    test_features_ood, epoch, losses_df = None, None, None
    train_features, train_labels = load_embeds(arch=args.arch, dataset=args.dataset, datapath=datapath,
                                               with_label=True, test=False)
    test_features, test_labels = load_embeds(arch=args.arch, dataset=args.dataset, datapath=datapath,
                                             with_label=True, test=True)
    if args.eval_ood:
        # TEMP hack - only evaluating 1 OOD dataset
        if isinstance(args.out_dist, list):
            ood = args.out_dist[0]
        else:
            ood = args.out_dist
        test_features_ood = load_embeds(arch=args.arch, dataset=ood, datapath=datapath,
                                        with_label=False, test=True)
        print(f"OOD  {ood} features loaded {test_features_ood.shape}")
    
    if args.embed_norm:
        mean, std = train_features.mean(dim=0), train_features.std(dim=0)
        train_features = (train_features - mean) / std
        test_features = (test_features - mean) / std
        if test_features_ood is not None:
            test_features_ood = (test_features_ood - mean) / std
                
    result = eval_features(args, epoch, test_features, test_features_ood, train_features, train_labels, test_labels,
                           writer)
    return [result]




def eval_semisup(extractor, dataset,ckpt, split="all"):
    extractor.load_ckpt(pretrained_weights=ckpt)
    dl_train_merged = extractor.get_dataloader(dataset, train=True)
    mask_unseen = dl_train_merged.dataset.mask
    dl_test_unlab_all = extractor.get_dataloader(dataset, train=False,  gen_split=split)
    embeds = extractor.get_embeds(dataloader=dl_test_unlab_all, save_key=split)
    test_logits = extractor.head_logits(embeds)
    test_labels = extractor.get_labels(dl_test_unlab_all.dataset, save_key=split) 
    res_dict = compute_semisup_accs(test_labels, test_logits, mask_unseen)
    return res_dict

def compute_semisup_accs(y_true, y_pred, mask):
    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """
    results = {}
    if torch.is_tensor(y_pred):
        y_pred = y_pred.argmax(dim=-1).cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    total_acc, acc_old, acc_new = semisup_cluster_eval(y_true, y_pred, mask) 
    results["ACC"] = OrderedDict(all=total_acc, seen=acc_old, unseen=acc_new)
    acc_unseen, nmi, anmi, ari = eval.classification.compute_metrics(y_true[mask], y_pred[mask], min_samples_per_class=5, print_results=False)
    results["Unseen"]= OrderedDict(acc=acc_unseen, nmi=nmi, anmi=anmi, ari=ari)
    acc_seen, nmi, anmi, ari = eval.classification.compute_metrics(y_true[~mask], y_pred[~mask], min_samples_per_class=5, print_results=False)
    results["Seen"]= OrderedDict(acc=acc_seen, nmi=nmi, anmi=anmi, ari=ari)
    cluster_acc, nmi, anmi, ari = eval.classification.compute_metrics(y_true, y_pred, min_samples_per_class=5, print_results=False)
    results["Metrics_all"]= OrderedDict(acc=cluster_acc, nmi=nmi, anmi=anmi, ari=ari)    
    return results

def semisup_cluster_eval(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Requires scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[~mask])
    new_classes_gt = set(y_true[mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances
    return total_acc*100, old_acc*100, new_acc*100
