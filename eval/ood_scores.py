"""
Eval ood performance
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm
from utils import numpy_input


def available_norms():
    return ["softmax", "l2", "l1", "none"]


@torch.no_grad()
def _norm_by_name(x, norm):
    assert norm in available_norms()
    if norm == "softmax":
        return x.softmax(dim=-1)
    elif norm == "l2":
        return torch.nn.functional.normalize(x, dim=-1, p=2)
    elif norm == "l1":
        return torch.nn.functional.normalize(x, dim=-1, p=1)
    elif norm == "none":
        return x
    raise ValueError(f"norm {norm} not recognized")


def norm_feats(*features, norm="softmax"):
    return [_norm_by_name(x, norm) for x in features]


@torch.no_grad()
def calc_maha_distance(embeds, means_c, inv_cov_c):
    diff = embeds - means_c
    dist = diff @ inv_cov_c * diff
    dist = dist.sum(dim=1)
    return dist


# TODO: make proper caching
def get_mahadist_singleton(*args, **kwargs):
    """
    Hacky singleton to avoid recomputing mahadist
    """
    if get_mahadist_singleton.instance is None:
        print('Computing mahadist statistics')
        get_mahadist_singleton.instance = MahaDist(*args, **kwargs)
    else:
        print('Using cached mahadist statistics')
    return get_mahadist_singleton.instance
get_mahadist_singleton.instance = None


class MahaDist:

    def __init__(
            self,
            train_embeds_in,
            train_labels_in,
            test_embeds_in,
            num_classes,
            std_all=False
    ):
        self.class_means, self.cov_invs, self.avg_train_mean, self.avg_train_inv_cov = compute_maha_vals(
            train_embeds_in, train_labels_in, num_classes, std_all=std_all
        )
        self.in_score, self.in_score_relative = self.compute_scores(test_embeds_in, relative=True)

    @torch.no_grad()
    def compute_scores(self, embeds, relative=False):
        embeds = embeds.double().cpu()
        scores = [calc_maha_distance(embeds, c_mean, self.cov_invs) for c_mean in self.class_means]
        scores = torch.stack(scores)

        scores_normal = -scores.min(dim=0)[0]

        if not relative:
            return scores_normal

        avg_score = calc_maha_distance(embeds, self.avg_train_mean, self.avg_train_inv_cov)
        scores -= avg_score
        scores_relative = -scores.min(dim=0)[0]

        return scores_normal, scores_relative

    @torch.no_grad()
    def __call__(self, test_embeds_out, relative=False):
        if relative:
            in_score = self.in_score_relative
            _, out_score = self.compute_scores(test_embeds_out, relative=True)
        else:
            in_score = self.in_score
            out_score = self.compute_scores(test_embeds_out, relative=False)

        scores = torch.cat((in_score, out_score))
        return scores


@torch.no_grad()
def compute_maha_vals(train_embeds_in, train_labels_in, num_classes, std_all=False):
    train_embeds_in = train_embeds_in.double().cpu()
    class_covs = []
    class_means = []
    # calculate class-wise means and covariances
    for c in range(num_classes):
        train_embeds_c = train_embeds_in[train_labels_in == c]
        if len(train_embeds_c) > 1:
            class_mean = train_embeds_c.mean(dim=0)
            cov = train_embeds_c.T.cov()
            class_covs.append(cov)
            class_means.append(class_mean)

    # class-wise std estimation
    if not std_all:
        cov_invs = torch.stack(class_covs, dim=0).mean(dim=0).inverse()
    else:
        # estimating the global std from train data
        cov_invs = train_embeds_in.T.cov().inverse()

    avg_train_mean = train_embeds_in.mean(dim=0)
    avg_train_inv_cov = train_embeds_in.T.cov().inverse()

    return class_means, cov_invs, avg_train_mean, avg_train_inv_cov



def available_metrics():
    return ["sim", "cos-sim", "temp-cos-sim", "BC", "MI", "l1", "l2", "norm-l1", "norm-l2"]


@torch.no_grad()
def _get_similarity_by_name(x, y, metric, args):
    assert metric in available_metrics()
    if metric == "cos-sim" or metric == "temp-cos-sim":
        x = nn.functional.normalize(x, dim=-1, p=2)
        y = nn.functional.normalize(y, dim=-1, p=2)
        sim = x @ y.T
        if metric == "temp-cos-sim":
            sim /= args.temperature
        return sim.exp()
    elif metric == "BC" or metric == "MI" or metric == "sim":
        pkx = (x / args.teacher_temp).softmax(dim=-1)
        pky = (y / args.teacher_temp).softmax(dim=-1)
        pk = pkx.mean(dim=0)
        if metric == "BC":
            return pkx.sqrt() @ pky.sqrt().T
        elif metric == "MI":
            return ((pkx / pk) @ pky.T).log()
        elif metric == "sim":
            return pkx @ pky.T
    elif metric == "l2":
        return -torch.cdist(x, y, p=2)
    elif metric == "l1":
        return -torch.cdist(x, y, p=1)
    elif metric == "norm-l2":
        return -torch.cdist(nn.functional.normalize(x, dim=-1, p=2), nn.functional.normalize(y, dim=-1, p=2), p=2)
    elif metric == "norm-l1":
        return -torch.cdist(nn.functional.normalize(x, dim=-1, p=1), nn.functional.normalize(y, dim=-1, p=1), p=1)
    else:
        raise NotImplementedError


@torch.no_grad()
def OOD_classifier_knn(train_features, test_features, k, args, metric, num_chunks=256):
    """
    Use k = -1 for whole trainset
    """
    all_imgs = len(train_features)
    if k < 0:
        k = all_imgs

    num_test_images = test_features.shape[0]
    if all_imgs > 100000:
        imgs_per_chunk = 32
    else:
        imgs_per_chunk = num_test_images // num_chunks
    cos_sim = torch.zeros(num_test_images)
    try:
        train_features = train_features.cuda()
        cos_sim = cos_sim.cuda()
    except:
        # Featues don't fit the GPU, ignoring OOD KNN similarity evaluation
        return None
    
    for idx in tqdm(range(0, num_test_images, imgs_per_chunk)):
        idx_next_chunk = min((idx + imgs_per_chunk), num_test_images)
        features = test_features[idx : idx_next_chunk, :]
        features = features.to(train_features.device)
        similarity = _get_similarity_by_name(features, train_features, metric, args)
        top_sim, _ = similarity.topk(k, largest=True, sorted=True, dim=-1)
        cos_sim[idx: idx_next_chunk] = top_sim.mean(dim=1)

    if getattr(args, "crops_number", 1) > 1:
        cos_sim = torch.chunk(cos_sim, args.crops_number)
        _, cos_sim_local_mean = torch.std_mean(torch.stack(cos_sim), 0)
        cos_sim = cos_sim_local_mean

    return cos_sim


def OOD_cls_max_val(test_features, test_features_ood, norm="softmax"):
    test_features, test_features_ood = norm_feats(test_features, test_features_ood, norm=norm)
    score_ood = torch.max(test_features_ood, dim=-1)[0]
    score_id = torch.max(test_features, dim=-1)[0]
    scores = torch.cat((score_id, score_ood))
    return scores


def msp(logits: torch.Tensor, temperature: float = 1.):
    logits = logits / temperature
    probs = logits.softmax(dim=-1)
    max_probs, _ = probs.max(dim=-1)
    return max_probs

def perplexity(logits: torch.Tensor, temperature: float = 1.):
    """ Actually negative perplexity """
    logits = logits / temperature
    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return -entropy.exp()


def free_energy(logits: torch.Tensor, temperature: float = 1.):
    """
    Negative free energy
    From Liu et al. 2020
    'Energy-based Out-of-distribution Detection'
    https://arxiv.org/abs/2010.03759
    """
    logits = logits / temperature
    return temperature * logits.logsumexp(dim=-1)

def l1_norm(logits: torch.Tensor, non_neg=True):
    """
    torch.linalg.norm(logits, ord=1, dim=-1)
    """
    if non_neg:
        logits = logits - logits.min(dim=-1, keepdim=True)[0]
    return torch.linalg.norm(logits, ord=1, dim=-1)