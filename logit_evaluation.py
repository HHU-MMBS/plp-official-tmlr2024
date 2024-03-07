"""
Script to evaluate OOD performance for logit based approaches.
Either using probing results, in which case the probing path must be specified,
or using CLIP zero shot, in which case the clip_arch and dataset must be specified.
"""
import argparse
import json
import types
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional

import eval.utils
import utils
from eval.utils import clip_label_embeddings
from eval.utils import eval_features
from loaders import check_dataset, get_ood
from model_builders import available_models, load_embeds, load_embed_stats
from utils import apply_batchwise


def load_all_embeds(arch, dataset, out_dists, norm=False, l2norm=False):
    train_features, train_labels = load_embeds(arch=arch, dataset=dataset,
                                               with_label=True, test=False)
    mean, std = load_embed_stats(arch=arch, dataset=dataset, test=False)
    test_features, test_labels = load_embeds(arch=arch, dataset=dataset,
                                             with_label=True, test=True)

    test_features_ood = {f"test_features_ood_{out_dist}":
                             load_embeds(arch=arch, dataset=out_dist,
                                         with_label=False, test=True)
                         for out_dist in out_dists}

    out = dict(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        **test_features_ood
    )
    feat_keys = [k for k in out.keys() if 'feat' in k]
    for k in feat_keys:
        if norm:
            out[k] = (out[k] - mean) / std
        if l2norm:
            out[k] = functional.normalize(out[k], dim=-1)
    return out


def get_probing_logits_fn(args):
    args.probing_path = args.probing_path.expanduser()
    with open(args.probing_path / 'hp.json', 'r') as f:
        probing_args = json.load(f)
    args.dataset = probing_args['dataset']
    arch = probing_args['arch']
    embed_norm = probing_args['embed_norm']
    l2_norm = probing_args.get("l2_norm", False)
    embeds = load_all_embeds(arch=arch, dataset=args.dataset, out_dists=args.out_dists, norm=embed_norm, l2norm=l2_norm)
    if probing_args.get('pseudo_labels', False):
        labels = torch.load(args.probing_path / 'pseudo_labels.pth')
        print('Loaded pseudo labels from probing path')
        embeds['train_labels'] = labels

    st_dict = torch.load(args.probing_path / 'model.pth', map_location='cpu')
    lin_layer = nn.Linear(*st_dict['weight'].T.shape).eval().cuda()
    lin_layer.load_state_dict(st_dict, strict=True)

    return lin_layer, embeds


def get_clip_logits_fn(args):
    label_embeds, proj = clip_label_embeddings(arch=args.clip_arch, dataset=args.dataset, device='cuda',
                                               logit_scale=False, context_prompts=args.clip_prompt)
    embeds = load_all_embeds(arch=args.clip_arch, dataset=args.dataset, out_dists=args.out_dists)

    def compute_logits(x):
        x = functional.normalize(x @ proj, dim=-1)
        return x @ label_embeds.T

    return compute_logits, embeds


def get_logits(args):
    if args.probing_path is not None:
        logit_fn, embeds = get_probing_logits_fn(args)
    else:
        logit_fn, embeds = get_clip_logits_fn(args)

    out = {}
    for k, v in embeds.items():
        if 'feat' in k:
            v = apply_batchwise(logit_fn, v, device='cuda')
        out[k] = v
    return out


def _check_clip_arch(arch):
    if arch in available_models('*clip*'):
        return arch
    raise argparse.ArgumentTypeError(f'{arch} is not a valid CLIP architecture')


def get_args():
    parser = get_args_parser()
    args = parser.parse_args()
    if args.clip_arch is not None and args.out_dir is None:
        print('WARNING: No out_dir specified, results will not be saved')
    if args.clip_arch is not None and args.dataset is None:
        raise ValueError('dataset must be specified when using CLIP zero shot')
    if args.probing_path is not None and args.out_dir is None:
        args.out_dir = args.probing_path
    return args


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="IN1K", type=check_dataset, help='dataset to use')
    parser.add_argument('--out_dists', type=check_dataset, default=None, nargs='*',
                        help='out of distribution datasets to use. If not set will use default datasets')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[1.0],
                        help='temperatures to use')
    parser.add_argument('--eval_maha', type=utils.bool_flag, default=False,
                        help="""Calculate Mahalanobis OOD scores""")
    parser.add_argument('--out_dir', type=Path, help='directory to save results to')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--probing_path', type=Path,
                       help='path to probing results')
    group.add_argument('--clip_arch', type=_check_clip_arch,
                       help='CLIP architecture to use')
    parser.add_argument('--clip_prompt', default=None, type=str, nargs='*',
                        help='CLIP prompt to use. By default, uses ensemble of 5 prompts.')
    return parser


def main():
    args = get_args()
    args.out_dists = get_ood(args.dataset) if args.out_dists is None else args.out_dists
    print(f"Using OOD datasets {args.out_dists}")
    args.eval_maha = False if args.dataset=="IN1K" else True
    data = get_logits(args)
    eval_args = types.SimpleNamespace(
        eval_ood=True,
        eval_cluster=False,
        eval_ood_maha=args.eval_maha,
        eval_knn_acc=False,
        eval_ood_knn=False,
        eval_ood_norm=False,
        eval_ood_logits=True,
        ood_logits_temp=args.temperatures,
        dataset=args.dataset
    )
    results = []
    for out_dist in args.out_dists:
        print('Out dist', out_dist)
        result = eval_features(
            args=eval_args,
            epoch=None,
            writer=None,
            test_features=data['test_features'],
            test_features_ood=data[f'test_features_ood_{out_dist}'],
            train_features=data['train_features'],
            train_labels=data['train_labels'],
            test_labels=data['test_labels']
        )
        empty_res = [k for k, v in result.items() if not v]
        for k in empty_res:
            result.pop(k)
        result['out_dist'] = out_dist
        result.move_to_end('out_dist', last=False)
        results.append(result)

    df = pd.DataFrame([eval.utils.flatten_result(r) for r in results])
    df = df.set_index('out_dist')

    if args.out_dir is not None:
        args.out_dir = args.out_dir.expanduser()
        args.out_dir.mkdir(exist_ok=True, parents=True)
        with open(args.out_dir / "eval_args.json", 'w') as f:
            args_d = vars(args).copy()
            args_d.pop('out_dir')
            if args.probing_path is not None:
                args_d['probing_path'] = str(args.probing_path)
            json.dump(args_d, f, indent=2)
        with open(args.out_dir / "ood_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)

        df.round(2).to_csv(args.out_dir / "ood_metrics.csv")


if __name__ == '__main__':
    main()
