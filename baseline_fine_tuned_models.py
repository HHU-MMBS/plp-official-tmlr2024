"""
Script to evaluate OOD performance for timm imagenet-finetuned models
How to run: 
conda activate plp && python baseline_fine_tuned_models.py --dataset IN1K --archs convnext_base_in22ft1k
"""
import argparse
import json
import types
from pathlib import Path

import pandas as pd
import torch
import timm
from torchvision import transforms

from eval.ood_scores import free_energy,msp
from eval.binary_metrics import auroc_score, fpr95_score
from loaders import  get_dataset, get_ood
from gen_embeds import compute_embedding
from linear_probing import *


def get_transforms(size):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    resize_size = int(256 * size / 224)
    resizes = [
                transforms.Resize(size=(resize_size,resize_size)),
                transforms.CenterCrop(size)
            ]
    transform = transforms.Compose([
            *resizes,
            transforms.ToTensor(),
            normalize ])
    return transform


def get_args_prob():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--outdir', type=str, default=None , help='out folder')
    parser.add_argument('--dataset',   default="IN1K", choices=available_datasets(), help='Dataset to use.')
    parser.add_argument('--ood', default=None, choices=available_datasets(), nargs='+', help='OOD datasets to use.')
    parser.add_argument('--archs',   default=["convnext_base_in22ft1k"], nargs='+')
    parser.add_argument('--batch_size', type=int, default=128, help="""Value for batch size.""")
    args = parser.parse_args()
    args.ood = get_ood(args.dataset) if args.ood is None else args.ood
    return args

@torch.no_grad()
def main():
    args = get_args_prob()
    temp = 1
    train = False
    results = []
    for arch in args.archs:
        results_arch = []
        if "384" in arch:
            size = 384
        elif "256" in arch:
            size = 256
        else:
            size = 224
        print(f"\n\n {arch}, {size}\n\n")
        transf = get_transforms(size)
        if args.outdir is not None:
            out_dir = Path(args.outdir).expanduser().resolve()
        else:
            out_dir = Path(f'./experiments/fine-tuned-IN1K-models-eval/{args.dataset}/{arch}').expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        print("Loading timm model with name:", arch)
        model = timm.create_model(
            arch,
            in_chans=3,
            num_classes=1000,
            pretrained=True).cuda()
        model = model.eval()
        indist_embeds = []
        dset = get_dataset(args.dataset, train=train, transform=transf)
        dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=False, num_workers=8)
        embeddings, _ = compute_embedding(model, dataloader)    
        embeddings = embeddings.squeeze()
        indist_embeds.append(embeddings)    
    
        for ood_data in args.ood:
            dset = get_dataset(ood_data, train=train, transform=transf)
            dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=False, num_workers=8)
            ood_embeddings, _ = compute_embedding(model, dataloader)    
            embeddings = embeddings.squeeze()
            
            for score, method in zip([free_energy, msp], ["free_energy", "msp"]):
                scores_in = score(indist_embeds[-1], temp)
                scores_out = score(ood_embeddings, temp)
                scores = torch.cat((scores_in, scores_out))
                
                num_in = int(indist_embeds[-1].shape[0])
                num_out = int(ood_embeddings.shape[0])
                labels = torch.cat((torch.ones(num_in), torch.zeros(num_out))).numpy()
                
                auroc = auroc_score(labels, scores) * 100
                fpr95 = fpr95_score(labels, scores) * 100
                results.append(dict(
                    indist=args.dataset,
                    ood=ood_data,
                    auroc=auroc,
                    fpr=fpr95,
                    method=method,
                    model=arch,
                ))
                
                results_arch.append(dict(
                    indist=args.dataset,
                    ood=ood_data,
                    auroc=auroc,
                    fpr=fpr95,
                    method=method,
                    model=arch,
                ))
        # save per model
        df = pd.DataFrame(results_arch)
        df.round(2).to_csv(out_dir / f'{arch}-fine-tuned-{args.dataset}.csv')

if __name__ == '__main__':
    main()
