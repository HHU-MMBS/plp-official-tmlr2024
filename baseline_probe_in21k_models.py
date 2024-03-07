"""
How to run: 
export CUDA_VISIBLE_DEVICES=1 && conda activate plp && python baseline_probe_in21k_models.py --dataset IN1K --archs convnext_base_in22k
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
from loaders import get_dataset,  get_ood
from gen_embeds import compute_embedding
from linear_probing import *

def get_args_prob():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dataset',   default="IN1K", choices=available_datasets(), help='Dataset to use.')
    parser.add_argument('--batch_size', type=int, default=1024, help="""Value for batch size.""")
    parser.add_argument('--ood',   default=None, choices=available_datasets(), nargs='+', help='OOD datasets to use.')
    parser.add_argument('--archs',   default=["vit_base_patch16_224_in21k"], nargs='+')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="""Value for learning rate. Linearly scaled with batch size / 256""")
    parser.add_argument('--wd', type=float, default=1e-3, help="""Value for weight decay.""")
    parser.add_argument('--optimizer', default='Adam', type=str, help="""Optimizer to use.""")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--num_epochs', type=int, default=100, help="""Number of epochs to train for.""")
    group.add_argument('--num_steps', type=int, help="""Number of steps to train for.""")
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help="""Whether to overwrite output directory.""")
    return parser.parse_args()

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


def main():
    args = get_args_prob()
    args.ood = get_ood(args.dataset) if args.ood is None else args.ood
    temp = 1
    results = []
    for arch in  args.archs:
        results_arch = []
        if "384" in arch:
            size = 384
        elif "256" in arch:
            size = 256
        else:
            size = 224
        print(f"\n\n {arch}, {size}\n\n")
        transf = get_transforms(size)
        out_dir = Path(f'./experiments/prob-in21k-models/{args.dataset}/{arch}').expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        backbone = timm.create_model(
            arch,
            in_chans=3,
            num_classes=-1,
            pretrained=True).cuda()
        backbone = backbone.eval()
        dset = get_dataset(args.dataset, train=True, transform=transf)
        dataloader = torch.utils.data.DataLoader(dset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=False, num_workers=8)
        embeddings, labels = compute_embedding(backbone, dataloader)    
        embeddings = embeddings.squeeze()
        
        # Train head!
        backbone = backbone.cpu()
        train_dset = TensorDataset(embeddings, labels)
        head = train_head(train_dset, "cuda", args)
        
        # Inference
        dataloader_head = torch.utils.data.DataLoader(train_dset, batch_size=1024, shuffle=False, drop_last=False, pin_memory=False, num_workers=8)
        indist_preds , _ = compute_embedding(head, dataloader_head)  

        for ood_data in args.ood:
            backbone = backbone.cuda()
            model = nn.Sequential(backbone, head)
            dset = get_dataset(ood_data, train=False, transform=transf)
            dataloader = torch.utils.data.DataLoader(dset, batch_size=1024, shuffle=False, drop_last=False, pin_memory=False, num_workers=8)
            ood_preds, _ = compute_embedding(model, dataloader)    
            
            for score, method in zip([free_energy, msp], ["free_energy", "msp"]):
                scores_in = score( indist_preds, temp)
                scores_out = score(ood_preds, temp)
                scores = torch.cat((scores_in, scores_out))
                
                num_in = int( indist_preds.shape[0])
                num_out = int(ood_preds.shape[0])
                labels = torch.cat((torch.ones(num_in), torch.zeros(num_out))).numpy()
                
                auroc = auroc_score(labels, scores) * 100
                fpr95 = fpr95_score(labels, scores) * 100
                results.append(dict(
                    indist=args.dataset,
                    ood=args.ood,
                    auroc=auroc,
                    fpr=fpr95,
                    method=method,
                    model=arch,
                ))
                
                results_arch.append(dict(
                    indist= args.dataset,
                    ood=ood_data,
                    auroc=auroc,
                    fpr=fpr95,
                    method=method,
                    model=arch,
                ))
        # save per model
        df = pd.DataFrame(results_arch)
        df.round(2).to_csv(out_dir / f'{arch}-fine-tuned-{ args.dataset}.csv')
if __name__ == '__main__':
    main()