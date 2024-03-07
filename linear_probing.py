"""
Train a linear classifier on top of precomputed embeddings,
optionally using CLIP pseudo labels for training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import utils
from eval.utils import clip_label_embeddings
from loaders import available_datasets, get_num_classes
from model_builders import available_models, load_embeds


def train_all_steps(model, train_loader, optimizer, scheduler, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_avg = utils.SmoothedValue(window_size=100)
    num_steps = len(scheduler)

    it = iter(train_loader)
    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader)
                x, y = next(it)

            optimizer.zero_grad(set_to_none=True)
            x = x.to(device)
            y = y.to(device)
            for param_group in optimizer.param_groups:
                param_group["lr"] = scheduler[step]
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())
            pbar.set_postfix(loss=loss_avg.avg)


def train_head(train_dset, device, args):
    embed_dim = train_dset[0][0].size(0)
    model = nn.Linear(embed_dim, get_num_classes(args.dataset))
    optimizer = get_optimizer(model.parameters(), args)
    model.to(device)

    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    if args.num_steps is None:
        steps_per_epoch = len(train_loader)
        args.num_steps = steps_per_epoch * args.num_epochs

    lr_schedule = utils.cosine_scheduler_iter(
        base_value=args.lr,
        final_value=0,
        total_iters=args.num_steps,
        warmup_fraction=0.1,
        start_warmup_value=0)

    print('Start training')
    train_all_steps(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=lr_schedule,
        device=device,
    )
    print('Finished training')
    return model


@torch.no_grad()
def validate(model, val_loader, device):
    accs = []
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=-1)
        accs.append(preds == y)
    avg_acc = torch.cat(accs).float().mean().item()
    return avg_acc


def few_shot_mask(labels, num_classes, num_shots):
    mask = torch.zeros_like(labels).bool()
    for i in range(num_classes):
        idx = np.random.choice((labels == i).nonzero().flatten(), num_shots, replace=False)
        mask[idx] = True
    return mask


def get_embed_dataset(args):
    embed_args = dict(
        arch=args.arch,
        dataset=args.dataset,
        norm=args.embed_norm,
        with_label=True
    )
    if args.datapath is not None:
        embed_args['datapath'] = args.datapath
    x_train, y_train = load_embeds(**embed_args, test=False)

    if args.pseudo_labels:
        pseudo_probs = clip_pseudo_probs(args)
        y_train = pseudo_probs.argmax(dim=-1)
        torch.save(y_train, args.output_dir / "pseudo_labels.pth")

    if args.num_shots is not None:
        mask = few_shot_mask(y_train, get_num_classes(args.dataset), args.num_shots)
        x_train = x_train[mask]
        y_train = y_train[mask]
    x_val, y_val = load_embeds(**embed_args, test=True)
    train_dset = TensorDataset(x_train, y_train)
    val_dset = TensorDataset(x_val, y_val)
    return train_dset, val_dset


def get_optimizer(params, args):
    lr = args.lr * args.batch_size / 256
    name = args.optimizer.lower()
    if name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=args.wd)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.wd)
    elif name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=args.wd, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer {name}")


@torch.no_grad()
def clip_pseudo_probs(args):
    text_emb, proj = clip_label_embeddings(arch=args.arch, dataset=args.dataset, context_prompts=args.pseudo_prompt)
    # Load embeddings again to circumvent normalization issues
    train_x = load_embeds(arch=args.arch, dataset=args.dataset, norm=False, test=False)
    logits = []
    for xi in DataLoader(train_x, batch_size=args.batch_size, shuffle=False):
        xi = nn.functional.normalize(xi.to(text_emb.device), dim=-1)
        if proj is not None:
            xi = xi @ proj
        logits.append((xi @ text_emb.T).cpu())
    return torch.cat(logits, dim=0).softmax(dim=-1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--datapath', type=str, help='Path to precomputed embeddings')
    parser.add_argument('--output_dir', type=str, help='Path to save model checkpoints', required=True)
    parser.add_argument('--dataset', required=True, choices=available_datasets(), help='Dataset to use.')
    parser.add_argument('--arch', required=True, choices=available_models(), help='Architecture to use.')
    parser.add_argument('--batch_size', type=int, default=256, help="""Value for batch size.""")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="""Value for learning rate. Linearly scaled with batch size / 256""")
    parser.add_argument('--wd', type=float, default=1e-3, help="""Value for weight decay.""")
    parser.add_argument('--embed_norm', default=True, type=utils.bool_flag, help="""Whether to normalize embeddings.""")
    parser.add_argument('--optimizer', default='Adam', type=str, help="""Optimizer to use.""")
    parser.add_argument('--num_shots', type=int, default=None, help='Number of shots for few-shot learning.')
    parser.add_argument('--pseudo_labels', default=False, type=utils.bool_flag,
                        help='Whether to use pseudo labels for training. Only for CLIP models.')
    parser.add_argument('--pseudo_prompt', type=str, nargs='*',
                        help='Prompt to use for pseudo labels. Only for CLIP models. By default, uses ensemble of 5 prompts.')

    # Mutually exclusive for length of training, either number of epochs or number of steps
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--num_epochs', type=int, help="""Number of epochs to train for.""")
    group.add_argument('--num_steps', type=int, help="""Number of steps to train for.""")

    parser.add_argument('--overwrite', action='store_true', default=False,
                        help="""Whether to overwrite output directory.""")
    return parser.parse_args()


def main():
    args = get_args()
    if args.seed is None:
        args.seed = np.random.randint(0, 2**32)
    utils.fix_random_seeds(args.seed)
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_result_path = args.output_dir / "val_acc.txt"
    if val_result_path.exists() and not args.overwrite:
        raise ValueError(f"Validation result file {val_result_path} already exists. "
                         f"Use --overwrite to overwrite.")
    with open(args.output_dir / "hp.json", 'wt') as f:
        json.dump(vars(args), f, indent=4, default=str)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dset, val_dset = get_embed_dataset(args)
    model = train_head(train_dset, device, args)
    torch.save(model.state_dict(), args.output_dir / "model.pth")

    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)
    val_acc = validate(model, val_loader, device)
    print(f"Validation accuracy: {val_acc:.4f}")
    with open(val_result_path, 'wt') as f:
        f.write(f"{val_acc:.4f}")


if __name__ == "__main__":
    main()
