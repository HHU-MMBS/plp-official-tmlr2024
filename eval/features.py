from collections import defaultdict

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import utils
from loaders import get_dataset
from model_builders import load_model, load_embeds

def get_preprocess(args):
    init_val = args.precomputed
    args.precomputed = False
    _, prepro = load_model(args, head=False)
    args.precomputed = init_val
    return prepro

class FeatureExtractionPipeline:
    def __init__(self, args, cache_backbone=False, model=None, transform=None, data_parallel=False):
        if not args.head and args.train_backbone:
            cache_backbone = False
        if not args.head and cache_backbone:
            raise ValueError("head must be True if cache_backbone is True")
        self.args = args
        self.precompute_arch = args.arch if args.precomputed else None
        self.cache_backbone = cache_backbone
        self.embeds, self.labels = {}, {}
        
        if hasattr(args, 'batch_size_per_gpu') and hasattr(args, 'batch_size'):
            self.batch_size = args.batch_size_per_gpu or args.batch_size
        else:
            self.batch_size = args.batch_size
        
        # Loading model from args
        if model is None:
            if args.head is False and args.train_backbone is True:
                print("\n\n Warning: Backbone only evaluation. No head is loaded. \n\n")
            elif args.head is False and args.train_backbone is False:
                raise ValueError("head must be True if train_backbone is False")
              
            self.model, self.transform = load_model(self.args, head=args.head)
            self.model.cuda().eval()
            if self.transform is None and not args.precomputed:
                self.transform = get_preprocess(args)
        # Using specified model and transform
        else:
            # alternative to passing the args
            self.model = model.cuda().eval()
            self.transform = transform
            args.precomputed = False
        
        if not self.cache_backbone and data_parallel:
            self.model = nn.DataParallel(self.model)        
        
    def get_labels(self, dataset, save_key=None):
        if hasattr(dataset, 'targets'):
            return np.array(dataset.targets, dtype=np.int64)
        elif hasattr(dataset, 'labels'):
            return np.array(dataset.labels, dtype=np.int64)
        elif save_key is not None:
            if save_key in self.labels.keys():
                return self.labels[save_key]
            else:
                raise ValueError("dataset does not have labels")
        else:
            raise ValueError("dataset does not have labels")

    @property
    def cached(self):
        return len(self.embeds)>0 and not self.args.train_backbone and self.cache_backbone
    
    def get_dataloader(self, dataset, train=False, gen_split="all"):
        pseudo_path = self.args.pseudo_labels
        if not train and self.args.pseudo_labels is not None:
            pseudo_path = self.args.pseudo_labels.replace('train', 'val')
            
        dataset = get_dataset(dataset, train=train,
                                      download=True, transform=self.transform,
                                      precompute_arch=self.precompute_arch,
                                      pseudo_path=pseudo_path, gen_split=gen_split)
        
        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False)
        
        return data_loader
    
    def load_ckpt(self, pretrained_weights=None):
        if pretrained_weights is not None:
            if isinstance(self.model, nn.DataParallel):
                module = self.model.module
            else:
                module = self.model
            if self.args.lin_eval:
                utils.load_pretrained_weights(module,
                                          pretrained_weights,
                                          self.args.checkpoint_key,
                                          head=True,
                                          head_only=False)
            else:
                utils.load_pretrained_weights(module,
                                          pretrained_weights,
                                          self.args.checkpoint_key,
                                          head=self.args.head,
                                          head_only=self.cached)
        self.model.eval()
        return self.model
                   
    @torch.no_grad()
    def get_logits(self, data_loader, pretrained_weights=None, save_key=None):
        if pretrained_weights is not None:
            self.load_ckpt(pretrained_weights=pretrained_weights)
        feats = None
        if not self.cache_backbone:
            feats = extract_features(self.model, data_loader, self.args.head)
        else:
            embeds = self.get_embeds(data_loader, save_key=save_key)
            feats = self.head_logits(embeds)
        labels = self.get_labels(data_loader.dataset)
        return feats, labels
    
    @torch.no_grad()
    def get_train_logits(self, pretrained_weights=None, return_feats=False):
        if pretrained_weights is not None:
            self.load_ckpt(pretrained_weights=pretrained_weights)
        # Get Dataloaders
        self.data_loader_train =  self.get_dataloader(self.args.dataset, train=True)
        self.data_loader_val = self.get_dataloader(self.args.dataset, train=False)
        self.dataset_labels = self.get_labels(self.data_loader_train.dataset)
        self.test_labels = self.get_labels(self.data_loader_val.dataset)
        # Get features  
        if not self.cache_backbone:
            print("Warning: NOT cached backbone. Extracting features...")
            train_features = extract_features(self.model, self.data_loader_train, self.args.head)
            test_features = extract_features(self.model, self.data_loader_val, self.args.head)
            if return_feats:
                return train_features, test_features, self.dataset_labels, self.test_labels
        else:
            embeds = self.get_embeds()
            if return_feats:
                return embeds['train'], embeds['test'], self.dataset_labels, self.test_labels
            train_features = self.head_logits(embeds['train'])
            test_features = self.head_logits(embeds['test'])
        return train_features, test_features, self.dataset_labels, self.test_labels
    
    @torch.no_grad()
    def head_logits(self, embeds):
        """
        Processes all the embeds of the dataset in one go
        """
        step_size = 100000
        if isinstance(embeds, list):
            embeds = embeds[0]
        if embeds.shape[0]<step_size:
            out = self.model.head(embeds.cuda()).cpu()
        else:
            out = []
            for i in range(0, embeds.shape[0], step_size):
                out.append(self.model.head(embeds[i:i+step_size].cuda()).cpu())
            out = torch.cat(out)
        return out
    
    def backbone_embed_dataloader(self, dataloader):
        embeds = []
        labels = []
        for datapoint in tqdm(dataloader):
            samples = datapoint[0]
            lab = datapoint[-1]
            labels.append(lab)
            if not isinstance(samples, list):
                samples = [samples]
                samples = torch.cat([im.cuda(non_blocking=True) for im in samples])
                output = self.model.backbone_embed(samples)
                embeds.append(output.cpu())
        return torch.cat(embeds).cpu(), torch.cat(labels).cpu()

    @torch.no_grad()
    def get_embeds(self, dataloader=None, save_key=None):
        if dataloader is None and save_key is None:
            # default behavior
            keys = ['train', 'test']
            embeds_local = {}
            if len(self.embeds)>0 and keys[0] in self.embeds.keys() and keys[1] in self.embeds.keys() and self.args.train_backbone is False:
                return self.embeds 
            for k, loader in zip(keys,
                                [self.data_loader_train, self.data_loader_val]):
                emb, labels = self.backbone_embed_dataloader(loader) 
                embeds_local[k] = emb
                self.labels[k] = labels
            return embeds_local
        
        if save_key is not None and save_key in self.embeds.keys() and self.args.train_backbone is False:
            print('Warning: Using cached embedding with dataloader!')
            return self.embeds[save_key]
        # compute backbone embeds for this dataloader
        emb, labels = self.backbone_embed_dataloader(dataloader)
        # store embeds
        if save_key is not None and save_key not in self.embeds.keys() and self.args.train_backbone is False:
            self.embeds[save_key] = emb
            self.labels[save_key] = labels
        return emb

@torch.no_grad()
def extract_features(model, data_loader, head):
    features = []
    for (samples, _) in tqdm(data_loader):
        if not isinstance(samples, list):
            samples = [samples]
        samples = torch.cat([im.cuda(non_blocking=True) for im in samples])
        if head:
            try:
                feats, _ = model(samples)
            except Exception:
                feats = model(samples)
        else:
            feats = model(samples)
        features.append(feats.cpu())
    return torch.cat(features, dim=0)