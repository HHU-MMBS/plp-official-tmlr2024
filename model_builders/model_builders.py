import fnmatch
import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional, Union

import timm
import torch
from torchvision import models as torchvision_models, transforms

from model_builders.model_utils import get_embed_dim, remove_prefix, split_normalization
from model_builders.multihead_backbone import MultiCropClip
from model_builders.openclip import _get_openclip_modelnames, build_openclip_visual
from model_builders.timm_utils import _get_timm_model, _get_timm_modelnames
from model_builders.vision_transformer import DINOHead

_PRECOMPUTED_PATH = '/home/shared/embeddings'
_AVAILABLE_MODELS = ( 
    "mae_vit_base",
    "dino_resnet50",
    "dino_vits16",
    "dinov2_vits14",
    "dinov2_vitb14",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "msn_vit_small",
    "msn_vit_base",
    "dinov2_vitl14",
    "dinov2_vitg14",
    "dino_vitb16",
    "msn_vit_large",
    "mocov3_vit_base",
    "mae_vit_large",
    "mae_vit_huge",
    "ibot_vit_large",
    "ibot_vit_large_in21k",
    "beit_vit_large_in21k",
    "beitv2_vit_large",
    *['timm_' + s for s in _get_timm_modelnames()],
    *['openclip_' + s for s in _get_openclip_modelnames()],
)


def available_models(pattern=None):
    if pattern is None:
        return _AVAILABLE_MODELS
    return tuple(fnmatch.filter(_AVAILABLE_MODELS, pattern))

def default_prepro(size=224, dataset=''):
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if 'IN' in dataset:
        resize_size = int(256 * size / 224)
        resizes = [
            transforms.Resize(size=(resize_size,resize_size)),
            transforms.CenterCrop(size)
        ]
    else:
        resizes = [transforms.Resize(size=(size,size), interpolation=transforms.InterpolationMode.BICUBIC)]
    return transforms.Compose([
        *resizes,
        transforms.ToTensor(),
        normalize
    ])

def load_model(config, head=True, split_preprocess=False):
    """
    config/args file
    head=False returns just the backbone for baseline evaluation
    split_preprocess=True returns resizing etc. and normalization/ToTensor as separate transforms
    """
    from main_args_temi import set_default_args
    config = set_default_args(config)

    if config.precomputed:
        backbone = config.arch
        preprocess = None
        normalize = None
    else:
        weights = config.backbone_weights if hasattr(config, "backbone_weights") else None
        backbone, preprocess = load_backbone(config.arch, weights=weights)
        print(f"Backbone {config.arch} loaded.")

    if not config.precomputed and preprocess is None:
        preprocess = default_prepro(config.vit_image_size, config.dataset)
        
    if head:
        if getattr(config, "embed_dim", None) is None:
            config.embed_dim = get_embed_dim(config, backbone)
        # Just get everything via reflection
        mmc_params = inspect.signature(MultiCropClip).parameters
        mmc_args = {k: v for k, v in config.__dict__.items() if k in mmc_params}
        model = MultiCropClip(backbone, **mmc_args)

        if config.embed_norm:
            model.set_mean_std(*load_embed_stats(config, test=False))
        print("Head loaded.")
    else:
        model = backbone.float()

    if split_preprocess:
        if not config.precomputed:
            preprocess, normalize = split_normalization(preprocess)
        return model, preprocess, normalize
    return model, preprocess


def load_backbone(arch, weights=None):
    preprocess = None
    if "timm" in arch:  # timm models
        arch = remove_prefix(arch, 'timm')
        backbone = timm.create_model(arch, pretrained=True, in_chans=3, num_classes=0)
    elif "convnext" in arch and not "openclip" in arch:
        backbone = getattr(torchvision_models, arch)(pretrained=True)
        backbone.classifier = torch.nn.Flatten(start_dim=1, end_dim=-1)
    elif arch in torchvision_models.__dict__:  # torchvision models
        backbone = torchvision_models.__dict__[arch](num_classes=0)
    elif "swag" in arch:
        arch = remove_prefix(arch, 'swag')
        backbone = torch.hub.load("facebookresearch/swag", model=arch)
        backbone.head = None
    elif "dinov2" in arch:
        backbone = torch.hub.load('facebookresearch/dinov2', arch)
    elif "dino" in arch:
        arch = arch.replace("-", "_")
        backbone = torch.hub.load('facebookresearch/dino:main', arch)
    elif "clip" in arch:  # load clip vit models from openai
        backbone, preprocess = build_openclip_visual(arch)
    elif "mae" in arch or "msn" in arch or "mocov3" or "ibot" in arch or "beit" in arch:
        backbone = _get_timm_model(arch)
    else:
        print(f"Architecture {arch} non supported")
        sys.exit(1)
    if preprocess is None:
        preprocess = default_prepro(224)
    if weights is not None:
        print("Loading weights for BACKBONE!")
        msg = backbone.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
        print(msg)
    return backbone, preprocess


def _build_from_config(
        precomputed: bool,
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    if isinstance(config, str) or isinstance(config, Path):
        p = Path(config)
        with open(p, "r") as f:
            config = json.load(f)
        config = Namespace(**config)
    if config is None:
        config = Namespace()
    config.num_heads = 1
    config.precomputed = precomputed
    if ckpt_path is not None:
        # Don't reload norms
        config.embed_norm = False

    d = None
    if ckpt_path is not None:
        d = torch.load(ckpt_path, map_location="cpu")
        if 'teacher' in d:
            d = d['teacher']
        if 'head.best_head_idx' in d:
            best_head_idx = d['head.best_head_idx']
            d2 = {k: v for k, v in d.items() if k in ('embed_mean', 'embed_std')}
            d2['head.best_head_idx'] = torch.tensor(0)
            for k, v in d.items():
                if k.startswith(f'head.heads.{best_head_idx}.'):
                    k = 'head.heads.0.' + k[len(f'head.heads.{best_head_idx}.'):]
                    d2[k] = v
            d = d2
        else:
            d['head.best_head_idx'] = torch.tensor(0)
        config.embed_dim = d['head.heads.0.mlp.0.weight'].size(1)

    model, _ = load_model(config, head=True)
    model.eval()
    if d is not None:
        model.load_state_dict(d, strict=False)

    return model


def build_head_from_config(
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    """
    config: Either path to hp.json or config namespace
    ckpt_path: Path to checkpoint
    """
    return _build_from_config(True, config, ckpt_path)


def build_model_from_config(
        config: Optional[Union[str, Path, Namespace]] = None,
        ckpt_path: Optional[Union[str, Path]] = None):
    """
    config: Either path to hp.json or config namespace
    ckpt_path: Path to checkpoint
    """
    return _build_from_config(False, config, ckpt_path)


def load_embeds(config=None,
                arch=None,
                dataset=None,
                test=False,
                norm=False,
                datapath=_PRECOMPUTED_PATH,
                with_label=False,
                label_key="label_split"):
    p, test_str = _embedding_path(config, arch, dataset, test, datapath)
    emb = torch.load(p / f'embeddings{test_str}.pt', map_location='cpu')
    if norm:
        # Always use train stats for normalization
        mean, std = load_embed_stats(config, arch, dataset, test=False, datapath=datapath)
        emb = (emb - mean) / std
    if not with_label:
        return emb
    label = torch.load(p / f'label{test_str}.pt', map_location='cpu')
    if isinstance(label, dict):
        if label_key in list(label.keys()):
            label = label[label_key]
        elif test and label_key not in list(label.keys()):
            label = label['label']
        else:
            raise ValueError(f"Label key {label_key} not found in {list(label.keys())}")            
    return emb, label


def _embedding_path(config, arch, dataset, test, datapath):
    assert bool(config) ^ bool(arch and dataset)
    if config:
        arch = config.arch
        dataset = config.dataset
    import gen_embeds
    test_str = '-test' if test else ''
    p = gen_embeds.get_outpath(arch, dataset, datapath)
    return p, test_str


def load_embed_stats(
        config=None,
        arch=None,
        dataset=None,
        test=False,
        datapath=_PRECOMPUTED_PATH):
    p, test_str = _embedding_path(config, arch, dataset, test, datapath)
    mean = torch.load(p / f'mean{test_str}.pt', map_location='cpu')
    std = torch.load(p / f'std{test_str}.pt', map_location='cpu')
    return mean, std

class HeadEmbed(DINOHead):
    def __init__(self, apply_norm=False, **kwargs):
        super().__init__(**kwargs)
        in_dim = kwargs["in_dim"]
        self.register_buffer("embed_mean", torch.zeros(in_dim))
        self.register_buffer("embed_std", torch.ones(in_dim))
        self.apply_norm = apply_norm
    def forward(self, x):
        if self.apply_norm:
            x = (x - self.embed_mean) / self.embed_std
        return super().forward(x)

@torch.no_grad()
def build_temi_head(hp, ckpt_path, embed_dim=None, apply_norm=True, load_weights=True, verbose=True):
    """
    Given a checkpoint path and, build a TEMI head with the same architecture as the checkpoint.
    """
    if isinstance(hp, str) or isinstance(hp, Path):
        hp = json.load(open(hp, "r"))
    if isinstance(hp, Namespace):
        hp = vars(hp)

    head_args = dict(
            in_dim=embed_dim if embed_dim is not None else hp["embed_dim"],
            out_dim=hp["out_dim"],
            use_bn=hp["use_bn_in_head"],
            dropout_p=hp["head_dropout_prob"],
            nlayers=hp["nlayers"],
            hidden_dim=hp["hidden_dim"],
            norm_last_layer=hp["norm_last_layer"],
            bottleneck_dim=hp["bottleneck_dim"]
        )
    head = HeadEmbed(apply_norm=apply_norm, **head_args)
    if load_weights:
        d = torch.load(ckpt_path, map_location=torch.device('cpu'))["teacher"]    
        best_head_idx = d['head.best_head_idx']
        d2 = {k: v for k, v in d.items() if k in ('embed_mean', 'embed_std')}
        for k, v in d.items():
            if k.startswith(f'head.heads.{best_head_idx}.'):
                k = k[len(f'head.heads.{best_head_idx}.'):]
                d2[k] = v
        
        msg = head.load_state_dict(d2, strict=False)
        if verbose:
            print("\n\n TEMI clustering head loaded with message \n", msg)
    return head
