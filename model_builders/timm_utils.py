""" Functions for models using the timm library. """
import os
from pathlib import Path

import timm
import torch
from timm.models.helpers import load_state_dict

from model_builders.model_utils import _download

def _get_timm_modelnames():
    return [
    "resnet50",
    "vit_small_patch16_224",
    "vit_small_patch16_224.augreg_in21k",
    "vit_base_patch16_224",
    "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224",
    'vit_large_patch16_224_in21k',
    'vit_huge_patch14_224_in21k',
    "vit_base_r50_s16_224_in21k",
    "convnext_base_in22k",
    'vit_large_patch16_224_in21k',
    'vit_huge_patch14_224_in21k',
    # 'beitv2_base_patch16_224',
    # 'beitv2_base_patch16_224_in22k',
    # 'beitv2_large_patch16_224',
    # 'beitv2_large_patch16_224_in22k'
    ]


def _load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    msg = model.load_state_dict(state_dict, strict=strict)
    print(msg)


_dict_models_urls = {
    "msn": {
        'vit_small_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar',
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vitb16_600ep.pth.tar',
        'vit_large_patch16_224': 'https://dl.fbaipublicfiles.com/msn/vitl16_600ep.pth.tar',
        "key": 'target_encoder'
    },
    "mae": {
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',
        'vit_large_patch16_224': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth',
        'vit_huge_patch14_224_in21k': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth',
        "key": 'model'
    },

    "mocov3": {
        'vit_small_patch16_224': 'https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar',
        'vit_base_patch16_224': 'https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar',
        "key":  "state_dict"
    },
    "ibot": {
        'vit_large_patch16_224': 'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_rand_mask/checkpoint_teacher.pth',
        'vit_large_patch16_224_in21k': 'https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16_pt22k/checkpoint_student.pth',
        "key": "state_dict"
    },
    "beit": {
       'vit_large_patch16_224_in21k': "https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_large_patch16_224_pt22k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D",
        "key": "model"
    },
    "beitv2": {
       'vit_large_patch16_224': "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D",
        "key": "model"
    },
}
_dict_timm_names = {
        "vit_huge": 'vit_huge_patch14_224_in21k',
        "vit_large": 'vit_large_patch16_224',
        "vit_large_in21k": 'vit_large_patch16_224_in21k',
        "vit_base": 'vit_base_patch16_224',
        "vit_small": 'vit_small_patch16_224',
        "vit_tiny": 'vit_tiny_patch16_224',
        "resnet50": 'resnet50',
        }


def _get_checkpoint_path(model_name: str):
    name = _get_timm_name(model_name)
    prefix = model_name.split("_")[0]
    model_url = _dict_models_urls[prefix][name]
    print(f"Loading {model_url}")
    root = Path('~/.cache/torch/checkpoints').expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'{model_name}.pth'
    if not path.is_file():
        print('Downloading checkpoint...')
        _download(model_url, path)
        d = torch.load(path, map_location='cpu')
        ckpt_key_name = _dict_models_urls[prefix]["key"]
        if ckpt_key_name in d.keys():
            state_dict = d[ckpt_key_name]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("momentum_encoder.", ""): v for k, v in state_dict.items()} # for mocov3
            torch.save(state_dict, path)
        else:
            raise KeyError(f"{ckpt_key_name} not found. Only {d.keys()} are available.")
    return path


def _get_timm_name(model_name: str):
    prefix = model_name.split("_")[0]
    # remove prefix
    model_name = model_name.replace("".join([prefix,"_"]),"")
    if model_name in _dict_timm_names.keys():
        return _dict_timm_names[model_name]
    else:
        raise ValueError(f"Model {model_name} not found")


def _get_timm_model(model_name: str):
    timm_name = _get_timm_name(model_name)
    model = timm.create_model(
        timm_name,
        in_chans=3,
        num_classes=0,
        pretrained=False)
    _load_checkpoint(model, _get_checkpoint_path(model_name), strict=False)
    return model
