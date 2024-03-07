from pathlib import Path

import requests
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm


def _backbone_param(model):
    try:
        return model.conv1.weight
    except AttributeError:
        return next(model.parameters())


def backbone_dtype(model):
    if not isinstance(model, nn.Module):
        return torch.float
    return _backbone_param(model).dtype


@torch.no_grad()
def get_embed_dim(args, model, size=224):
    if args is not None:
        from model_builders import load_embeds
        try:
            return load_embeds(args).shape[-1]
        except Exception:
            raise ValueError('Embeddings could not be loaded and thus could not infer embed_dim. Generate embeddings first.')
    if isinstance(model, nn.Module):
        p = _backbone_param(model)
        dummy_in = torch.empty(1, 3, size, size,
                               device=p.device, dtype=p.dtype)
        dummy_out = model(dummy_in)
        return dummy_out.size(-1)
    raise ValueError('Could not infer embed_dim')


def remove_prefix(arch, prefix):
    for join_char in ["-", "_"]:
        arch = arch.replace(prefix + join_char, "")
    return arch


def _split_preprocess_aux(preprocess):
    if isinstance(preprocess, transforms.Compose):
        trafos = preprocess.transforms
        if len(trafos) == 1:
            raise ValueError(f"Can't handle preprocess: {preprocess}")
        preprocess, rest_trafos = trafos[:-1], trafos[-1]
        rest_preprocess, norm = _split_preprocess_aux(rest_trafos)
        return preprocess + rest_preprocess, norm
    elif isinstance(preprocess, transforms.Normalize):
        return [], preprocess
    raise ValueError(f"Can't handle preprocess: {preprocess}")


def split_normalization(preprocess):
    """
    Split a preprocessing pipeline into a normalization and a preprocessing
    """
    preprocess, norm = _split_preprocess_aux(preprocess)
    return transforms.Compose(preprocess), norm


def _download(url: str, filename: Path):
    """from https://stackoverflow.com/a/37573701"""
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception(f"Could not download from {url}")
