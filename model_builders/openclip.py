import open_clip

from model_builders.model_utils import remove_prefix


def _get_openclip_modelnames():
    def map_pair(name, pretraining):
        unwanted = any([x in name for x in ["RN101",  "RN50-", "RN50x", 'ViT-B-32', "ViT-L-14-336", "roberta", "xlm"]])
        pretrainings = 'openai' in pretraining or (name, pretraining) in _laion_name_to_dataset.items()
        if unwanted or not pretrainings:
            return None
        
        if 'laion2b' in pretraining:
            pretraining = 'laion2b'
        return f'{name}/{pretraining}'
    models = [map_pair(name, pretraining) for name, pretraining in open_clip.list_pretrained()]
    return [m for m in models if m is not None]


# Chose laion datasets for particular model
_laion_name_to_dataset = {
        'ViT-B-16': 'laion2b_s34b_b88k',
        'ViT-L-14': 'laion2b_s32b_b82k',
        'ViT-H-14': 'laion2b_s32b_b79k',
        'ViT-bigG-14': 'laion2b_s39b_b160k',
        'convnext_base': 'laion400m_s13b_b51k',
        'convnext_base_w': 'laion2b_s13b_b82k',
        'convnext_large_d': 'laion2b_s26b_b102k_augreg',
        'convnext_xxlarge': 'laion2b_s34b_b82k_augreg',
}


def _map_openclip_pretraining(pretraining, arch):
    if pretraining == 'laion2b':
        return _laion_name_to_dataset[arch]
    return pretraining


def build_openclip_visual(arch):
    model, preprocess, _ = build_openclip_complete(arch)
    backbone = model.visual
    backward_compat = 'openclip' not in arch
    if not backward_compat and hasattr(backbone, "proj"):
        backbone.proj = None
    return backbone, preprocess


def build_openclip_text(arch):
    """
    Returns model, tokenizer, proj
    where model is the CLIP model without the visual part and proj is projection matrix
    for the visual embedding or None if there is no projection matrix
    """
    model, _, tokenizer = build_openclip_complete(arch)
    if hasattr(model.visual, "proj"):
        visual_proj = model.visual.proj.detach()
        visual_proj.requires_grad = False
    else:
        visual_proj = None
    model.visual = None
    return model, tokenizer, visual_proj


def build_openclip_complete(arch):
    backward_compat = 'openclip' not in arch
    if backward_compat:
        # For backwards compatibility
        arch = remove_prefix(arch, 'clip').replace('/', '-')
        pretraining = 'openai'
    else:
        arch = remove_prefix(arch, 'openclip')
        arch, pretraining = arch.rsplit('/', 1)
    pretraining = _map_openclip_pretraining(pretraining, arch)
    avail_models, avail_pretrainings = zip(*open_clip.list_pretrained())
    assert arch in avail_models, f"Model {arch} not available. Available models are {avail_models}"
    tokenizer = open_clip.get_tokenizer(arch)
    if pretraining is not None:
        assert pretraining in avail_pretrainings, \
            f"Pretraining {pretraining} not available. Available pretrainings are {open_clip.list_pretrained()}"
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretraining)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(arch)
    model.eval()
    return model, preprocess, tokenizer
