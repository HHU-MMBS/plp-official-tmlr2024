try:
    # for python version >= 3.8
    from functools import cached_property
except ImportError:
    from functools import lru_cache

    def cached_property(func):
        return property(lru_cache()(func))

import torch
from torch import nn

from model_builders.model_utils import backbone_dtype
from model_builders.vision_transformer import DINOHead


class MultiHead(nn.Module):

    def __init__(self, dino_head_args, num_heads=1):
        super().__init__()
        self.num_heads = num_heads

        if self.num_heads < 1:
            raise ValueError("Number of heads must be at least 1.")
            # List of all heads
        self.heads = nn.ModuleList([DINOHead(**dino_head_args) for _ in range(self.num_heads)])
        # best head is arbitrary at beginning
        self.register_buffer("best_head_idx", torch.tensor(0))

    @property
    def best_head(self):
        return self.heads[self.best_head_idx]

    def set_losses(self, losses):
        """losses should be (num_heads,) tensor"""
        if self.num_heads == 1:
            return
        if len(losses) != self.num_heads:
            raise ValueError("Number of losses does not match number of heads.")
        self.best_head_idx = torch.argmin(losses)

    def forward(self, x):
        if not self.training or self.num_heads == 1:
            return self.best_head(x)
        return [head(x) for head in self.heads]


class MultiCropClip(nn.Module):
    def __init__(self, backbone,
                 embed_dim=512,
                 out_dim=4096,
                 use_bn_in_head=False,
                 head_dropout_prob=0.0,
                 head_final_gelu=False,
                 norm_last_layer=False,
                 req_grad=False,
                 l2_norm=False,
                 nlayers=3,
                 hidden_dim=1024,
                 layer_norm_only=False,
                 bottleneck_dim=256,
                 num_heads=1):
        super().__init__()

        self.register_buffer("embed_mean", torch.zeros(embed_dim))
        self.register_buffer("embed_std", torch.ones(embed_dim))

        precomputed = not isinstance(backbone, nn.Module)
        if layer_norm_only:
            for name, param in backbone.named_parameters():
                if "ln_" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                req_grad = True
        elif not precomputed:
            for p in backbone.parameters():
                p.requires_grad = req_grad

        self.backbone = backbone
        if precomputed:
            # self.backbone is just name of backbone that computed embeddings
            assert isinstance(self.backbone, str)
            self.backbone_embed = self.idendity_backbone
        if not req_grad:
            self.backbone_embed = torch.no_grad()(self.backbone_embed)
        self.l2_norm = l2_norm
        self.embed_dim = embed_dim
        self.req_grad = req_grad

        head_args = dict(
            in_dim=embed_dim,
            out_dim=out_dim,
            use_bn=use_bn_in_head,
            final_gelu=head_final_gelu,
            dropout_p=head_dropout_prob,
            nlayers=nlayers,
            hidden_dim=hidden_dim,
            norm_last_layer=norm_last_layer,
            bottleneck_dim=bottleneck_dim
        )
        self.head = MultiHead(head_args, num_heads)

        self.dtype = backbone_dtype(self.backbone)

    def set_mean_std(self, mean, std):
        self.embed_mean.data = mean.clone()
        self.embed_std.data = std.clone()

    def idendity_backbone(self, x):
        if isinstance(x, list):
            x = torch.cat(x)
        return (x - self.embed_mean) / self.embed_std

    def backbone_embed(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        shapes_sorted, sort_idx = torch.sort(torch.Tensor([inp.shape[-1] for inp in x]))
        idx_crops = torch.cumsum(torch.unique_consecutive(shapes_sorted, return_counts=True)[1], 0)
        start_idx = 0
        output = torch.empty((len(x), len(x[0]), self.embed_dim), device=x[0].device)
        for end_idx in idx_crops:
            batch_idx = sort_idx[start_idx:end_idx]  # The indices of tensors of this shape
            _in_batched = torch.cat([x[i] for i in batch_idx])  # Batch them together
            _out = self.backbone(_in_batched.type(self.dtype)).float()

            # accumulate outputs
            _out = torch.stack(_out.chunk(len(batch_idx)))
            output.index_copy_(0, batch_idx.cuda(), _out)
            start_idx = end_idx
        output = torch.cat(torch.unbind(output))
        return (output - self.embed_mean) / self.embed_std

    def apply_head(self, embedded):
        if self.l2_norm:
            embedded /= embedded.norm(dim=-1, keepdim=True)
        # Run the head forward on the concatenated features.
        if not self.req_grad:
            embedded = embedded.detach()
        return self.head(embedded)

    def forward(self, x):
        return self.apply_head(self.backbone_embed(x))
