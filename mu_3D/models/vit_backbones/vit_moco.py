#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from https://github.com/facebookresearch/moco-v3/blob/main/vits.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncodingPermute3D

from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.layers.helpers import to_2tuple, to_3tuple
from timm.layers.trace_utils import _assert

__all__ = [
    'vit_small',
    'vit_base',
    'vit_conv_small',
    'vit_conv_base',
]

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            # in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten

        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # print("PatchEmbed3D, forward", x.shape)
        B, C, H, W, D = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        _assert(D == self.img_size[2], f"Input image width ({D}) doesn't match model ({self.img_size[2]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x.half())
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x.half()).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)#.to(x.dtype)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



class VisionTransformerMoCo(VisionTransformer):
    def __init__(self,
                 init_values=None,
                 pre_norm=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=None,
                 stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)

        embed_dim = kwargs['embed_dim']
        img_size = kwargs['img_size']
        in_chans = kwargs['in_chans']
        patch_size = kwargs['patch_size']
        norm_layer = kwargs['norm_layer']
        num_heads = kwargs['num_heads']
        qkv_bias = kwargs['qkv_bias']
        mlp_ratio = kwargs['mlp_ratio']
        depth = kwargs['depth']

        self.depth = depth
        
        # from timm
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #if class_token else None
        # embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        act_layer = act_layer or nn.GELU

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])


        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    # from timm
    # def _pos_embed(self, x):
    #     if self.no_embed_class:
    #         # deit-3, updated JAX (big vision)
    #         # position embedding does not overlap with class token, add then concat
    #         x = x + self.pos_embed
    #         if self.cls_token is not None:
    #             x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #     else:
    #         # original timm, JAX, and deit vit impl
    #         # pos_embed has entry for class token, concat then add
    #         if self.cls_token is not None:
    #             x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #         x = x + self.pos_embed
    #     return self.pos_drop(x)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]

        x = self.norm(x)

        return x

    # from timm
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    # from timm
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class VisionTransformerMoCo_3D(VisionTransformer):
    def __init__(self,
                 init_values=None,
                 pre_norm=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=None,
                 stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)

        embed_dim = kwargs['embed_dim']
        img_size = kwargs['img_size']
        patch_size = kwargs['patch_size']
        norm_layer = kwargs['norm_layer']
        num_heads = kwargs['num_heads']
        qkv_bias = kwargs['qkv_bias']
        mlp_ratio = kwargs['mlp_ratio']
        depth = kwargs['depth']

        self.depth = depth

        # from timm
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #if class_token else None
        # embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        act_layer = act_layer or nn.GELU

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)])

        # Use fixed 2D sin-cos position embedding
        # self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed3D):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    # def build_2d_sincos_position_embedding(self, temperature=10000.):
    #     h, w, d = self.patch_embed.grid_size
    #     grid_w = torch.arange(w, dtype=torch.float32)
    #     grid_h = torch.arange(h, dtype=torch.float32)
    #     grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    #     assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    #     pos_dim = self.embed_dim // 4
    #     omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    #     omega = 1. / (temperature ** omega)
    #     out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    #     out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    #     pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    #
    #     # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
    #     pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
    #     self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    #     self.pos_embed.requires_grad = False

    # from timm
    # def _pos_embed(self, x):
    #     if self.no_embed_class:
    #         # deit-3, updated JAX (big vision)
    #         # position embedding does not overlap with class token, add then concat
    #         x = x + self.pos_embed
    #         if self.cls_token is not None:
    #             x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #     else:
    #         # original timm, JAX, and deit vit impl
    #         # pos_embed has entry for class token, concat then add
    #         if self.cls_token is not None:
    #             x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    #         x = x + self.pos_embed
    #     return self.pos_drop(x)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]

        x = self.norm(x)

        return x

    # from timm
    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    # from timm
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def vit_small(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base(**kwargs):
    model = VisionTransformerMoCo(
        img_size=224, in_chans=3,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_small(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_conv_base(**kwargs):
    # minus one ViT block
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

def vit_base_3D(**kwargs):
    model = VisionTransformerMoCo_3D(
        img_size=224, #in_chans=3,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

# load checkpoint 3D
def load_checkpoint_clip3D(
        model,
        state_dict,
        bootstrap_method="inflation",
        depth_chans=224
):
    # print('bootstrap_method= ', bootstrap_method)
    # state_dict = torch.load(pretrained_weights, map_location=map_location)
    # if checkpoint_key is not None and checkpoint_key in state_dict:
    #     print(f"Take key {checkpoint_key} in provided checkpoint dict")
    #     state_dict = state_dict[checkpoint_key]

    # --- starting inflate/center weights ---
    n_slices = model.patch_embed.patch_size[-1]
    n_chans = 1 #model.patch_embed.in_chans
    # n_slices = state_dict["visual.conv1.weight"].shape[-1]  # patch_size
    # n_chans = state_dict["visual.conv1.weight"].shape[1]  # in_chanes
    # n_slices = model.visual.conv1.weight.shape[-1]  # patch_size
    # n_chans = model.visual.conv1.weight.shape[1]  # in_chanes

    # print('load_checkpoint_clip3D n_slices', n_slices)
    # print('load_checkpoint_clip3D n_chans', n_chans)

    key = "patch_embed.proj.weight"
    # key = "visual.conv1.weight"
    emb = state_dict[key]
    print("load_checkpoint_clip3D patch_embed.conv1.weight Old:", emb.shape, emb.sum())

    emb = emb.sum(1, keepdim=True)  # from colored to grayed
    emb = (
            emb.repeat(1, n_chans, 1, 1) / n_chans
    )  # from 1-channel grayed to n-channel grayed
    # emb = emb.unsqueeze(2).repeat(1, 1, n_slices, 1, 1) # from 2D to 3D
    emb = emb.unsqueeze(4)
    emb = emb.repeat(1, 1, 1, 1, n_slices)  # from 2D to 3D
    if bootstrap_method == "inflation":
        print("bootstrap_method = Inflation!!!")
        emb = emb / n_slices
    elif bootstrap_method == "centering":
        print("bootstrap_method = Centering!!!")
        center_idx = n_slices // 2
        all_idxs = list(range(n_slices))
        all_idxs.pop(center_idx)
        emb[:, :, :, :, all_idxs] = 0
    else:
        raise
    print("load_checkpoint_clip3D visual.conv1.weight New:", emb.shape, emb.sum())
    state_dict[key] = emb#.float()
    # print('load_checkpoint_clip3D state_dict[key]', state_dict[key].shape)
    # --- ending inflate/center weights ---

    ori_num_patches = state_dict["pos_embed"].shape[1] - 1
    cur_num_patches = model.patch_embed.num_patches
    # ori_num_patches = state_dict["visual.positional_embedding"].data.shape[0] - 1
    # cur_num_patches = model.visual.positional_embedding.data.shape[0] - 1

    # print("load_checkpoint_clip3D ori_num_patches", ori_num_patches)
    # print("load_checkpoint_clip3D cur_num_patches", cur_num_patches)

    # different sum
    # if ori_num_patches != cur_num_patches:
    #     # print("load_checkpoint_clip3D ori_num_patches != cur_num_patches:")
    #     # emb = state_dict["pos_embed"]
    #     emb = state_dict["visual.positional_embedding"] # torch.Size([197, 768])
    #     # cls_emb = emb[:, 0]
    #     # emb = emb[:, 1:]
    #     cls_emb = emb[0, :]
    #     emb = emb[1:, :]
    #     ori_patch_size = int(ori_num_patches ** 0.5)
    #     cur_patch_size = ori_patch_size # int(cur_num_patches ** 0.5)
    #     feature_size = emb.shape[-1]
    #     # print('load_checkpoint_clip3D emb', emb.shape)
    #     # print('load_checkpoint_clip3D ori_patch_size', ori_patch_size)
    #     # print('load_checkpoint_clip3D cur_patch_size', cur_patch_size)
    #     # print('load_checkpoint_clip3D feature_size', feature_size)
    #     emb_resize = emb.view(1, ori_patch_size, ori_patch_size, feature_size)
    #     emb_resize = emb_resize.permute(0, 3, 1, 2)
    #     # print('load_checkpoint_clip3D emb_resize', emb_resize.shape)    # torch.Size([1, 768, 14, 14])
    #     emb_resize = emb_resize.reshape(emb_resize.shape[0], emb_resize.shape[1], ori_patch_size**2) # torch.Size([1, 768, 14*14])
    #     # emb_new = F.interpolate(emb_resize, (cur_patch_size, cur_patch_size, cur_patch_size))
    #     emb_new = F.interpolate(emb_resize, (ori_patch_size**3))   # torch.Size([1, 768, 14*14*14])
    #     # emb_new = emb_new.reshape(emb_resize.shape[0], emb_resize.shape[1], cur_patch_size, cur_patch_size, cur_patch_size)
    #     # print('load_checkpoint_clip3D emb_new', emb_new.shape)
    #     # emb_new = emb_new.permute(0, 2, 3, 1)
    #     emb_new = emb_new.permute(0, 2, 1)   # torch.Size([1, 14*14*14, 768])
    #     # emb_new = emb_new.reshape(1, cur_patch_size * cur_patch_size * cur_patch_size, feature_size)
    #     emb_new = emb_new.squeeze(0)
    #     # print('load_checkpoint_clip3D emb_new 0', emb_new.shape)
    #     # print('load_checkpoint_clip3D', cls_emb.shape)
    #     emb_new = torch.cat((emb_new, cls_emb.unsqueeze(0)))
    #     # print('load_checkpoint_clip3D emb_new 1', emb_new.shape)
    #     emb_new = emb_new.squeeze()
    #     # print('load_checkpoint_clip3D', emb_new.shape)
    #     # state_dict["pos_embed"] = emb_new
    #     state_dict["visual.positional_embedding"] = emb_new#.float()

    # print("load_checkpoint_clip3D model.pos_embed shape", model.pos_embed.shape)
    # print("load_checkpoint_clip3D, ori_num_patches, cur_num_patches",  ori_num_patches, cur_num_patches)
    # same sum
    if ori_num_patches != cur_num_patches:
        emb = state_dict["pos_embed"].squeeze() # torch.Size([1, 197, 768])
        # print("load_checkpoint_clip3D state_dict pos_embed shape", emb.shape)
        cls_emb = emb[0, :]
        emb = emb[1:, :]
        ori_patch_size = int(ori_num_patches ** 0.5)
        # cur_patch_size = ori_patch_size # int(cur_num_patches ** 0.5)
        cur_patch_size = int(round(cur_num_patches**(1/3)))
        feature_size = emb.shape[-1]


        emb_resize = emb.view(1, ori_patch_size, ori_patch_size, feature_size)
        emb_resize = emb_resize.permute(0, 3, 1, 2)
        emb_resize = emb_resize.reshape(emb_resize.shape[0], emb_resize.shape[1], ori_patch_size**2) # torch.Size([1, 768, 14*14])
        print("load_checkpoint_clip3D emb_resize:", emb_resize.shape, emb_resize.sum())

        # ratio = cur_num_patches/ori_patch_size
        # ratio = cur_num_patches**3/ori_patch_size**2
        # print('ratio', ratio)
        # print('ratio', cur_num_patches/ori_patch_size)
        emb_new = F.interpolate(emb_resize/cur_patch_size, (cur_patch_size**2 * depth_chans//n_slices))   # (14**2 * 224// 16) # torch.Size([1, 768, 14*14*14])
        # emb_new = F.interpolate(emb_resize * ratio, (cur_patch_size ** 2 * depth_chans // n_slices))  # (14**2 * 224// 16) # torch.Size([1, 768, 14*14*14])
        print("load_checkpoint_clip3D emb_new:", emb_new.shape, emb_new.sum())
        emb_new = emb_new.permute(0, 2, 1)   # torch.Size([1, 14*14*14, 768])
        emb_new = emb_new.squeeze(0)
        emb_new = torch.cat((emb_new, cls_emb.unsqueeze(0)))
        emb_new = emb_new.squeeze()
        state_dict["pos_embed"] = emb_new.unsqueeze(0)

    # print("load_checkpoint_clip3D ori_num_patches == cur_num_patches:")
    # remove `module.` prefix
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    # msg = model.load_state_dict(state_dict, strict=False)
    # print(
    #     "Pretrained weights found at {} and loaded with msg: {}".format(
    #         pretrained_weights, msg
    #     )
    # )

    return state_dict