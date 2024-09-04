#!/usr/bin/env python3
import numpy as np
import torch
import os
# from .vit_backbones.swin_transformer import SwinTransformer
# from vit_backbones.vit import VisionTransformer
from .vit_backbones.vit_moco import vit_base, vit_base_3D, load_checkpoint_clip3D as load_ckpt_moco3D
from .vit_backbones.vit_moco_ss import vit_base_ss, vit_base_3D_ss, load_checkpoint_clip3D as load_ckpt_moco3D
from .vit_backbones.vit_mae import build_model as mae_vit_model
from .vit_backbones.vit_mae import load_checkpoint_clip3D as load_ckpt_mae3D

# from .vit_prompt.vit import PromptedVisionTransformer
# from .vit_prompt.swin_transformer import PromptedSwinTransformer
# from .vit_prompt.vit_moco import vit_base as prompt_vit_base
# from .vit_prompt.vit_mae import build_model as prompt_mae_vit_model
#
# from .vit_adapter.vit_mae import build_model as adapter_mae_vit_model
# from .vit_adapter.vit_moco import vit_base as adapter_vit_base
#
# from .vit_adapter.vit import ADPT_VisionTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ZOO = {
    "swint_imagenet": "swin_tiny_patch4_window7_224.pth",
    "swint_imagenet_ssl": "moby_swin_t_300ep_pretrained.pth",
    "swins_imagenet": "swin_small_patch4_window7_224.pth",
    "swinb_imagenet_224": "swin_base_patch4_window7_224.pth",
    "swinb_imagenet_384": "swin_base_patch4_window12_384.pth",
    "swinb_imagenet22k_224": "swin_base_patch4_window7_224_22k.pth",
    "swinb_imagenet22k_384": "swin_base_patch4_window12_384_22k.pth",
    "swinl_imagenet22k_224": "swin_large_patch4_window7_224_22k.pth",
    "sup_vitb8": "ViT-B_8.npz",
    "sup_vitb16_224": "ViT-B_16-224.npz",
    "sup_vitb16": "ViT-B_16.npz",
    "sup_vitl16_224": "ViT-L_16-224.npz",
    "sup_vitl16": "ViT-L_16.npz",
    "sup_vitb8_imagenet21k": "imagenet21k_ViT-B_8.npz",
    "sup_vitb32_imagenet21k": "imagenet21k_ViT-B_32.npz",
    "sup_vitb16_imagenet21k": "imagenet21k_ViT-B_16.npz",
    "sup_vitl16_imagenet21k": "imagenet21k_ViT-L_16.npz",
    "sup_vitl32_imagenet21k": "imagenet21k_ViT-L_32.npz",
    "sup_vith14_imagenet21k": "imagenet21k_ViT-H_14.npz",
    "mae_vith14": "mae_pretrain_vit_huge.pth",
    "mae_vitb16": "mae_pretrain_vit_base.pth", # "mae_pretrain_vit_b.pth",
    "mae_vitl16": "mae_pretrain_vit_large.pth",
    "sam_vitb16": "sam_vit_b_01ec64.pth",
    "mocov3_vitb": "linear-vit-b-300ep.pth.tar" #"vit-b-300ep.pth"
}


def build_mae_model(
        # model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None
        model_root="/project/hnguyen2/stly/code/prompting/pretrained",
        train_3D=False, bootstrap_method='inflation'
):
    # if prompt_cfg is not None:
    #     model = prompt_mae_vit_model(model_type, prompt_cfg)
    # elif adapter_cfg is not None:
    #     model = adapter_mae_vit_model(model_type, adapter_cfg)
    # else:
    #     model = mae_vit_model(model_type)

    # out_dim = model.embed_dim

    ckpt = os.path.join(model_root, 'mae', MODEL_ZOO["mae_vitb16"])
    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['model']

    if train_3D:
        # in_chans_3D = 1  # cfg.MODEL.BACKBONE.NAME
        model = mae_vit_model("mae_3D")
        depth_chans_3D = 224 #cfg.INPUT.SIZE[0]
        state_dict = load_ckpt_mae3D(model, state_dict, bootstrap_method, depth_chans_3D)
    else:
        model = mae_vit_model("mae_vitb16")

    msg = model.load_state_dict(state_dict, strict=False)
    print('Load MAE ckpt', msg)
    model.head = torch.nn.Identity()

    convert_weights(model)
    return model#, out_dim


def build_mocov3_model(
        # model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None
        model_root="/project/hnguyen2/stly/code/prompting/pretrained",
        train_3D=False, bootstrap_method='inflation'
):
    # if model_type != "mocov3_vitb":
    #     raise ValueError("Does not support other arch")
    # if prompt_cfg is not None:
    #     model = prompt_vit_base(prompt_cfg)
    # elif adapter_cfg is not None:
    #     model = adapter_vit_base(adapter_cfg)
    # else:
    #     model = vit_base()

    out_dim = 768
    ckpt = os.path.join(model_root, "mocov3", MODEL_ZOO["mocov3_vitb"])

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    if train_3D:
        # in_chans_3D = 1  # cfg.MODEL.BACKBONE.NAME
        model = vit_base_3D()
        depth_chans_3D = 224 #cfg.INPUT.SIZE[0]
        state_dict = load_ckpt_moco3D(model, state_dict, bootstrap_method, depth_chans_3D)
    else:
        model = vit_base()

    msg = model.load_state_dict(state_dict, strict=False)
    print('Load MoCo-v3 ckpt', msg)
    model.head = torch.nn.Identity()

    convert_weights(model)
    return model#, out_dim

def build_mocov3_model_ss(
        # model_type, crop_size, prompt_cfg, model_root, adapter_cfg=None
        model_root="/project/hnguyen2/stly/code/prompting/pretrained",
        train_3D=False, bootstrap_method='inflation'
):

    out_dim = 768
    ckpt = os.path.join(model_root, "mocov3", MODEL_ZOO["mocov3_vitb"])

    checkpoint = torch.load(ckpt, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    if train_3D:
        # in_chans_3D = 1  # cfg.MODEL.BACKBONE.NAME
        model = vit_base_3D_ss()
        depth_chans_3D = 224 #cfg.INPUT.SIZE[0]
        state_dict = load_ckpt_moco3D(model, state_dict, bootstrap_method, depth_chans_3D)
    else:
        model = vit_base_ss()

    msg = model.load_state_dict(state_dict, strict=False)
    print('Load MoCo-v3 ckpt', msg)
    model.head = torch.nn.Identity()

    convert_weights(model)
    return model#, out_dim

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        # if isinstance(l, nn.MultiheadAttention):
        #     for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
        #         tensor = getattr(l, attr)
        #         if tensor is not None:
        #             tensor.data = tensor.data.half()

        # for name in ["text_projection", "proj"]:
        #     if hasattr(l, name):
        #         attr = getattr(l, name)
        #         if attr is not None:
        #             attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

"""
# SWIN AND SUP
def build_swin_model(model_type, crop_size, prompt_cfg, model_root):
    if prompt_cfg is not None:
        return _build_prompted_swin_model(
            model_type, crop_size, prompt_cfg, model_root)
    else:
        return _build_swin_model(model_type, crop_size, model_root)


def _build_prompted_swin_model(model_type, crop_size, prompt_cfg, model_root):
    if model_type == "swint_imagenet":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swint_imagenet_ssl":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4

    elif model_type == "swins_imagenet":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swinb_imagenet_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet_384":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4

    elif model_type == "swinb_imagenet22k_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet22k_384":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinl_imagenet22k_224":
        model = PromptedSwinTransformer(
            prompt_cfg,
            img_size=crop_size,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 192
        num_layers = 4

    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model']

    if crop_size == 448:
        for k in list(state_dict.keys()):
            if "attn_mask" not in k:
                # remove prefix
                state_dict[k] = state_dict[k]
            # delete renamed or unused k
            else:
                del state_dict[k]

    # rename some keys for ssl models
    if model_type.endswith("ssl"):
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def _build_swin_model(model_type, crop_size, model_root):
    if model_type == "swint_imagenet":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,  # setting to a negative value will make head as identity
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swint_imagenet_ssl":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4

    elif model_type == "swins_imagenet":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.3,
            num_classes=-1,
        )
        embed_dim = 96
        num_layers = 4
    elif model_type == "swinb_imagenet_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet_384":
        model = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4

    elif model_type == "swinb_imagenet22k_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinb_imagenet22k_384":
        model = SwinTransformer(
            img_size=384,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 128
        num_layers = 4
    elif model_type == "swinl_imagenet22k_224":
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=-1,
        )
        embed_dim = 192
        num_layers = 4

    feat_dim = int(embed_dim * 2 ** (num_layers - 1))
    # load checkpoint
    model_w = os.path.join(model_root, MODEL_ZOO[model_type])
    checkpoint = torch.load(model_w, map_location='cpu')
    state_dict = checkpoint['model']

    if crop_size == 448:
        for k in list(state_dict.keys()):
            if "attn_mask" not in k:
                # remove prefix
                state_dict[k] = state_dict[k]
            # delete renamed or unused k
            else:
                del state_dict[k]

    # rename some keys for ssl models
    if model_type.endswith("ssl"):
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder.'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    return model, feat_dim


def build_vit_sup_models(
        # model_type, crop_size, prompt_cfg=None, model_root=None, adapter_cfg=None, load_pretrain=True, vis=False
model_type, crop_size, prompt_cfg=None, model_root=None, adapter_cfg=None, load_pretrain=True, vis=False
):
    # image size is the size of actual image
    m2featdim = {
        "sup_vitb16_224": 768,
        "sup_vitb16": 768,
        "sup_vitl16_224": 1024,
        "sup_vitl16": 1024,
        "sup_vitb8_imagenet21k": 768,
        "sup_vitb16_imagenet21k": 768,
        "sup_vitb32_imagenet21k": 768,
        "sup_vitl16_imagenet21k": 1024,
        "sup_vitl32_imagenet21k": 1024,
        "sup_vith14_imagenet21k": 1280,
    }
    # if prompt_cfg is not None:
    #     model = PromptedVisionTransformer(
    #         prompt_cfg, model_type,
    #         crop_size, num_classes=-1, vis=vis
    #     )
    # elif adapter_cfg is not None:
    #     model = ADPT_VisionTransformer(model_type, crop_size, num_classes=-1, adapter_cfg=adapter_cfg)
    # 
    # else:
    #     model = VisionTransformer(
    #         model_type, crop_size, num_classes=-1, vis=vis)
    
    model = VisionTransformer(
        model_type, crop_size, num_classes=-1, vis=vis)
    
    if load_pretrain:
        model.load_from(np.load(os.path.join(model_root, MODEL_ZOO[model_type])))

    return model, m2featdim[model_type]
"""

# load checkpoint 3D
# def load_checkpoint_clip3D(
#         model,
#         state_dict,
#         bootstrap_method="inflation",
#         depth_chans=224
# ):
#     print('bootstrap_method= ', bootstrap_method)
#     # state_dict = torch.load(pretrained_weights, map_location=map_location)
#     # if checkpoint_key is not None and checkpoint_key in state_dict:
#     #     print(f"Take key {checkpoint_key} in provided checkpoint dict")
#     #     state_dict = state_dict[checkpoint_key]
#
#     # --- starting inflate/center weights ---
#     n_slices = model.patch_embed.patch_size[-1]
#     n_chans = 1 #model.patch_embed.in_chans
#     # n_slices = state_dict["visual.conv1.weight"].shape[-1]  # patch_size
#     # n_chans = state_dict["visual.conv1.weight"].shape[1]  # in_chanes
#     # n_slices = model.visual.conv1.weight.shape[-1]  # patch_size
#     # n_chans = model.visual.conv1.weight.shape[1]  # in_chanes
#
#     # print('load_checkpoint_clip3D n_slices', n_slices)
#     # print('load_checkpoint_clip3D n_chans', n_chans)
#
#     key = "patch_embed.proj.weight"
#     # key = "visual.conv1.weight"
#     emb = state_dict[key]
#     print("load_checkpoint_clip3D patch_embed.conv1.weight Old:", emb.shape, emb.sum())
#
#     emb = emb.sum(1, keepdim=True)  # from colored to grayed
#     emb = (
#             emb.repeat(1, n_chans, 1, 1) / n_chans
#     )  # from 1-channel grayed to n-channel grayed
#     # emb = emb.unsqueeze(2).repeat(1, 1, n_slices, 1, 1) # from 2D to 3D
#     emb = emb.unsqueeze(4)
#     emb = emb.repeat(1, 1, 1, 1, n_slices)  # from 2D to 3D
#     if bootstrap_method == "inflation":
#         print("Inflation!!!")
#         emb = emb / n_slices
#     elif bootstrap_method == "centering":
#         print("Centering!!!")
#         center_idx = n_slices // 2
#         all_idxs = list(range(n_slices))
#         all_idxs.pop(center_idx)
#         emb[:, :, :, :, all_idxs] = 0
#     else:
#         raise
#     print("load_checkpoint_clip3D visual.conv1.weight New:", emb.shape, emb.sum())
#     state_dict[key] = emb#.float()
#     # print('load_checkpoint_clip3D state_dict[key]', state_dict[key].shape)
#     # --- ending inflate/center weights ---
#
#     ori_num_patches = state_dict["pos_embed"].shape[1] - 1
#     cur_num_patches = model.patch_embed.num_patches
#     # ori_num_patches = state_dict["visual.positional_embedding"].data.shape[0] - 1
#     # cur_num_patches = model.visual.positional_embedding.data.shape[0] - 1
#
#     # print("load_checkpoint_clip3D ori_num_patches", ori_num_patches)
#     # print("load_checkpoint_clip3D cur_num_patches", cur_num_patches)
#
#     # same sum
#     print("load_checkpoint_clip3D model.pos_embed shape", model.pos_embed.shape)
#     print("load_checkpoint_clip3D, ori_num_patches, cur_num_patches",  ori_num_patches, cur_num_patches)
#     if ori_num_patches != cur_num_patches:
#         emb = state_dict["pos_embed"] # torch.Size([197, 768])
#         cls_emb = emb[0, :]
#         emb = emb[1:, :]
#         ori_patch_size = int(ori_num_patches ** 0.5)
#         # cur_patch_size = ori_patch_size # int(cur_num_patches ** 0.5)
#         cur_patch_size = int(round(cur_num_patches**(1/3)))
#         feature_size = emb.shape[-1]
#
#         print("load_checkpoint_clip3D emb pos_embed shape", emb.shape)
#         # emb_resize = emb.view(1, ori_patch_size, ori_patch_size, feature_size)
#         emb_resize = emb.view(1, ori_patch_size, ori_patch_size, -1)
#         emb_resize = emb_resize.permute(0, 3, 1, 2)
#         emb_resize = emb_resize.reshape(emb_resize.shape[0], emb_resize.shape[1], ori_patch_size**2) # torch.Size([1, 768, 14*14])
#         print("load_checkpoint_clip3D emb_resize:", emb_resize.shape, emb_resize.sum())
#
#         # ratio = cur_num_patches/ori_patch_size
#         # ratio = cur_num_patches**3/ori_patch_size**2
#         # print('ratio', ratio)
#         # print('ratio', cur_num_patches/ori_patch_size)
#         emb_new = F.interpolate(emb_resize/cur_patch_size, (cur_patch_size**2 * depth_chans//n_slices))   # (14**2 * 224// 16) # torch.Size([1, 768, 14*14*14])
#         # emb_new = F.interpolate(emb_resize * ratio, (cur_patch_size ** 2 * depth_chans // n_slices))  # (14**2 * 224// 16) # torch.Size([1, 768, 14*14*14])
#         print("load_checkpoint_clip3D emb_new:", emb_new.shape, emb_new.sum())
#         emb_new = emb_new.permute(0, 2, 1)   # torch.Size([1, 14*14*14, 768])
#         emb_new = emb_new.squeeze(0)
#         emb_new = torch.cat((emb_new, cls_emb.unsqueeze(0)))
#         emb_new = emb_new.squeeze()
#         state_dict["visual.positional_embedding"] = emb_new
#
#     return state_dict




def convert_statedict(state_dict: dict, model, reduce_times: int):
    '''
    interpolate weights of state_dict
    :param state_dict:
    :param reduce_times:
    :return:
    '''

    def convert_1d(param):
        param = param.view(1,1,1,param.shape[0])
        new_param = F.interpolate(param, (1, param.shape[-1]//reduce_times))
        new_param *= param.sum() / new_param.sum()
        return new_param.squeeze()

    def convert_pos_embed(param):   # torch.Size([2745, 384]) dict_shape torch.Size([2745, 768])
        param = param.view(param.shape[0], 1, 1, param.shape[1])    # [patches, 1, 1, dim]
        new_param = F.interpolate(param, (1, param.shape[-1]//reduce_times)) # [patches, 1, 1, dim//reduce_times]
        new_param *= param.sum() / new_param.sum()  # [patches, 1, 1, dim//reduce_times]
        return new_param.squeeze()  # [patches, dim//reduce_times]

    def convert_proj(param):    # torch.Size([384, 512]) dict_shape torch.Size([768, 512])
        param = param.view(param.shape[1], 1, 1, param.shape[0])  # [patches, 1, 1, dim]
        new_param = F.interpolate(param, (1, param.shape[-1] // reduce_times))  # [patches, 1, 1, dim//reduce_times]
        new_param *= param.sum() / new_param.sum()  # [patches, 1, 1, dim//reduce_times]
        return new_param.squeeze().permute(1, 0)  # [dim//reduce_times, patches]

    def convert_conv1(param):    # torch.Size([192, 1, 16, 16, 16]) dict_shape torch.Size([768, 1, 16, 16, 16])
        patch_size = param.shape[-1]
        param = param.view(-1, 1, 1, param.shape[0])    # [patch_size**3, 1, 1, dim]
        new_param = F.interpolate(param, (1, param.shape[-1] // reduce_times))  # [patch_size, patch_size, patch_size, 1, dim]
        new_param *= param.sum() / new_param.sum()  # [patch_size, patch_size, patch_size, 1, dim]
        return new_param.squeeze().view(new_param.shape[-1], 1, patch_size, patch_size, patch_size)

    def convert_2d(param):  # torch.Size([1152, 384]) dict_shape torch.Size([2304, 768])
        param = param.view(1, 1, param.shape[0], param.shape[1])  # [patches, 1, 1, dim]
        new_param = F.interpolate(param, (param.shape[-2] // reduce_times, param.shape[-1] // reduce_times))  # [patches, 1, 1, dim//reduce_times]
        new_param *= param.sum() / new_param.sum()  # [patches, 1, 1, dim//reduce_times]
        return new_param.squeeze()  # [dim//reduce_times, dim//reduce_times]

    # count_model = 0
    # count_dict = 0
    # for (name_d, param_d), (name_m, param_m) in zip(state_dict.items(), model.named_parameters()):
    #     if 'visual' in name_d and 'visual' in name_m:
    #         print('name_d: ', name_d, 'name_m: ', name_m)
    # print('##################################')
    new_dict = {}
    scale_shift_name = ['visual.scale', 'visual.shift']
    for name, param in model.named_parameters():
        if 'visual' in name and name not in scale_shift_name:
            print('name: ', name, 'param shape: ', param.shape,
                  'dict_shape', state_dict[name].shape)

            if 'class_embedding' in name or 'ln_pre' in name or 'ln_post' in name:
                new_dict[name] = convert_1d(state_dict[name].float())
            elif 'transformer.resblocks' in name:
                if 'in_proj_bias' in name or '.bias' in name:# or '.weight' in name:
                    new_dict[name] = convert_1d(state_dict[name].float())
            elif 'positional_embedding' in name:
                new_dict[name] = convert_pos_embed(state_dict[name].float())
            elif 'visual.proj' in name:
                new_dict[name] = convert_proj(state_dict[name].float())
            elif 'visual.conv1' in name:
                new_dict[name] = convert_conv1(state_dict[name].float())
            else:
                new_dict[name] = convert_2d(state_dict[name].float())
            del state_dict[name]
    # print('##################################')
    # for name, param in state_dict.items():
    #     if 'visual' in name:
    #         print('name: ', name, 'param shape: ', param.shape)
            # count_dict += 1

    # print('len model, len state_dict', count_model, count_model)
    return new_dict


# model = build_mae_model(train_3D=True)
# model = build_mocov3_model(train_3D=True)
# print(out_dim)
# print(model)