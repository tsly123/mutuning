import os.path
import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data import DataManager
from dassl.data.data_manager import build_data_loader_medmnist3d as build_data_loader
from dassl.data.datasets import build_dataset
from dassl.data.samplers import build_sampler
from dassl.data.transforms import INTERPOLATION_MODES, build_transform_3D as build_transform
from tabulate import tabulate

from trainers.vision_benchmark.evaluation import construct_dataloader, construct_multitask_dataset

from clip import clip_ss as clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

import numpy as np
# torch.set_printoptions(precision=10)
from sklearn.metrics import roc_auc_score, confusion_matrix


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    in_chans_3D = 1  # cfg.MODEL.BACKBONE.NAME
    depth_chans_3D = cfg.INPUT.SIZE[0]
    print('depth_chans_3D', depth_chans_3D)
    bootstrap_method = "centering"  # 'centering', 'inflation', 'centering_square, cfg.MODEL.BACKBONE.BOOTSTRAP

    if in_chans_3D is not None:
        model = clip.build_model(state_dict or model.state_dict(), in_chans_3D, bootstrap_method, depth_chans_3D)
    else:
        model = clip.build_model(state_dict or model.state_dict())

    return model


# def load_clip_to_cpu2(cfg):
#     backbone_name = cfg.MODEL.BACKBONE.NAME
#     url = clip._MODELS[backbone_name]
#     model_path = clip._download(url)
#
#     try:
#         # loading JIT archive
#         model = torch.jit.load(model_path, map_location="cpu").eval()
#         state_dict = None
#
#     except RuntimeError:
#         state_dict = torch.load(model_path, map_location="cpu")
#
#     clip.convert_statedict(state_dict or model.state_dict(), 2)


# load checkpoint 3D
# def load_checkpoint_clip3D(
#         model,
#         state_dict,
#         bootstrap_method="centering",
# ):
#
#     # state_dict = torch.load(pretrained_weights, map_location=map_location)
#     # if checkpoint_key is not None and checkpoint_key in state_dict:
#     #     print(f"Take key {checkpoint_key} in provided checkpoint dict")
#     #     state_dict = state_dict[checkpoint_key]
#
#     # --- starting inflate/center weights ---
#     # n_slices = model.patch_embed.patch_size[-1]
#     # n_chans = model.patch_embed.in_chans
#     # n_slices = state_dict["visual.conv1.weight"].shape[-1]  # patch_size
#     # n_chans = state_dict["visual.conv1.weight"].shape[1]  # in_chanes
#     n_slices = model.visual.conv1.weight.shape[-1]  # patch_size
#     n_chans = model.visual.conv1.weight.shape[1]  # in_chanes
#
#     print('load_checkpoint_clip3D n_slices', n_slices)
#     print('load_checkpoint_clip3D n_chans', n_chans)
#
#     # key = "patch_embed.proj.weight"
#     key = "visual.conv1.weight"
#     emb = state_dict[key]
#     print("load_checkpoint_clip3D Old:", emb.shape, emb.sum())
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
#     print("load_checkpoint_clip3D New:", emb.shape, emb.sum())
#     state_dict[key] = emb
#     print('load_checkpoint_clip3D state_dict[key]', state_dict[key].shape)
#     # --- ending inflate/center weights ---
#
#     # ori_num_patches = state_dict["pos_embed"].shape[1] - 1
#     # cur_num_patches = model.patch_embed.num_patches
#     ori_num_patches = state_dict["visual.positional_embedding"].data.shape[0] - 1
#     cur_num_patches = model.visual.positional_embedding.data.shape[0] - 1
#     print("load_checkpoint_clip3D ori_num_patches", ori_num_patches)
#     print("load_checkpoint_clip3D cur_num_patches", cur_num_patches)
#
#     if ori_num_patches != cur_num_patches:
#         print("load_checkpoint_clip3D ori_num_patches != cur_num_patches:")
#         emb = state_dict["pos_embed"]
#         cls_emb = emb[:, 0]
#         emb = emb[:, 1:]
#         ori_patch_size = int(ori_num_patches ** 0.5)
#         cur_patch_size = int(cur_num_patches ** 0.5)
#         feature_size = emb.shape[-1]
#         emb_resize = emb.view(1, ori_patch_size, ori_patch_size, feature_size)
#         emb_resize = emb_resize.permute(0, 3, 1, 2)
#         emb_new = F.interpolate(emb_resize, (cur_patch_size, cur_patch_size))
#         emb_new = emb_new.permute(0, 2, 3, 1)
#         emb_new = emb_new.reshape(1, cur_patch_size * cur_patch_size, feature_size)
#         emb_new = emb_new.squeeze(0)
#         emb_new = torch.cat((emb_new, cls_emb))
#         emb_new = emb_new.unsqueeze(0)
#         state_dict["pos_embed"] = emb_new
#
#     print("load_checkpoint_clip3D ori_num_patches == cur_num_patches:")
#     # remove `module.` prefix
#     # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
#     # remove `backbone.` prefix induced by multicrop wrapper
#     # state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
#     # msg = model.load_state_dict(state_dict, strict=False)
#     # print(
#     #     "Pretrained weights found at {} and loaded with msg: {}".format(
#     #         pretrained_weights, msg
#     #     )
#     # )
#     return state_dict


# NOT progressive
def init_ssf_scale_shift(dim, dtype):
    scale = nn.Parameter(torch.ones(dim, dtype=dtype))
    shift = nn.Parameter(torch.zeros(dim, dtype=dtype))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        if x.ndim == 5:
            return x * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
        elif x.ndim == 4:
            return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


class ImageEncoder(nn.Module):
    def __init__(self, clip_model, mvlpt_model, classnames):
        super().__init__()
        # HACK: Assume all is vision transformer
        self.visual = clip_model.visual
        self.mvlpt_model = mvlpt_model
        layer = self.visual.transformer.layers
        width = clip_model.visual.conv1.weight.shape[0]
        dtype = clip_model.dtype
        embed_dim = clip_model.text_projection.shape[1]

        # self.scale = nn.Parameter(torch.ones(layer, width, dtype=dtype))
        # self.shift = nn.Parameter(torch.zeros(layer, width, dtype=dtype))
        # nn.init.normal_(self.scale, mean=1, std=.02)
        # nn.init.normal_(self.shift, std=.02)

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(width, dtype=dtype)
        # self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(width, dtype=dtype)
        # self.ssf_scale_3, self.ssf_shift_3 = init_ssf_scale_shift(width, dtype=dtype)
        # self.ssf_scale_4, self.ssf_shift_4 = init_ssf_scale_shift(width, dtype=dtype)
        # if self.visual.proj is not None:
        #     self.ssf_scale_5, self.ssf_shift_5 = init_ssf_scale_shift(embed_dim, dtype=dtype)

        self.head = nn.Sequential(torch.nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6),
                                  nn.Linear(embed_dim, len(classnames)))

        # print('image encoder shape', embed_dim, width)

    def forward(self, x: torch.Tensor, vpt_embeddings=None, vpt_embeddings_deep=None, extract_layer=False):
        # print('ImageEncoder x shape', x.shape)  # torch.Size([64, 3, 224, 224])
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]

        x = ssf_ada(x, self.ssf_scale_1, self.ssf_shift_1)

        # print('ImageEncoder x shape 2', x.shape)  # torch.Size([16, 768, 14, 14, 14])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] (16, 14 ** 3, 768)
        # print('ImageEncoder x shape 3', x.shape)
        # print('ImageEncoder class_embedding shape 3', self.visual.class_embedding.data.shape)
        # torch.Size([197, 768])    (16, 1, 14**3*768)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                             device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        # print('ImageEncoder x shape 4', x.shape)
        # print('ImageEncoder positional_embedding shape 4', self.visual.positional_embedding.data.shape)
        x = x + self.visual.positional_embedding.to(x.dtype)

        # x = ssf_ada(x, self.ssf_scale_2, self.ssf_shift_2)

        x = self.visual.ln_pre(x)

        # x = ssf_ada(x, self.ssf_scale_3, self.ssf_shift_3)  # .to(x.type())

        # print('ImageEncoder vpt_embeddings', vpt_embeddings.shape)
        # print('ImageEncoder vpt_embeddings_deep', vpt_embeddings_deep.shape)

        B = x.shape[0]

        x, ctx = self.mvlpt_model.forward_vpt(x, vpt_embeddings)
        x = x.permute(1, 0, 2)  # NLD -> LND

        input_x = []
        ext_x = []
        if self.mvlpt_model.vpt_deep and (
                vpt_embeddings_deep is not None or self.mvlpt_model.vpt_embeddings_deep is not None):
            if vpt_embeddings_deep is None:
                vpt_embeddings_deep = self.mvlpt_model.vpt_embeddings_deep

            for layer_idx in range(self.visual.transformer.layers):
                layer = self.visual.transformer.resblocks[layer_idx]

                if layer_idx == 0:
                    x = layer(x)
                elif layer_idx <= vpt_embeddings_deep.shape[0]:
                    vpt_emb_deep = self.mvlpt_model.vpt_dropout(self.mvlpt_model.vpt_proj(
                        vpt_embeddings_deep[layer_idx - 1]).expand(B, -1, -1)).to(x.dtype)

                    vpt_emb_deep = vpt_emb_deep.permute(1, 0, 2)  # NLD -> LND

                    x = torch.cat((
                        x[:1, :, :],
                        vpt_emb_deep,
                        x[(1 + self.mvlpt_model.vpt_n_ctx):, :, :]
                    ), dim=0)

                    x_input = x.detach().permute(1, 0, 2).unsqueeze(0)
                    x = layer(x)

                    ######### EXTRACT FEATURE FOR EACH LAYER #########
                    if extract_layer:
                        '''
                        extract features
                        ext_x: layer's output feature, [L, N, D]
                        ext_vpt_emb_deep: input prompt vector, [L, N, D]
                        ext_visual: input image patch, [L, N, D]
                        '''
                        input_x.append(x_input)
                        ext_x.append(x.detach().permute(1, 0, 2).unsqueeze(0))
                        # ext_vpt_emb_deep.append(vpt_emb_deep.detach().permute(1, 0, 2).unsqueeze(0))
                        # ext_visual.append(x[(1+self.mvlpt_model.vpt_n_ctx):, :, :].detach().permute(1, 0, 2).unsqueeze(0))

                        # print('ImageEncoder ext_x', self.ext_x.shape)
                        # print('ImageEncoder ext_vpt_emb_deep', self.ext_vpt_emb_deep.shape)
                        # print('ImageEncoder ext_visual', self.ext_visual.shape)
                    ######### END EXTRACT FEATURE FOR EACH LAYER #########

                # x = ssf_ada(x, self.scale[layer_idx], self.shift[layer_idx])

        else:
            x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.visual.ln_post(x[:, 0, :])
        # x = ssf_ada(x, self.ssf_scale_4, self.ssf_shift_4)  # .to(x.type())

        if self.visual.proj is not None:
            x = x @ self.visual.proj
            # print('image encoder shape', x.shape, self.ssf_scale_5.shape, self.ssf_shift_5.shape)
            # x = ssf_ada(x, self.ssf_scale_5, self.ssf_shift_5)

        x = self.head(x)
        ######### EXTRACT FEATURE FOR EACH LAYER #########
        if extract_layer:
            ext_x = torch.cat(ext_x, 0).unsqueeze(0)  # [n_layer, B, token_len, dim]
            # ext_vpt_emb_deep = torch.cat(ext_vpt_emb_deep, 0)   # [n_layer, B, token_len, dim]
            # ext_visual = torch.cat(ext_visual, 0)   # [n_layer, B, token_len, dim]
            input_x = torch.cat(input_x, 0).unsqueeze(0)

            # print('ImageEncoder ext_x', ext_x.shape)
            # print('ImageEncoder input_x', input_x.shape)

            # compare prompt input for each layer
            # eq = []
            # for layer_idx in range(len(ext_vpt_emb_deep)):
            #     for i in range(len(ext_vpt_emb_deep[layer_idx])):
            #         eq.append(torch.equal(ext_vpt_emb_deep[layer_idx][0],
            #                               ext_vpt_emb_deep[layer_idx][i]))
            #
            # print('ImageEncoder test eq:', eq)  # ALL TRUE for EACH LAYER. WHY???

            return x, ext_x, input_x
        ######### END EXTRACT FEATURE FOR EACH LAYER #########

        else:
            return x


# progressive
# class ImageEncoder(nn.Module):
#     def __init__(self, clip_model, mvlpt_model):
#         super().__init__()
#         # HACK: Assume all is vision transformer
#         self.visual = clip_model.visual
#         self.mvlpt_model = mvlpt_model
#
#     def forward(self, x: torch.Tensor, vpt_embeddings=None, vpt_embeddings_deep=None, vpt_task_embeddings=None, vpt_task_embeddings_deep=None, alphas=None, task=None, extract_layer=False):
#         ### progressive
#         task_unique = torch.unique(task)
#         task_unique_list = task_unique.tolist()
#         idx = torch.tensor([task_unique_list.index(x) for x in task])
#
#         # layer 0
#         alphas = F.sigmoid(alphas)
#         alpha_0 = alphas[task_unique, 0]
#         vpt_embeddings = torch.cat(len(task_unique) * [vpt_embeddings], 0)
#         vpt_task_embeddings = vpt_task_embeddings[task_unique].squeeze(1)
#         # print('image encoder 0 vpt_embeddings shape', vpt_embeddings.shape)
#         # print('image encoder 0 vpt_task_embeddings shape', vpt_task_embeddings.shape)
#         assert vpt_embeddings.shape == vpt_task_embeddings.shape
#         vpt_embeddings_0 = vpt_embeddings * alpha_0[:, None, None] + vpt_task_embeddings * (1 - alpha_0[:, None, None])
#         vpt_embeddings_0 = torch.cat([vpt_embeddings_0[x].unsqueeze(0) for x in idx])
#         # print('image encoder 0 vpt_embeddings_0 shape', vpt_embeddings_0.shape)
#         ### end progressive
#
#         x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.visual.positional_embedding.to(x.dtype)
#         x = self.visual.ln_pre(x)
#
#         B = x.shape[0]
#
#         ## progressive
#         x, ctx = self.mvlpt_model.forward_vpt(x, vpt_embeddings_0)
#         ### end progressive
#
#         x = x.permute(1, 0, 2)  # NLD -> LND
#
#         input_x = []
#         ext_x = []
#         if self.mvlpt_model.vpt_deep and (
#                 vpt_embeddings_deep is not None or self.mvlpt_model.vpt_embeddings_deep is not None):
#             if vpt_embeddings_deep is None:
#                 vpt_embeddings_deep = self.mvlpt_model.vpt_embeddings_deep
#
#             for layer_idx in range(self.visual.transformer.layers):
#                 layer = self.visual.transformer.resblocks[layer_idx]
#
#                 if layer_idx == 0:
#                     x = layer(x)
#                 elif layer_idx <= vpt_embeddings_deep.shape[0]:
#                     vpt_emb_deep = self.mvlpt_model.vpt_dropout(self.mvlpt_model.vpt_proj(
#                         vpt_embeddings_deep[layer_idx - 1]).unsqueeze(0)).to(x.dtype)
#
#                     ### progressive
#                     # print('image encoder 1 vpt_emb_deep shape', vpt_emb_deep.shape)
#                     alpha_i = alphas[task_unique, layer_idx]
#                     vpt_emb_deep = torch.cat(len(task_unique) * [vpt_emb_deep],0)
#
#                     vpt_task_emb_deep = self.mvlpt_model.vpt_dropout(self.mvlpt_model.vpt_proj(
#                         vpt_task_embeddings_deep[task_unique, layer_idx - 1])).to(x.dtype)
#                     # print('image encoder 2 vpt_task_emb_deep shape', vpt_task_emb_deep.shape)
#                     assert vpt_emb_deep.shape == vpt_task_emb_deep.shape
#
#                     vpt_emb_deep_i = vpt_emb_deep * alpha_i[:, None, None] + vpt_task_emb_deep * (1 - alpha_i[:, None, None])
#                     vpt_emb_deep_i = torch.cat([vpt_emb_deep_i[x].unsqueeze(0) for x in idx])
#                     vpt_emb_deep_i = vpt_emb_deep_i.permute(1, 0, 2)  # NLD -> LND
#                     # print('image encoder 3 vpt_emb_deep_i shape', vpt_emb_deep_i.shape)
#                     ### end progressive
#
#                     x = torch.cat((
#                         x[:1, :, :],
#                         ### progressive
#                         vpt_emb_deep_i,
#                         ### end progressive
#                         x[(1 + self.mvlpt_model.vpt_n_ctx):, :, :]
#                     ), dim=0)
#
#                     x_input = x.detach().permute(1, 0, 2).unsqueeze(0)
#                     x = layer(x)
#
#                     ######### EXTRACT FEATURE FOR EACH LAYER #########
#                     if extract_layer:
#                         '''
#                         extract features
#                         ext_x: layer's output feature, [L, N, D]
#                         ext_vpt_emb_deep: input prompt vector, [L, N, D]
#                         ext_visual: input image patch, [L, N, D]
#                         '''
#                         input_x.append(x_input)
#                         ext_x.append(x.detach().permute(1, 0, 2).unsqueeze(0))
#                         # ext_vpt_emb_deep.append(vpt_emb_deep.detach().permute(1, 0, 2).unsqueeze(0))
#                         # ext_visual.append(x[(1+self.mvlpt_model.vpt_n_ctx):, :, :].detach().permute(1, 0, 2).unsqueeze(0))
#
#                         # print('ImageEncoder ext_x', self.ext_x.shape)
#                         # print('ImageEncoder ext_vpt_emb_deep', self.ext_vpt_emb_deep.shape)
#                         # print('ImageEncoder ext_visual', self.ext_visual.shape)
#                     ######### END EXTRACT FEATURE FOR EACH LAYER #########
#         else:
#             x = self.visual.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#
#         x = self.visual.ln_post(x[:, 0, :])
#
#         if self.visual.proj is not None:
#             x = x @ self.visual.proj
#
#         ######### EXTRACT FEATURE FOR EACH LAYER #########
#         if extract_layer:
#             ext_x = torch.cat(ext_x, 0).unsqueeze(0)  # [n_layer, B, token_len, dim]
#             # ext_vpt_emb_deep = torch.cat(ext_vpt_emb_deep, 0)   # [n_layer, B, token_len, dim]
#             # ext_visual = torch.cat(ext_visual, 0)   # [n_layer, B, token_len, dim]
#             input_x = torch.cat(input_x, 0).unsqueeze(0)
#
#             # print('ImageEncoder ext_x', ext_x.shape)
#             # print('ImageEncoder input_x', input_x.shape)
#
#             # compare prompt input for each layer
#             # eq = []
#             # for layer_idx in range(len(ext_vpt_emb_deep)):
#             #     for i in range(len(ext_vpt_emb_deep[layer_idx])):
#             #         eq.append(torch.equal(ext_vpt_emb_deep[layer_idx][0],
#             #                               ext_vpt_emb_deep[layer_idx][i]))
#             #
#             # print('ImageEncoder test eq:', eq)  # ALL TRUE for EACH LAYER. WHY???
#
#             return x, ext_x, input_x
#         ######### END EXTRACT FEATURE FOR EACH LAYER #########
#
#         else:
#             return x

# class TextEncoder(nn.Module):
#     def __init__(self, clip_model, cfg=None):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype
#         self.cfg = cfg
#
#     def forward(self, prompts, tokenized_prompts):
#         if not self.cfg.TRAINER.CUT_CONTEXTLEN:
#             x = prompts + self.positional_embedding.type(self.dtype)
#             x = x.permute(1, 0, 2)  # NLD -> LND
#             x = self.transformer(x)
#             x = x.permute(1, 0, 2)  # LND -> NLD
#         else:
#             x = prompts + self.positional_embedding.type(self.dtype)[:prompts.shape[1], :]
#             x = x.permute(1, 0, 2)  # NLD -> LND
#
#             for block in self.transformer.resblocks:
#                 if block.attn_mask.shape[0] != x.shape[0]:
#                     block.attn_mask = block.attn_mask[:x.shape[0], :x.shape[0]]
#             # x = self.transformer(x)
#             from torch.utils.checkpoint import checkpoint_sequential
#             act_chunk_size = min(self.cfg.TRAINER.ACT_CKPT, len(self.transformer.resblocks))
#             x = checkpoint_sequential(self.transformer.resblocks, act_chunk_size, x)
#             x = x.permute(1, 0, 2)  # LND -> NLD
#
#         x = self.ln_final(x).type(self.dtype)
#
#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
#
#         return x

from torch.nn import Dropout
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair


# progressive
# class Alphalayers(nn.Module):
#     def __init__(self, n_tasks, n_layers,  dtype):
#         super().__init__()
#
#         self.alphas = nn.Parameter(torch.rand((n_tasks, n_layers), dtype=dtype))
#
#     def forward(self):
#         alphas = F.sigmoid(self.alphas)
#         return alphas

# multi prompts
# class Alphalayers(nn.Module):
#     def __init__(self, n_prompts, dtype):
#         super().__init__()
#
#         self.a = nn.Parameter(torch.ones(n_prompts, dtype=dtype))
#
#     def forward(self):
#         alphas = F.softmax(self.a)
#         return alphas

# multi prompts
# class MPrompts(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # self.vpt_embeddings = nn.Parameter(torch.zeros(
#         #     n_prompts, vpt_n_ctx, vpt_dim, dtype=dtype))
#         # nn.init.uniform_(self.vpt_embeddings.data, -val, val)
#         #
#         # self.alphas = nn.Parameter(torch.ones(n_prompts))
#         #
#         # self.vpt_embeddings_deep = nn.Parameter(torch.zeros(
#         #     sn_prompts, vision_layers - 1, vpt_n_ctx, vpt_dim, dtype=dtype))
#         # # xavier_uniform initialization
#         # nn.init.uniform_(self.vpt_embeddings_deep.data, -val, val)
#         #
#         # self.alphas_deep = nn.Parameter(torch.ones(n_prompts))
#
#         # self.a = nn.Parameter(torch.ones(n_prompts, dtype=dtype))
#         self.softmax = nn.Softmax()
#         self.softmax_deep = nn.Softmax()
#
#     def forward(self, vpt_embeddings, vpt_embeddings_deep, alphas, alphas_deep):
#
#         # alphas = self.softmax(alphas)
#         # vpt_emb = alphas[:, None, None].to(alphas.device) * vpt_embeddings
#         vpt_emb = self.softmax(alphas[:, None, None]) * vpt_embeddings
#         vpt_emb = vpt_emb.sum(0).unsqueeze(0)
#
#         # alphas_deep = self.softmax_deep(alphas_deep)
#         # vpt_emb_deep = alphas_deep[:, None, None, None].to(alphas_deep.device) * vpt_embeddings_deep
#         vpt_emb_deep = self.softmax_deep(alphas_deep[:, None, None, None]) * vpt_embeddings_deep
#         vpt_emb_deep = vpt_emb_deep.sum(0)
#
#         return vpt_emb, vpt_emb_deep


class MultitaskVLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        # DEFAULT is VPT
        n_cls = len(classnames)
        coop_n_ctx = cfg.TRAINER.MVLPT.COOP.N_CTX
        cocoop_n_ctx = cfg.TRAINER.MVLPT.COCOOP.N_CTX
        vpt_n_ctx = cfg.TRAINER.MVLPT.VPT.N_CTX

        coop_ctx_init = cfg.TRAINER.MVLPT.COOP.CTX_INIT
        cocoop_ctx_init = cfg.TRAINER.MVLPT.COCOOP.CTX_INIT
        vpt_ctx_init = cfg.TRAINER.MVLPT.VPT.CTX_INIT

        dtype = clip_model.dtype
        coop_ctx_dim = clip_model.ln_final.weight.shape[0]
        cocoop_ctx_dim = coop_ctx_dim
        vpt_ctx_dim = clip_model.visual.conv1.weight.shape[0]

        vis_dim = clip_model.visual.output_dim

        # HACK: this is for VisualTransformer model
        clip_patchsize = clip_model.visual.conv1.weight.shape[-1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ### progressive
        # self.n_tasks = len(cfg.DATASET.DATASET.split(','))
        ### end progressive

        self.vpt_dropout = Dropout(cfg.TRAINER.MVLPT.VPT.DROPOUT)
        self.vpt_deep = cfg.TRAINER.MVLPT.VPT.DEEP
        self.vpt_embeddings = None
        self.vpt_embeddings_deep = None

        ### multi prompts
        # self.n_prompts = cfg.TRAINER.N_PROMPTS
        ### end multi prompts

        ### progressive
        # self.vpt_task_embeddings = None
        # self.vpt_task_embeddings_deep = None
        # self.vpt_task_alpha = None
        ### end progressive
        if vpt_n_ctx != 0:
            if cfg.TRAINER.MVLPT.VPT.PROJECT > -1:
                vpt_dim = cfg.TRAINER.MVLPT.VPT.PROJECT
                self.vpt_proj = nn.Linear(
                    vpt_dim, vpt_ctx_dim).type(dtype)
                nn.init.kaiming_normal_(
                    self.vpt_proj.weight, a=0, mode='fan_out')
            else:
                vpt_dim = vpt_ctx_dim
                self.vpt_proj = nn.Identity()

            if vpt_ctx_init:
                # Don't support ctx init for MVLPT
                raise ValueError("CTX initiation scheme is not supported")
            else:
                # random initialization
                clip_patchsize = _pair(clip_patchsize)
                val = math.sqrt(6. / float(3 * reduce(mul, clip_patchsize, 1) + vpt_dim))  # noqa

                self.vpt_embeddings = nn.Parameter(torch.zeros(
                    1, vpt_n_ctx, vpt_dim, dtype=dtype))
                # xavier_uniform initialization
                nn.init.uniform_(self.vpt_embeddings.data, -val, val)

                ### multi prompts
                # self.vpt_embeddings_mprompts = nn.Parameter(torch.zeros(
                #     self.n_prompts, vpt_n_ctx, vpt_dim, dtype=dtype))
                # # xavier_uniform initialization
                # nn.init.uniform_(self.vpt_embeddings_mprompts.data, -val, val)
                #
                # self.alphas = nn.Parameter(torch.rand(self.n_prompts, dtype=dtype))
                ### end multi prompts

                ### progressive
                # self.vpt_task_embeddings = nn.Parameter(torch.zeros(
                #     self.n_tasks, 1, vpt_n_ctx, vpt_dim, dtype=dtype))
                # nn.init.uniform_(self.vpt_task_embeddings.data, -val, val)
                ### end progressive

                if self.vpt_deep:  # noqa
                    self.vision_layers = len([k for k in clip_model.state_dict().keys() if
                                              k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

                    self.vpt_embeddings_deep = nn.Parameter(torch.zeros(
                        self.vision_layers - 1, vpt_n_ctx, vpt_dim, dtype=dtype))
                    # 9-1, vpt_n_ctx, vpt_dim, dtype=dtype))
                    # xavier_uniform initialization
                    nn.init.uniform_(self.vpt_embeddings_deep.data, -val, val)

                    ### multi prompts
                    # self.vpt_embeddings_deep_mprompts = nn.Parameter(torch.zeros(
                    #     self.n_prompts, self.vision_layers-1, vpt_n_ctx, vpt_dim, dtype=dtype))
                    # # xavier_uniform initialization
                    # nn.init.uniform_(self.vpt_embeddings_deep_mprompts.data, -val, val)
                    #
                    # self.alphas_deep = nn.Parameter(torch.rand(self.n_prompts, dtype=dtype))
                    ### end multi prompts

                    ### progressive
                    # self.vpt_task_embeddings_deep = nn.Parameter(torch.zeros(
                    #     self.n_tasks, self.vision_layers - 1, vpt_n_ctx, vpt_dim, dtype=dtype))
                    # # xavier_uniform initialization
                    # nn.init.uniform_(self.vpt_task_embeddings_deep.data, -val, val)
                    #
                    # self.vpt_task_alpha = nn.Parameter(torch.rand((self.n_tasks, self.vision_layers),dtype=dtype))
                    # self.vpt_task_alpha = Alphalayers(self.n_tasks, self.vision_layers, dtype)
                    ### end progressive

                prompt_prefix = "a photo of a "

                print(f'VPT Initial context: "{prompt_prefix}"')
                print(f"VPT Number of context words (tokens): {vpt_n_ctx}")

        self.ctx = None
        if coop_n_ctx != 0:
            if coop_ctx_init:
                # use given words to initialize context vectors
                coop_ctx_init = coop_ctx_init.replace("_", " ")
                coop_n_ctx = len(coop_ctx_init.split(" "))
                prompt = clip.tokenize(coop_ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + coop_n_ctx, :]
                prompt_prefix = coop_ctx_init
            else:
                # random initialization
                if cfg.TRAINER.MVLPT.COOP.CSC:
                    print("Initializing class-specific contexts")
                    ctx_vectors = torch.empty(n_cls, coop_n_ctx, coop_ctx_dim, dtype=dtype)
                else:
                    print("Initializing a generic context")
                    ctx_vectors = torch.empty(coop_n_ctx, coop_ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * coop_n_ctx)

            print(f'COOP Initial context: "{prompt_prefix}"')
            print(f"COOP Number of context words (tokens): {coop_n_ctx}")

            self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.mvlpt_proj = nn.Identity()
        if vpt_n_ctx != 0 and coop_n_ctx != 0:
            self.mvlpt_proj_ctx_dim = cfg.TRAINER.MVLPT.PROJECT_DIM

            if cfg.TRAINER.MVLPT.PROJECT_METHOD == 'identity':
                self.mvlpt_proj = nn.Identity()
            else:
                # match dimension
                self.mvlpt_proj_ctx_vpt_pre, self.mvlpt_proj_ctx_vpt_post = nn.Identity(), nn.Identity()
                self.mvlpt_proj_ctx_coop_pre, self.mvlpt_proj_ctx_coop_post = nn.Identity(), nn.Identity()

                if coop_ctx_dim != self.mvlpt_proj_ctx_dim:
                    self.mvlpt_proj_ctx_coop_pre = nn.Linear(coop_ctx_dim, self.mvlpt_proj_ctx_dim, dtype=dtype)
                    self.mvlpt_proj_ctx_coop_post = nn.Linear(self.mvlpt_proj_ctx_dim, coop_ctx_dim, dtype=dtype)
                if vpt_ctx_dim != self.mvlpt_proj_ctx_dim:
                    self.mvlpt_proj_ctx_vpt_pre = nn.Linear(vpt_ctx_dim, self.mvlpt_proj_ctx_dim, dtype=dtype)
                    self.mvlpt_proj_ctx_vpt_post = nn.Linear(self.mvlpt_proj_ctx_dim, vpt_ctx_dim, dtype=dtype)

                if cfg.TRAINER.MVLPT.PROJECT_METHOD == 'mlp':
                    self.mvlpt_proj = nn.GeLU()

                elif cfg.TRAINER.MVLPT.PROJECT_METHOD == 'transformer':
                    from clip.model import Transformer
                    self.mvlpt_proj = Transformer(width=self.mvlpt_proj_ctx_dim, layers=1, heads=1)
                    # for n, m in self.MVLPT_proj.named_modules():
                    #     m.type(dtype)
        self.cocoop_ctx = None
        if cocoop_n_ctx != 0:
            if cocoop_ctx_init:
                # use given words to initialize context vectors
                cocoop_ctx_init = cocoop_ctx_init.replace("_", " ")
                cocoop_n_ctx = len(cocoop_ctx_init.split(" "))
                prompt = clip.tokenize(cocoop_ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + cocoop_n_ctx, :]
                prompt_prefix = cocoop_ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(cocoop_n_ctx, cocoop_ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * cocoop_n_ctx)

            print(f'Initial context: "{prompt_prefix}"')
            print(f"Number of context words (tokens): {cocoop_n_ctx}")

            self.cocoop_ctx = nn.Parameter(ctx_vectors)

            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(vis_dim // 16, cocoop_ctx_dim))
            ]))

            if cfg.TRAINER.MVLPT.COCOOP.PREC == "fp16":
                self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        if cfg.TRAINER.CUT_CONTEXTLEN:
            sot_token = _tokenizer.encoder["<|startoftext|>"]
            eot_token = _tokenizer.encoder["<|endoftext|>"]
            max_length = min(clip_model.context_length,
                             max([len([sot_token] + _tokenizer.encode(p) + [eot_token]) for p in prompts]))
        else:
            max_length = clip_model.context_length
        print("Current Context Length is: ", max_length)
        # exit()
        tokenized_prompts = torch.cat([clip.tokenize(p, context_length=max_length) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if cocoop_n_ctx != 0:
            self.register_buffer("token_suffix", embedding[:, 1 + cocoop_n_ctx:, :])  # CLS, EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + coop_n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.vpt_n_ctx = vpt_n_ctx
        self.coop_n_ctx = coop_n_ctx
        self.cocoop_n_ctx = cocoop_n_ctx

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.MVLPT.COOP.CLASS_TOKEN_POSITION

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward_cocoop(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.cocoop_ctx  # (n_ctx, ctx_dim)
        if ctx is None:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)

        return prompts

    ### multi prompts
    # def forward_mprompts(self):
    #     '''
    #     convex combination of n_prompts
    #     :param vpt_emb: [n_prompt, len, dim]
    #     :param vpt_emb_deep: [n_prompt, n_layers, len, dim]
    #     :param scale: [n_prompt]
    #     :return: vpt_emb [1, len, dim], vpt_emb_deep [n_layers, len, dim]
    #     '''
    #
    #     # # alphas = F.softmax(self.alphas)
    #     # vpt_emb = F.softmax(self.alphas)[:, None, None].to(self.alphas.device) * self.vpt_embeddings_mprompts
    #     # vpt_emb = vpt_emb.sum(0).unsqueeze(0)
    #     #
    #     # # alphas_deep = F.softmax(alpha_deep)
    #     # vpt_emb_deep = F.softmax(self.alphas_deep)[:, None, None, None].to(self.alphas_deep.device) * self.vpt_embeddings_deep_mprompts
    #     # vpt_emb_deep = vpt_emb_deep.sum(0)
    #
    #     vpt_emb = torch.einsum('ijk,i->ijk', [self.vpt_embeddings_mprompts, F.softmax(self.alphas)]).sum(0).unsqueeze(0)
    #     vpt_emb_deep = torch.einsum('ijkf,i->ijkf', [self.vpt_embeddings_deep_mprompts, F.softmax(self.alphas_deep)]).sum(0)
    #
    #     return vpt_emb, vpt_emb_deep

    def forward_mvlpt_proj(self, dtype=torch.float):
        if self.coop_n_ctx == 0 or isinstance(self.mvlpt_proj, nn.Identity) or self.vpt_n_ctx == 0:
            # print('forward_mvlpt_proj', self.coop_n_ctx == 0, isinstance(self.mvlpt_proj, nn.Identity), self.vpt_n_ctx == 0)
            # vpt_embeddings, vpt_embeddings_deep = self.vpt_embeddings, self.vpt_embeddings_deep
            # vpt_embeddings, vpt_embeddings_deep = self.forward_mprompts()
            # vpt_embeddings, vpt_embeddings_deep = self.m_prompts(self.vpt_embeddings_mprompts, self.vpt_embeddings_deep_mprompts,
            #                                                      self.alphas, self.alphas_deep)
            return self.ctx, self.vpt_embeddings, self.vpt_embeddings_deep  # , self.vpt_task_embeddings, self.vpt_task_embeddings_deep, self.vpt_task_alpha

        # print('vpt', self.vpt_embeddings.dtype, 'vpt_proj', self.MVLPT_proj_ctx_vpt_pre.weight.dtype)
        # print('coop_emb', self.vpt_embeddings.dtype, 'coop_emb_proj', self.MVLPT_proj_ctx_vpt_pre.weight.dtype)

        vpt_emb = self.vpt_embeddings  # 1*vpt_n_ctx*vpt_ctx_dim
        if self.vpt_deep:
            vpt_emb = torch.cat([vpt_emb, self.vpt_embeddings_deep], dim=0)  # vision_layers*vpt_n_ctx*vpt_ctx_dim

        ### multi prompts
        # vpt_emb = self.vpt_embeddings # 1*vpt_n_ctx*vpt_ctx_dim
        # vpt_emb_deep = self.vpt_embeddings_deep
        #
        # vpt_emb, vpt_emb_deep = self.forward_prompts(vpt_emb, vpt_emb_deep,
        #                                              self.alphas, self.alphas_deep)
        #
        # if self.vpt_deep:
        #     vpt_emb = torch.cat([ vpt_emb, vpt_emb_deep ], dim=0) # vision_layers*vpt_n_ctx*vpt_ctx_dim
        ### end multi prompts

        ### progressive
        # project must be nn.Identity()
        # vpt_alpha = self.vpt_task_alpha
        # vpt_task_emb = self.vpt_task_embeddings
        # vpt_task_emb_deep = self.vpt_task_embeddings_deep
        ### end progressive

        vpt_ctx_dim = vpt_emb.shape[-1]
        vpt_emb = vpt_emb.reshape(1, -1, vpt_ctx_dim)

        coop_emb = self.ctx  # n_ctx, ctx_dim or n_cls, n_ctx, ctx_dim
        coop_ctx_dim = self.ctx.shape[-1]

        if coop_emb.dim() == 2:
            coop_emb = coop_emb.unsqueeze(0)
        coop_emb = coop_emb.reshape(1, -1, coop_ctx_dim)

        coop_emb_n_ctx = coop_emb.shape[1]

        # match dimension
        coop_emb = self.mvlpt_proj_ctx_coop_pre(coop_emb)
        vpt_emb = self.mvlpt_proj_ctx_vpt_pre(vpt_emb)

        mvlpt_emb = torch.cat([coop_emb, vpt_emb], dim=1)

        # print('mvlpt_emb', mvlpt_emb.dtype, 'mvlpt_emb_proj', self.MVLPT_proj.resblocks[0].attn.in_proj_weight.dtype)
        mvlpt_emb = self.mvlpt_proj(mvlpt_emb.float())  # nn.Identity()
        mvlpt_emb = mvlpt_emb.type(dtype)
        coop_emb, vpt_emb = mvlpt_emb[:, :coop_emb_n_ctx, :], mvlpt_emb[:, coop_emb_n_ctx:, :]

        coop_emb = self.mvlpt_proj_ctx_coop_post(coop_emb).reshape(-1, self.coop_n_ctx, coop_ctx_dim).squeeze(0)
        vpt_emb = self.mvlpt_proj_ctx_vpt_post(vpt_emb).reshape(-1, self.vpt_n_ctx, vpt_ctx_dim)
        vpt_emb_deep = None if vpt_emb.shape[0] == 1 else vpt_emb[1:, :, :]
        vpt_emb = vpt_emb[0, :, :].unsqueeze(0)
        return coop_emb, vpt_emb, vpt_emb_deep  # , vpt_task_emb, vpt_task_emb_deep, vpt_alpha

    def forward_vpt(self, x, vpt_embeddings=None):
        B = x.shape[0]  # (batch_size, 1 + n_patches, hidden_dim)

        if vpt_embeddings is None:
            if self.vpt_embeddings is None:
                return x
            vpt_embeddings = self.vpt_embeddings

        ctx = self.vpt_dropout(self.vpt_proj(vpt_embeddings).expand(B, -1, -1)).to(x.dtype)
        prefix = x[:, :1, :]
        suffix = x[:, 1:, :]

        prompts = torch.cat(
            [
                prefix,  # (B, 1, dim)
                ctx,  # (B, n_ctx, dim)
                suffix,  # (B, n_patches, dim)
            ],
            dim=1,
        )

        return prompts, ctx

    def forward_coop(self, ctx=None):
        if ctx is None:
            ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        if ctx is None:
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.coop_n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, dm=None):
        super().__init__()
        self.prompt_learner = MultitaskVLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model, self.prompt_learner, classnames)
        # self.text_encoder = TextEncoder(clip_model, cfg)
        # self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.multi_task_label_pertask = cfg.DATASET.MULTITASK_LABEL_PERTASK
        if self.multi_task_label_pertask:
            self.class_index_pertask_start = torch.arange(dm._num_classes)
            self.class_index_pertask_end = torch.arange(dm._num_classes)
            start_index = 0

            for class_index, task in enumerate(dm._task_names):
                class_num = len(dm._labelmap[task])
                self.class_index_pertask_start[class_index] = start_index
                start_index += class_num
                self.class_index_pertask_end[class_index] = start_index
            self.index = torch.arange(dm._num_classes).unsqueeze(0)

    def forward(self, image, task=None):
        coop_emb, vpt_emb, vpt_emb_deep = self.prompt_learner.forward_mvlpt_proj(self.dtype)
        # coop_emb, vpt_emb, vpt_emb_deep, vpt_task_embeddings, vpt_task_embeddings_deep, vpt_task_alpha = self.prompt_learner.forward_mvlpt_proj(self.dtype)

        # print('CustomCLIP vpt_emb', vpt_emb.shape)
        # print('CustomCLIP vpt_embeddings_deep', vpt_emb_deep.shape)
        # print('CustomCLIP alphas', alphas)
        # print('CustomCLIP alphas_deep', alphas_deep)

        logits = self.image_encoder(image.type(self.dtype), vpt_emb, vpt_emb_deep)
        # image_features = self.image_encoder(image.type(self.dtype), vpt_emb, vpt_emb_deep, vpt_task_embeddings,
        #                                     vpt_task_embeddings_deep, vpt_task_alpha, task)
        # image_features = self.image_encoder(image.type(self.dtype), vpt_emb, vpt_emb_deep, task)
        # image_features, ext_x, input_x = self.image_encoder(image.type(self.dtype), vpt_emb, vpt_emb_deep)

        # if self.prompt_learner.cocoop_ctx == None:
        #     prompts = self.prompt_learner.forward_coop(coop_emb)
        #     tokenized_prompts = self.tokenized_prompts
        #     text_features = self.text_encoder(prompts, tokenized_prompts)
        #
        #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #
        #     logit_scale = self.logit_scale.exp()
        #     logits = logit_scale * image_features @ text_features.t()
        #
        # else:
        #
        #     tokenized_prompts = self.tokenized_prompts
        #     logit_scale = self.logit_scale.exp()
        #
        #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #
        #     prompts = self.prompt_learner.forward_cocoop(image_features)
        #
        #     logits = []
        #     for pts_i, imf_i in zip(prompts, image_features):
        #         text_features = self.text_encoder(pts_i, tokenized_prompts)
        #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        #         l_i = logit_scale * imf_i @ text_features.t()
        #         logits.append(l_i)
        #     logits = torch.stack(logits)

        if self.multi_task_label_pertask:
            # Here we perform prompt selection
            domain_start_indexs = self.class_index_pertask_start[task].unsqueeze(-1)
            domain_end_indexs = self.class_index_pertask_end[task].unsqueeze(-1)
            # print(domain_start_indexs.shape, domain_end_indexs.shape, logits.shape)
            select_index = self.index.repeat(logits.shape[0], 1)
            select_index = (select_index >= domain_start_indexs).float() * (select_index < domain_end_indexs).float()
            # exit()
            logits = logits * select_index.to(logits.device)

        return logits
        # return logits, ext_x, input_x


class MVLPTCOOPDataManager(DataManager):

    def __init__(self, cfg, custom_tfm_train=None, custom_tfm_test=None, dataset_wrapper=None):
        # Load dataset
        label_offset = 0
        self.num_classes_list = []
        self.classnames_list = []
        self.lab2cname_list = {}
        self.dataset = None
        self._task_names = cfg.DATASET.DATASET.split(',')
        self._id2task = {}
        self._task_class_idx = {}

        for domain, dataset_name in enumerate(self._task_names):
            cfg.defrost()
            cfg.DATASET.NAME = dataset_name
            cfg.freeze()
            self._id2task[domain] = dataset_name
            dataset = build_dataset(cfg)
            self.num_classes_list.append(dataset._num_classes)
            self.classnames_list += dataset._classnames
            new_lab2cname_dict = {}
            for key, value in dataset._lab2cname.items():
                new_lab2cname_dict[key + label_offset] = value
            self.lab2cname_list.update(new_lab2cname_dict)
            for i in range(len(dataset._train_x)):
                dataset._train_x[i]._label += label_offset
                dataset._train_x[i]._domain = domain

            if dataset._train_u:
                for i in range(len(dataset._train_u)):
                    dataset._train_u[i]._label += label_offset
                    dataset._train_u[i]._domain = domain
                if self.dataset is not None:
                    self.dataset._train_u = self.dataset._train_u + dataset._train_u
            if dataset._val:
                for i in range(len(dataset._val)):
                    dataset._val[i]._label += label_offset
                    dataset._val[i]._domain = domain

            for i in range(len(dataset._test)):
                dataset._test[i]._label += label_offset
                dataset._test[i]._domain = domain

            if self.dataset is not None:
                self.dataset._train_x = self.dataset._train_x + dataset._train_x
                self.dataset._val = self.dataset.val + dataset.val
                self.dataset._test = self.dataset.test + dataset.test

            print(dataset._train_u is None, dataset._val is None)
            if self.dataset is None:
                self.dataset = dataset

            self._task_class_idx[dataset_name] = (label_offset, label_offset + dataset._num_classes)
            label_offset += dataset._num_classes

        dataset = self.dataset
        dataset._classnames = self.classnames_list
        dataset._lab2cname = self.lab2cname_list
        dataset._num_classes = sum(self.num_classes_list)
        print(self.num_classes_list, len(dataset._classnames), dataset._lab2cname, dataset._num_classes)

        # Build transform
        if custom_tfm_train is None:
            tfm_train = build_transform(cfg, is_train=True)
        else:
            print("* Using custom transform for training")
            tfm_train = custom_tfm_train

        if custom_tfm_test is None:
            tfm_test = build_transform(cfg, is_train=False)
        else:
            print("* Using custom transform for testing")
            tfm_test = custom_tfm_test

        # Build train_loader_x
        train_loader_x = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            # sampler_type="RandomDomainSampler",
            # sampler_type="RandomDomainSampler_K",
            data_source=dataset.train_x,
            batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            # n_domain=1,
            n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )

        ### progressive
        # train_loader_x_share = build_data_loader(
        #     cfg,
        #     # sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
        #     sampler_type="RandomDomainSampler",
        #     # sampler_type="RandomDomainSampler_K",
        #     data_source=dataset.train_x,
        #     batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        #     # n_domain=cfg.DATALOADER.TRAIN_X.N_DOMAIN,
        #     n_domain=8,
        #     n_ins=cfg.DATALOADER.TRAIN_X.N_INS,
        #     tfm=tfm_train,
        #     is_train=True,
        #     dataset_wrapper=dataset_wrapper
        # )
        #
        # self.train_loader_x_share = train_loader_x_share

        # Build train_loader_u
        train_loader_u = None
        if dataset.train_u:
            sampler_type_ = cfg.DATALOADER.TRAIN_U.SAMPLER
            batch_size_ = cfg.DATALOADER.TRAIN_U.BATCH_SIZE
            n_domain_ = cfg.DATALOADER.TRAIN_U.N_DOMAIN
            n_ins_ = cfg.DATALOADER.TRAIN_U.N_INS

            if cfg.DATALOADER.TRAIN_U.SAME_AS_X:
                sampler_type_ = cfg.DATALOADER.TRAIN_X.SAMPLER
                batch_size_ = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
                n_domain_ = cfg.DATALOADER.TRAIN_X.N_DOMAIN
                n_ins_ = cfg.DATALOADER.TRAIN_X.N_INS

            train_loader_u = build_data_loader(
                cfg,
                sampler_type=sampler_type_,
                data_source=dataset.train_u,
                batch_size=batch_size_,
                n_domain=n_domain_,
                n_ins=n_ins_,
                tfm=tfm_train,
                is_train=True,
                dataset_wrapper=dataset_wrapper
            )
        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader_x = train_loader_x
        self.train_loader_u = train_loader_u
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

        # print('CHECK DATASET LEN')
        # print('TRAIN', len(dataset.train_x))
        # print('VAL', len(dataset.val))
        # print('TEST', len(dataset.test))
        # if dataset.train_u:
        #     print('TRAIN U ', len(dataset.train_u))


from trainers.vision_benchmark.datasets import class_map_metric, get_metric
import random


class MVLPTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, class_map, train_dataset = construct_dataloader(cfg)

        self._metric = get_metric(class_map_metric[cfg.DATASET.DATASET])
        self._metric_name = class_map_metric[cfg.DATASET.DATASET]

        # Attributes
        self._num_classes = len(class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}
        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in enumerate(class_map):
            if isinstance(value, list):
                # value = random.choice(value)
                value = value[0]
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            pass
            # self.show_dataset_summary(cfg)


class MVLPTMTDataManager(DataManager):

    def __init__(self, cfg):
        # Load dataset
        train_loader_x, val_loader, test_loader, train_dataset, test_dataloader_by_task = construct_multitask_dataset(
            cfg)

        self._labelmap = train_dataset.labelmap
        self._task_names = train_dataset._task_names
        self._task2id = {v: k for k, v in enumerate(self._task_names)}
        self._id2task = {k: v for k, v in enumerate(self._task_names)}
        self._metric = {task: get_metric(class_map_metric[task]) for task in self._task_names}
        self._metric_name = {task: class_map_metric[task] for task in self._task_names}

        class_idx = 0
        self._task_class_idx = {}
        for task in self._task_names:
            class_num = len(self._labelmap[task])
            self._task_class_idx[task] = (class_idx, class_idx + class_num)
            class_idx += class_num

        from trainers.vision_benchmark.datasets import class_map

        print(self._task_names)
        print(self._labelmap)
        print(class_map.keys())

        mt_class_map = dict()
        for task in self._labelmap:
            for label_idx, label in enumerate(class_map[task]):
                cnt = train_dataset._get_cid(label_idx, task)
                mt_class_map[cnt] = label

        print(mt_class_map)
        # Attributes
        self._num_classes = len(mt_class_map)
        self._num_source_domains = len(cfg.DATASET.SOURCE_DOMAINS)
        self._lab2cname = {}

        # random.seed(cfg.DATASET.RANDOM_SEED_SAMPLING)
        for key, value in mt_class_map.items():
            if isinstance(value, list):
                value = value[0]  # random.choice(value)
            self._lab2cname[key] = value

        # Dataset and data-loaders
        # self.dataset.train_x = train_dataset
        self.train_loader_x = train_loader_x
        # self.train_loader_u = train_loader_u
        self.train_loader_u = None
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_dataloader_by_task = test_dataloader_by_task
        if cfg.VERBOSE:
            pass


@TRAINER_REGISTRY.register()
class MVLPT_ss_bias_last_trainer(TrainerX):
    """Context Optimization (MVLPT).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.MVLPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        if self.cfg.DATASET.COOP:
            classnames = self.dm.dataset.classnames
        else:
            classnames = self.dm.lab2cname.values()

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.train()

        if cfg.TRAINER.MVLPT.PREC == "fp32" or cfg.TRAINER.MVLPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, dm=self.dm)

        if cfg.TRAINER.MVLPT.PREC == "fp16":
            self.model.image_encoder.head[-1].weight.data = self.model.image_encoder.head[-1].weight.data.half()
            if self.model.image_encoder.head[-1].bias is not None:
                self.model.image_encoder.head[-1].bias.data = self.model.image_encoder.head[-1].bias.data.half()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            # if "prompt_learner" not in name and 'visual.conv1' not in name and 'visual.positional_embedding' not in name and 'visual.scale' not in name and 'visual.shift' not in name and 'class_embedding' not in name:
            #     param.requires_grad_(False)
            # else:
            #     # print(name, param.shape)
            #     param.requires_grad_(True)

            param.requires_grad_(False)
            if 'prompt_learner' in name or 'visual.conv1' in name or 'visual.positional_embedding' in name or 'visual.scale' in name \
                    or 'visual.shift' in name or 'class_embedding' in name or 'visual.ln_pre' in name:
                param.requires_grad_(True)

            if ('image_encoder' in name and 'scale' in name) or ('image_encoder' in name and 'shift' in name):
                param.requires_grad_(True)
                # print(name, param.shape)

            if "image_encoder" in name and 'head' in name:
                param.requires_grad_(True)
                # print(name)

            if 'image_encoder' in name and 'bias' in name:
                param.requires_grad_(True)

        print(
            f"Tunable Param: {sum([p.numel() for p in self.model.parameters() if p.requires_grad]) / 10 ** 6}M, Original CLIP {sum([p.numel() for p in self.model.parameters() if not p.requires_grad]) / 10 ** 6}M")

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)

        # NOTE: update other re-initialized layers
        # param_groups = [{'params': self.model.prompt_learner.parameters()},
        #                 {'params': self.model.image_encoder.visual.conv1.parameters()},
        #                 {'params': self.model.image_encoder.visual.positional_embedding},   # nn.Parameter
        #                 {'params': self.model.image_encoder.visual.scale},                  # nn.Parameter
        #                 {'params': self.model.image_encoder.visual.shift},                  # nn.Parameter
        #                 ]
        param_groups = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_groups.append({'params': param})
                print(name, param.shape)

        # param_groups: If provided, directly optimize param_groups and abandon model
        # https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/optim/optimizer.py
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM, param_groups=param_groups)

        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        # NOTE: register "model", not just prompt_learner
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MVLPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        self.count_batch = 0

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        self.multi_task = self.cfg.DATASET.MULTITASK
        self.multi_task_label_pertask = self.cfg.DATASET.MULTITASK_LABEL_PERTASK

        if self.cfg.DATASET.COOP:
            dm = MVLPTCOOPDataManager(self.cfg)
        elif self.cfg.DATASET.MULTITASK:
            dm = MVLPTMTDataManager(self.cfg)
            # self.test_dataloader_by_task = dm.self.test_dataloader_by_task
        else:
            dm = MVLPTDataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        # self.train_loader_x = dm.test_dataloader_by_task
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

        # self.train_loader_x_share = dm.train_loader_x_share

        # ######### EXTRACT FEATURES #########
        # print('CHECK DATASET LEN')
        # print('TRAIN', len(self.train_loader_x))
        # print('VAL', len(self.val_loader ))
        # print('TEST', len(self.test_loader))
        # if self.train_loader_u:
        #     print('TRAIN U ', len(self.train_loader_u))

    def forward_backward(self, batch):
        # n_tasks = torch.arange(len(self.cfg.DATASET.DATASET.split(',')))

        image, label, tasks_ = self.parse_batch_train(batch)

        # print('forward_backward image', image.shape)
        # print('forward_backward label', label.shape)

        # n_tasks = torch.arange(len(self.cfg.DATASET.DATASET.split(',')))

        # print('forward_backward', tasks_)
        # finetune = False
        # if tasks_ is not None:
        #     task_unique = torch.unique(tasks_)
        #     if len(task_unique) == 1:
        #         mask_tasks = n_tasks[n_tasks != task_unique]
        #     elif len(task_unique) > 1: # mix many tasks
        #         mask_tasks = None
        # print('tasks_', tasks_, task_unique, mask_tasks)

        # HACK: for multi-label classification, either works
        if len(label.shape) > 1 and label.shape[-1] > 1:
            label = label.float()
            label /= label.sum(dim=-1, keepdim=True)

        prec = self.cfg.TRAINER.MVLPT.PREC
        if prec == "amp":
            with autocast():
                try:
                    output = self.model(image, task=tasks_)
                except:
                    output = self.model.module(image, task=tasks_)
                # output, ext_x, input_x = self.model(image, task=tasks_)
                loss = F.cross_entropy(output, label)
            # print('forward_backward loss shape', loss.shape)
            self.optim.zero_grad()

            ### progressive
            # if self.model.module.prompt_learner.vpt_embeddings.grad is not None:
            #     print('forward_backward 11', self.model.module.prompt_learner.vpt_embeddings.grad.shape)
            #     print('forward_backward 11 check 0s', torch.all(self.model.module.prompt_learner.vpt_embeddings.grad))
            #
            # if self.model.module.prompt_learner.vpt_task_embeddings.grad is not None:
            #     print('forward_backward 12', self.model.module.prompt_learner.vpt_task_embeddings.grad.shape)
            #     print('forward_backward 12 check 0s', torch.all(self.model.module.prompt_learner.vpt_task_embeddings.grad))
            #
            # if self.model.module.prompt_learner.vpt_embeddings_deep.grad is not None:
            #     print('forward_backward 13', self.model.module.prompt_learner.vpt_embeddings_deep.grad.shape)
            #     print('forward_backward 13 check 0s', torch.all(self.model.module.prompt_learner.vpt_embeddings_deep.grad))
            #
            # if self.model.module.prompt_learner.vpt_task_embeddings_deep.grad is not None:
            #     print('forward_backward 14', self.model.module.prompt_learner.vpt_task_embeddings_deep.grad.shape)
            #     print('forward_backward 14 check 0s', torch.all(self.model.module.prompt_learner.vpt_task_embeddings_deep.grad))
            ### end progressive

            self.scaler.scale(loss).backward()
            ### progressive
            # if self.model.module.prompt_learner.vpt_embeddings.grad is not None:
            #     print('forward_backward 21', self.model.module.prompt_learner.vpt_embeddings.grad.shape)
            #     print('forward_backward 21 check 0s', torch.all(self.model.module.prompt_learner.vpt_embeddings.grad))
            #
            # if self.model.module.prompt_learner.vpt_task_embeddings.grad is not None:
            #     print('forward_backward 22', self.model.module.prompt_learner.vpt_task_embeddings.grad.shape)
            #     print('forward_backward 22 check 0s', torch.all(self.model.module.prompt_learner.vpt_task_embeddings.grad))
            #
            # if self.model.module.prompt_learner.vpt_embeddings_deep.grad is not None:
            #     print('forward_backward 23', self.model.module.prompt_learner.vpt_embeddings_deep.grad.shape)
            #     print('forward_backward 23 check 0s', torch.all(self.model.module.prompt_learner.vpt_embeddings_deep.grad))
            #
            # if self.model.module.prompt_learner.vpt_task_embeddings_deep.grad is not None:
            #     print('forward_backward 24', self.model.module.prompt_learner.vpt_task_embeddings_deep.grad.shape)
            #     print('forward_backward 24 check 0s', torch.all(self.model.module.prompt_learner.vpt_task_embeddings_deep.grad))
            ### end progressive

            self.scaler.step(self.optim)

            ### progressive
            # if self.model.module.prompt_learner.vpt_embeddings.grad is not None:
            #     print('forward_backward 31', self.model.module.prompt_learner.vpt_embeddings.grad.shape)
            #     print('forward_backward 31 check 0s', torch.all(self.model.module.prompt_learner.vpt_embeddings.grad))
            #
            # if self.model.module.prompt_learner.vpt_task_embeddings.grad is not None:
            #     print('forward_backward 32', self.model.module.prompt_learner.vpt_task_embeddings.grad.shape)
            #     print('forward_backward 32 check 0s', torch.all(self.model.module.prompt_learner.vpt_task_embeddings.grad))
            #
            # if self.model.module.prompt_learner.vpt_embeddings_deep.grad is not None:
            #     print('forward_backward 33', self.model.module.prompt_learner.vpt_embeddings_deep.grad.shape)
            #     print('forward_backward 33 check 0s', torch.all(self.model.module.prompt_learner.vpt_embeddings_deep.grad))
            #
            # if self.model.module.prompt_learner.vpt_task_embeddings_deep.grad is not None:
            #     print('forward_backward 34', self.model.module.prompt_learner.vpt_task_embeddings_deep.grad.shape)
            #     print('forward_backward 34 check 0s', torch.all(self.model.module.prompt_learner.vpt_task_embeddings_deep.grad))
            ### end progressive

            self.scaler.update()
        else:
            output = self.model(image, task=tasks_)
            # try:
            #     output = self.model(image, task=tasks_)
            # except:
            #     output = self.model.module(image, task=tasks_)

            # output, ext_x, input_x = self.model(image, task=tasks_)

            # print(label.shape, output.shape, label.dtype, output.dtype, tasks_, label.sum(dim=-1))

            # if self.model.module.prompt_learner.vpt_embeddings_mprompts.grad is not None:
            #     old_vpt_embeddings_grad = self.model.module.prompt_learner.vpt_embeddings_mprompts.grad.clone()
            #     old_vpt_embeddings = self.model.module.prompt_learner.vpt_embeddings_mprompts.clone()
            #     old_vpt_embeddings_deep_grad = self.model.module.prompt_learner.vpt_embeddings_deep_mprompts.grad.clone()
            #     old_vpt_embeddings_deep = self.model.module.prompt_learner.vpt_embeddings_deep_mprompts.clone()
            #     old_vpt_alpha = self.model.module.prompt_learner.alphas.clone()
            #     old_vpt_alpha_grad = self.model.module.prompt_learner.alphas.grad.clone()
            #     old_vpt_alpha_deep = self.model.module.prompt_learner.alphas_deep.clone()
            #     old_vpt_alpha_deep_grad = self.model.module.prompt_learner.alphas_deep.grad.clone()
            #
            # print('VALUE 1 before backward alphas', self.model.module.prompt_learner.alphas.grad)
            # print('VALUE 1 before backward alphas_deep', self.model.module.prompt_learner.alphas_deep.grad)

            # print('self.model.module.vpt_embeddings.sum 0', self.model.module.prompt_learner.vpt_embeddings.data.sum())
            # print('self.model.module.vpt_embeddings_deep.sum 0', self.model.module.prompt_learner.vpt_embeddings_deep.data.sum())
            # print('self.model.module.image_encoder.conv1.sum 0', self.model.module.image_encoder.visual.conv1.weight.sum())
            # print('self.model.module.image_encoder.positional_embedding.sum 0', self.model.module.image_encoder.visual.positional_embedding.data.sum())
            # print('self.model.module.image_encoder.scale.sum 0', self.model.module.image_encoder.visual.scale.data.sum())
            # print('self.model.module.image_encoder.shift.sum 0', self.model.module.image_encoder.visual.shift.data.sum())

            loss = F.cross_entropy(output, label)
            # self.mask_grad()
            self.model_backward_and_update(loss)

            # print('self.model.module.vpt_embeddings.sum 0', self.model.module.prompt_learner.vpt_embeddings.data.sum())
            # print('self.model.module.vpt_embeddings_deep.sum 0', self.model.module.prompt_learner.vpt_embeddings_deep.data.sum())
            # print('self.model.module.image_encoder.conv1.sum 1', self.model.module.image_encoder.visual.conv1.weight.sum())
            # print('self.model.module.image_encoder.positional_embedding.sum 0', self.model.module.image_encoder.visual.positional_embedding.data.sum())
            # print('self.model.module.image_encoder.scale.sum 1', self.model.module.image_encoder.visual.scale.data.sum())
            # print('self.model.module.image_encoder.shift.sum 1', self.model.module.image_encoder.visual.shift.data.sum())

            # self.model_backward_and_update(loss, tasks=mask_tasks, is_finetune=finetune)

            # if self.model.module.prompt_learner.vpt_embeddings_mprompts.grad is not None:

            # print('GRAD SHAPE self.vpt_embeddings', self.model.module.prompt_learner.vpt_embeddings.grad.shape)
            # print('GRAD SHAPE self.vpt_embeddings_deep', self.model.module.prompt_learner.vpt_embeddings_deep.grad.shape)
            # print('GRAD SHAPE self.vpt_task_embeddings', self.model.module.prompt_learner.vpt_task_embeddings.grad.shape)
            # print('GRAD SHAPE self.vpt_task_embeddings_deep', self.model.module.prompt_learner.vpt_task_embeddings_deep.grad.shape)
            # print('GRAD SHAPE self.vpt_task_alpha', self.model.module.prompt_learner.vpt_task_alpha.grad)
            # print('GRAD SHAPE self.alphas', self.model.module.prompt_learner.alphas.grad)
            # print('GRAD SHAPE self.alphas_deep', self.model.module.prompt_learner.alphas_deep.grad)

            # try:
            #     print('GRAD self.vpt_embeddings same???',
            #           torch.equal(old_vpt_embeddings_grad, self.model.module.prompt_learner.vpt_embeddings_mprompts.grad))
            #     print('VALUE self.vpt_embeddings same???',
            #           torch.equal(old_vpt_embeddings, self.model.module.prompt_learner.vpt_embeddings_mprompts))
            #     print('GRAD self.vpt_embeddings_deep same???',
            #           torch.equal(old_vpt_embeddings_deep_grad, self.model.module.prompt_learner.vpt_embeddings_deep_mprompts.grad))
            #     print('VALUE self.vpt_embeddings_deep same???',
            #           torch.equal(old_vpt_embeddings_deep, self.model.module.prompt_learner.vpt_embeddings_deep_mprompts))
            #     print('GRAD self.alphas same???',
            #           torch.equal(old_vpt_alpha_grad, self.model.module.prompt_learner.alphas.grad))
            #     print('VALUE self.alphas same???',
            #           torch.equal(old_vpt_alpha, self.model.module.prompt_learner.alphas))
            #     print('GRAD self.alphas_deep same???',
            #           torch.equal(old_vpt_alpha_deep_grad, self.model.module.prompt_learner.alphas_deep.grad))
            #     print('VALUE self.alphas_deep same???',
            #           torch.equal(old_vpt_alpha_deep, self.model.module.prompt_learner.alphas_deep))
            # except:
            #     pass

            # old_vpt_embeddings_grad = self.model.module.prompt_learner.vpt_embeddings.grad
            # old_vpt_embeddings = self.model.module.prompt_learner.vpt_embeddings
            # old_vpt_embeddings_deep_grad = self.model.module.prompt_learner.vpt_embeddings_deep.grad
            # old_vpt_embeddings_deep = self.model.module.prompt_learner.vpt_embeddings_deep

            # old_vpt_alpha = self.model.module.prompt_learner.alphas
            # old_vpt_alpha_grad = self.model.module.prompt_learner.alphas.grad
            # old_vpt_alpha_deep = self.model.module.prompt_learner.alphas_deep
            # old_vpt_alpha_deep_grad = self.model.module.prompt_learner.alphas_deep.grad
            #
            # print('VALUE 1 after backward alphas', self.model.module.prompt_learner.alphas)
            # print('VALUE 1 after backward alphas_deep', self.model.module.prompt_learner.alphas_deep)

        # HACK: During training, we hack the eval of multi-label by selecting only one class
        if len(label.shape) > 1 and label.shape[-1] > 1:
            label = torch.argmax(label, dim=1)

        # result = self.dm._metric(label.squeeze().cpu().detach().numpy(), output.cpu().detach().numpy())

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            # "acc": result,
        }
        if tasks_ is not None:
            loss_summary.update({"num_tasks": len(set(tasks_.tolist()))})

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        # print('END 1 ITER')
        self.count_batch += 1
        return loss_summary

    def parse_batch_train(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        # print(label.shape, 'label', input.shape, 'input')
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        # input = batch["img"]
        # label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, tasks

    def parse_batch_test(self, batch):
        if self.cfg.DATASET.COOP:
            inp_key, lab_key, task_key = 'img', 'label', 'domain'
        else:
            inp_key, lab_key, task_key = 0, 1, 3
        input = batch[inp_key]
        label = batch[lab_key]
        tasks = None
        if self.multi_task:
            tasks = batch[task_key]
        # input = batch["img"]
        # label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, tasks

    def model_inference(self, input, task=None):
        try:
            return self.model(input, task=task)
        except:
            return self.model.module(input, task=task)

    @torch.no_grad()
    def test(self, split=None):
        from tqdm import tqdm
        import copy
        import numpy as np
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            print('val_loader might be None')
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        self.evaluator_task = dict()

        self.elevator_evaluator = {'y_pred': [], 'y_true': []}

        if self.multi_task:
            if self.cfg.DATASET.COOP:
                self.evaluator_task = {task: copy.deepcopy(self.evaluator) for task in self.dm._task_names}
            else:
                self.evaluator_task = {task: copy.deepcopy(self.elevator_evaluator) for task in self.dm._task_names}
        #
        # input_x_layers = []
        # extract_x_layers = []
        # labels = []
        # count=0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label, tasks_ = self.parse_batch_test(batch)
            output = self.model_inference(input, task=tasks_)
            # output, ext_x, input_x = self.model_inference(input, task=tasks_)
            # HACK: make everything one-hot vector label!
            if self.cfg.DATASET.COOP:
                self.evaluator.process(output, label)

            else:
                self.elevator_evaluator['y_pred'].append(output.cpu().detach().numpy())
                self.elevator_evaluator['y_true'].append(label.cpu().detach().numpy())

            # print('output', output.shape, output)

            if tasks_ is not None:
                for out, lab, task in zip(output, label, tasks_):
                    task = self.dm._id2task[task.item()]

                    if self.cfg.DATASET.COOP:
                        class_start, class_end = self.dm._task_class_idx[task]
                        # Evaluate on the task-specific class
                        out = out[class_start:class_end]
                        lab -= class_start
                        self.evaluator_task[task].process(out.unsqueeze(0), lab.unsqueeze(0))
                    else:
                        self.evaluator_task[task]['y_pred'].append([out.cpu().detach().numpy()])
                        self.evaluator_task[task]['y_true'].append([lab.cpu().detach().numpy()])

            ######## EXTRACT FEATURES ########
            # print('test loop ext_x', ext_x.shape)
            # print('test loop ext_vpt_emb_deep', ext_vpt_emb_deep.shape)
            # print('test loop ext_visual', ext_visual.shape)
            # print('test loop loop output', output.shape)
            # print('test loop loop label', label.shape)

            # if count <= 10:
            #     input_x_layers.append(input_x)
            #     extract_x_layers.append(ext_x)
            #     labels.append(label)

            # print(output.shape)
            # count += 1
            # if count == 10:
            #     break

            # if (not (count % 20) and count != 0) or len(label) < 64:

            # input_x_layers_save = torch.cat([input_x], 0) #([1, 11, 78, 213, 768])
            # extract_x_layers_save = torch.cat([ext_x], 0) #([1, 11, 78, 213, 768])
            # labels_save = torch.cat([label], 0)

            # input_x_layers_save = torch.cat(input_x_layers, 0)
            # extract_x_layers_save = torch.cat(extract_x_layers, 0)
            # labels_save = torch.cat(labels, 0)

            # print('test loop ext_x', input_x_layers_save.shape)
            # print('test loop extract_x_layers', extract_x_layers_save.shape)
            # print('test loop labels', labels_save.shape)
            # # break
            # input_x_layers_save = input_x_layers_save.reshape(input_x_layers_save.shape[1],
            #                                                   -1, input_x_layers_save.shape[-2],
            #                                                   input_x_layers_save.shape[-1])
            #
            # extract_x_layers_save = extract_x_layers_save.reshape(extract_x_layers_save.shape[1],
            #                                                   -1, extract_x_layers_save.shape[-2],
            #                                                   extract_x_layers_save.shape[-1])
            #
            # print('test loop ext_x', input_x_layers_save.shape)
            # print('test loop extract_x_layers', extract_x_layers_save.shape)
            # print('test loop labels', labels_save.shape)
            #
            # input_class = input_x_layers_save[:, :, :1, :]
            # input_prompt = input_x_layers_save[:, :, 1:17, :]
            # input_visual = input_x_layers_save[:, :, 17:, :]
            # output_class = extract_x_layers_save[:, :, :1, :]
            # output_prompt = extract_x_layers_save[:, :, 1:17, :]
            # output_visual = extract_x_layers_save[:, :, 17:, :]
            #
            # input_prompt = input_prompt.mean(2)
            # output_prompt = output_prompt.mean(2)
            # input_visual = input_visual.mean(2)
            # output_visual = output_visual.mean(2)
            # input_class = input_class.squeeze(2)
            # output_class = output_class.squeeze(2)

            # input_x_layers = torch.cat(input_x_layers,0)
            # input_x_layers = input_x_layers.view(input_x_layers.shape[1], -1,
            #                                      input_x_layers.shape[3],
            #                                      input_x_layers.shape[4])
            # extract_x_layers = torch.cat(extract_x_layers, 0)
            # extract_x_layers = extract_x_layers.view(extract_x_layers.shape[1], -1,
            #                                          extract_x_layers.shape[3],
            #                                          extract_x_layers.shape[4])
            # labels = torch.cat(labels, 0)

            # print('test loop ext_x', input_x_layers.shape)
            # print('test loop extract_x_layers', extract_x_layers.shape)
            # print('test loop labels', labels.shape)

            # save_path = '/project/hnguyen2/stly/code/prompting/logs/extract_features'
            # # save_path = '/project/hnguyen2/stly/code/prompting/logs/extract_features_finetune'
            # extracted_features_path = os.path.join(save_path, self.cfg.DATASET.DATASET, 'extracted_features%s.npz' %count)
            # np.savez(extracted_features_path,
            #          input_prompt=input_prompt.detach().cpu().numpy(),
            #          output_prompt=output_prompt.detach().cpu().numpy(),
            #          input_visual=input_visual.detach().cpu().numpy(),
            #          output_visual=output_visual.detach().cpu().numpy(),
            #          input_class=input_class.detach().cpu().numpy(),
            #          output_class=output_class.detach().cpu().numpy(),
            #          labels_save=labels_save.detach().cpu().numpy(),
            #          )
            # input_x_layers=input_x_layers_save.detach().cpu().numpy(),
            # extract_x_layers=extract_x_layers_save.detach().cpu().numpy(),
            # labels=labels_save.detach().cpu().numpy())

            # input_x_layers = []
            # extract_x_layers = []
            # labels = []
            # del input_x_layers_save, extract_x_layers_save, labels_save
            # torch.cuda.empty_cache()

            # else:
            #     # ext_x, input_x
            #     input_x_layers.append(input_x)
            #     extract_x_layers.append(ext_x)
            #     labels.append(label)

            # count += 1

        results_overall = {}
        for task in self.evaluator_task:
            print(f"evaluate on the *{task}* !")
            if self.cfg.DATASET.COOP:
                results = self.evaluator_task[task].evaluate()
                results_overall[task] = results['accuracy']
            else:
                y_true = np.concatenate(self.evaluator_task[task]['y_true'], axis=0)
                y_pred = np.concatenate(self.evaluator_task[task]['y_pred'], axis=0)
                class_start, class_end = self.dm._task_class_idx[task]
                y_true = y_true[:, class_start:class_end]
                y_pred = y_pred[:, class_start:class_end]

                if self.dm._metric_name[task] == 'accuracy':
                    y_true = np.argmax(y_true, axis=-1)
                metric_result = self.dm._metric[task](y_true, y_pred)
                results = {self.dm._metric_name[task]: metric_result}
                results_overall[task] = metric_result
            print('results', results)
            for k, v in results.items():
                tag = f"{split}/{task}/{k}"
                self.write_scalar(tag, v, self.epoch)

        print(f"Overall evaluation !")
        if self.multi_task:
            multi_task_evalkey = self.cfg.DATASET.MULTITASK_EVALKEY
            if multi_task_evalkey == 'average':
                results = {'average': sum([v for k, v in results_overall.items()]) / len(results_overall)}
            else:
                assert multi_task_evalkey in results_overall
                results = {multi_task_evalkey: results_overall[multi_task_evalkey]}
                print(f"select {multi_task_evalkey} as the evaluation key")
        else:
            if not self.cfg.DATASET.COOP:
                y_true = np.concatenate(self.elevator_evaluator['y_true'], axis=0)
                y_pred = np.concatenate(self.elevator_evaluator['y_pred'], axis=0)
                results = {self.dm._metric_name: self.dm._metric(y_true, y_pred)}
            else:
                results = self.evaluator.evaluate()

        # compute AUC
        auc, sen, spec = self.get_auc(results)
        print(f"* AUC: {auc:.2f}")
        print(f"* Sensitivity: {sen:.2f}")
        print(f"* Specificity: {spec:.2f}")
        del results["_y_true"]
        del results["_y_pred"]
        results["AUC"] = auc
        results["Sensitivity"] = sen
        results["Specificity"] = spec

        print('results', results)
        for k, v in results.items():
            tag = f"/{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return results["accuracy"], results["AUC"]

    def get_auc(self, results):

        _y_true = results["_y_true"]
        _y_pred = results["_y_pred"]
        # print('_y_true', _y_true)
        # print('_y_pred', _y_pred)

        true = np.array(_y_true)
        pred = np.array(_y_pred)

        if true.max() == 1:  # binary-class
            auc = roc_auc_score(true, pred)
            tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            return auc, sensitivity, specificity

        else:  # multi-classes
            auc = 0
            sensitivity = 0
            specificity = 0
            for i in range(true.max() + 1):
                y_true_binary = np.where(true == i, 1, 0)
                y_score_binary = np.where(pred == i, 1, 0)
                temp_auc = roc_auc_score(y_true_binary, y_score_binary)
                auc += temp_auc
                tn, fp, fn, tp = confusion_matrix(y_true_binary, y_score_binary).ravel()
                sen = tp / (tp + fn)
                spec = tn / (tn + fp)
                print(f"Class {i}: AUC {temp_auc:.2f}, sensitivity {sen:.2f}, specificity {spec:.2f}")
                sensitivity += sen
                specificity += spec
            ret = auc / (true.max() + 1)
            ret_sen = sensitivity / (true.max() + 1)
            ret_spec = specificity / (true.max() + 1)
            return ret, ret_sen, ret_spec

    def load_model(self, directory, epoch=None, name_model=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        print('load_model', names)

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        if name_model is not None:
            model_file = name_model

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            # issue 1 bug fix for UPT key name mismatch
            state_dict = {k.replace("upt_proj", "mvlpt_proj"): v for k, v in state_dict.items()}
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print('START load model state_dict')
            new_state_dict = state_dict.copy()
            for k, v in state_dict.items():
                if k == 'vpt_embeddings':
                    del new_state_dict[k]
                    new_state_dict['prompt_learner.vpt_embeddings'] = state_dict[k]
                elif k == 'vpt_embeddings_deep':
                    del new_state_dict[k]
                    new_state_dict['prompt_learner.vpt_embeddings_deep'] = state_dict[k]
            print('END load model state_dict')

            # new_state_dict = state_dict.copy()
            # for key, params in list(new_state_dict.items()):
            #     if '_task_' in key:
            #         del new_state_dict[key]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            # msg = self._models[name].load_state_dict(state_dict, strict=False)
            msg = self._models[name].load_state_dict(new_state_dict, strict=False)
            print('load_model msg', msg)

        ### progressive
        # if self.cfg.IS_PRETRAIN and self.cfg.IS_PROGRESSIVE:
        # print('PRETRAINING: copy params for initialization!!!')
        # layer = self._models['prompt_learner']
        # if layer.vpt_task_embeddings is not None:
        #     n_tasks = layer.vpt_task_embeddings.shape[0]
        #     # print('load model 1 n_tasks', n_tasks)
        #     for i in range(n_tasks):
        #         for param_b, param_m in zip(layer.vpt_embeddings, layer.vpt_task_embeddings[i]):
        #             # print('load model param_b shape', param_b.shape)
        #             # print('load model param_m shape', param_m.shape)
        #             param_m.data.copy_(param_b.data)  # initialize
        #
        # if layer.vpt_task_embeddings_deep is not None:
        #     n_tasks = layer.vpt_task_embeddings_deep.shape[0]
        #     # print('load model 2 n_tasks', n_tasks)
        #     for i in range(n_tasks):
        #         for param_b, param_m in zip(layer.vpt_embeddings_deep, layer.vpt_task_embeddings_deep[i]):
        #             # print('load model param_b shape', param_b.shape)
        #             # print('load model param_m shape', param_m.shape)
        #             param_m.data.copy_(param_b.data)  # initialize
        # print('PRETRAINING: Done copy params for initialization!!!')

        # elif (not self.cfg.IS_PRETRAIN) and self.cfg.IS_PROGRESSIVE:
        # print('FINETUNING: copy params for initialization!!!')
        # layer = self._models['prompt_learner']
        # if layer.vpt_task_embeddings is not None:
        #     model_path = osp.join(directory, 'prompt_learner', model_file)
        #     assert osp.isfile(model_path), 'Wrong ckpt path!!!'
        #     # model_path = '/project/hnguyen2/stly/code/prompting/logs/pretrain/trial1/ImageNet,Caltech101,Food101,StanfordCars,OxfordPets,OxfordFlowers,FGVCAircraft,SUN397,DescribableTextures,EuroSAT,UCF101/VPT/vit_b16_20shots/nctx16_csc_ctp/seed3/prompt_learner/model-best.pth.tar'
        #     checkpoint = load_checkpoint(model_path)
        #     state_dict = checkpoint["state_dict"]
        #
        #     # get state_dict of task-specific and take the mean
        #     pretrained_vpt_task_embeddings = state_dict['vpt_task_embeddings'].mean(0).unsqueeze(0)
        #     pretrained_vpt_task_embeddings_deep = state_dict['vpt_task_embeddings_deep'].mean(0).unsqueeze(0)
        #
        #     n_tasks = layer.vpt_task_embeddings.shape[0]
        #     pretrained_vpt_task_embeddings = torch.cat(n_tasks*[pretrained_vpt_task_embeddings], 0)
        #     pretrained_vpt_task_embeddings_deep = torch.cat(n_tasks * [pretrained_vpt_task_embeddings_deep], 0)
        #
        #     assert layer.vpt_task_embeddings.shape == pretrained_vpt_task_embeddings.shape
        #     assert layer.vpt_task_embeddings_deep.shape == pretrained_vpt_task_embeddings_deep.shape
        #
        #     layer.vpt_task_embeddings.data.copy_(pretrained_vpt_task_embeddings)
        #     layer.vpt_task_embeddings_deep.data.copy_(pretrained_vpt_task_embeddings_deep)
        # print('FINETUNING: Done copy params for initialization!!!')
        ### end progressive

