from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_
import math

def init_ssf_scale_shift(dim, dtype=torch.half):
    scale = nn.Parameter(torch.ones(dim, dtype=dtype))
    shift = nn.Parameter(torch.zeros(dim, dtype=dtype))
    # scale = nn.Parameter(torch.ones(dim))
    # shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)
    # scale.data.half()
    # shift.data.half()
    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    # .type(x.type()
    # orig_type = x.dtype
    # scale = scale.type(orig_type)
    # shift = shift.type(orig_type)
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    # elif x.shape[1] == scale.shape[0]:
    #     return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    elif x.shape[1] == scale.shape[0]:
        if x.ndim == 5:
            return x * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
        elif x.ndim == 4:
            return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')


def ssf_attn(x, scale, shift):
    return scale[:, None] * x + shift[:, None]

class Adapter(nn.Module):
    def __init__(self,
                 d_model=768,
                 bottleneck=64, # paper: 64
                 dropout=0.1,
                 init_option="lora",
                 adapter_scalar="1.0",  # paper: 0.1
                 adapter_layernorm_option="none",
                 d_out=None):  # paper: none
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1, dtype=torch.half))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        if d_out is not None:
            self.d_out = d_out
        else:
            self.d_out = self.n_embd

        self.up_proj = nn.Linear(self.down_size, self.d_out)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP_ss(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(d_model, dtype=torch.half))
        self.shift = nn.Parameter(torch.zeros(d_model, dtype=torch.half))

        nn.init.normal_(self.scale, mean=1, std=.02)
        nn.init.normal_(self.shift, std=.02)

    def forward(self, x):
        if x.shape[-1] == self.scale.shape[0]:
            return x * self.scale + self.shift
        elif x.shape[1] == self.scale.shape[0]:
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            raise ValueError('the input tensor shape does not match the shape of the scale factor.')


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 ssf_scale_1=None, ssf_shift_1=None,
                                 ssf_scale_2=None, ssf_shift_2=None,
                                 lora1=None, lora2=None,
                                 mode: str = 'lorass_qkvattn',
                                 average_attn_weights=True
                                 ):
    tgt_len, bsz, embed_dim = query.size()
    # print('multi_head_attention_forward query', query.size())   # torch.Size([213, 624, 768])

    head_dim = embed_dim // num_heads   # 64
    scaling = float(head_dim) ** -0.5

    # q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    # scale bias 1
    # if 'qkv_only' in mode:
    #     q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    # elif 'ss' in mode:
    qkv = F.linear(query, in_proj_weight, in_proj_bias)
    # print('multi_head_attention_forward qkv', qkv.shape)  # torch.Size([213, 624, 2304])

    qkv = ssf_ada(qkv, ssf_scale_1, ssf_shift_1)
    adapt_query = lora1(query, add_residual=False)
    qkv = qkv + adapt_query
    q, k, v = qkv.chunk(3, dim=-1)


    q = q * scaling

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1) # 213, 624*12, 64
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    # print('multi_head_attention_forward 0', q.shape, k.shape) #torch.Size([7488, 213, 64]) torch.Size([7488, 213, 64])


    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        attn_mask = pad(attn_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # print('multi_head_attention_forward 1', attn_output_weights.shape)  # torch.Size([7488, 213, 213])
    if attn_mask is not None:
        attn_output_weights += attn_mask

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    # print('multi_head_attention_forward 2', attn_output_weights.shape)  # torch.Size([7488, 213, 213])
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    # attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=False)
    # print('multi_head_attention_forward 3', attn_output_weights.shape)  # torch.Size([7488, 213, 213])

    attn_output = torch.bmm(attn_output_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    # # scale bias 2
    # attn_output = ssf_ada(attn_output, ssf_scale_2, ssf_shift_2)


    adapt_attn = lora2(attn_output, add_residual=False)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = ssf_ada(attn_output, ssf_scale_2, ssf_shift_2)
    attn_output = attn_output + adapt_attn

    # return attn_output, None

    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
    if average_attn_weights:
        attn_output_weights = attn_output_weights.mean(dim=1)

    return attn_output, attn_output_weights#.view(bsz, num_heads, tgt_len, tgt_len)

class NonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)

class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, mode: str = 'lorass_qkvattn'):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim, dtype=torch.half))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, dtype=torch.half))

        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)
        # self.register_parameter('in_proj_bias', None)


        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=True).to(torch.half)
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

        self.mode = mode
        print('MultiheadAttention mode:', mode)

        self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(embed_dim * 3)
        self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(embed_dim)

        # Lora
        self.lora1 = Adapter(d_model=embed_dim,
                            bottleneck=64,
                            adapter_scalar='learnable_scalar',
                            d_out=embed_dim*3)  # paper: 0.1 or learnable_scalar

        self.lora2 = Adapter(d_model=embed_dim,
                            bottleneck=64,
                            adapter_scalar='learnable_scalar')  # paper: 0.1 or learnable_scalar


    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        return multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask,
            ssf_scale_1=self.ssf_scale_1, ssf_shift_1=self.ssf_shift_1,
            ssf_scale_2=self.ssf_scale_2, ssf_shift_2=self.ssf_shift_2,
            lora1=self.lora1, lora2=self.lora2,
            mode=self.mode)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mode: str or None, attn_mask: torch.Tensor = None):
        """

        :param d_model:
        :param n_head:
        :param attn_mask:
        :param mode: lora (ori lora), lorass_attn, lorass_qkv, lorass_qkvattn
        """
        super().__init__()
        self.mode = mode
        print('ResidualAttentionBlock mode: ', mode)
        if self.mode is None or self.mode == "qkvsb":
            self.attn = nn.MultiheadAttention(d_model, n_head)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]))

            if self.mode == "qkvsb":
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(d_model)
                self.ssf_scale_3, self.ssf_shift_3 = init_ssf_scale_shift(d_model)
                self.ssf_scale_4, self.ssf_shift_4 = init_ssf_scale_shift(d_model)
                self.ssf_scale_5, self.ssf_shift_5 = init_ssf_scale_shift(d_model)
        else:
            if self.mode == 'lora':   # lora
                self.attn = nn.MultiheadAttention(d_model, n_head)
                self.mlp = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model))
                ]))
            elif "attn" in self.mode: # lorass_attn, lorass_qkvattn
                self.attn = MultiheadAttention(d_model, n_head)         # attn
                self.mlp = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("fc_ss1", MLP_ss(d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                    ("fc_ss2", MLP_ss(d_model)),
                ]))
            elif self.mode == 'lorass_qkvattn':
                self.attn = MultiheadAttention(d_model, n_head)         # attn
                self.mlp = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("fc_ss1", MLP_ss(d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                    ("fc_ss2", MLP_ss(d_model)),
                ]))

            else:   # lorass_qkv, ss_qkvsb
                self.attn = nn.MultiheadAttention(d_model, n_head)
                self.mlp = nn.Sequential(OrderedDict([
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("fc_ss1", MLP_ss(d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                    ("fc_ss2", MLP_ss(d_model)),
                ]))


            if 'ss' in self.mode:
                self.ssf_scale_1, self.ssf_shift_1 = init_ssf_scale_shift(d_model)
                self.ssf_scale_2, self.ssf_shift_2 = init_ssf_scale_shift(d_model)

            if 'qkv' in self.mode:
                self.ssf_scale_3, self.ssf_shift_3 = init_ssf_scale_shift(d_model)
                self.ssf_scale_4, self.ssf_shift_4 = init_ssf_scale_shift(d_model)
                self.ssf_scale_5, self.ssf_shift_5 = init_ssf_scale_shift(d_model)


            self.lora = Adapter(d_model=d_model,
                                bottleneck=64,
                                adapter_scalar='learnable_scalar')  # paper: 0.1 or learnable_scalar

        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        if self.mode is None or 'qkv' not in self.mode:
            return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)#[0]

        elif 'qkv' in self.mode or self.mode == 'lorass_qkvattn':
            q = ssf_ada(x, self.ssf_scale_3, self.ssf_shift_3)
            k = ssf_ada(x, self.ssf_scale_4, self.ssf_shift_4)
            v = ssf_ada(x, self.ssf_scale_5, self.ssf_shift_5)
            # return self.attn(q, k, v, need_weights=False, attn_mask=self.attn_mask)#[0]
            return self.attn(q, k, v, attn_mask=self.attn_mask)#[0]

    def forward(self, x: torch.Tensor, return_attn_weight=False, return_attention=False):
        if self.mode is None:
            # if return_attention:
            #     return self.attention(self.ln_1(x))[1]

            attention_res = self.attention(self.ln_1(x))
            attn_weights, attn = attention_res[0], attention_res[1]

            # attn_weights = self.attention(self.ln_1(x))
            x = x + attn_weights

            x = x + self.mlp(self.ln_2(x))

        elif self.mode == 'lora':
            # if return_attention:
            #     return self.attention(self.ln_1(x))[1]

            attention_res = self.attention(self.ln_1(x))
            attn_weights, attn = attention_res[0], attention_res[1]

            # attn_weights = self.attention(self.ln_1(x))

            x = x + attn_weights
            adapt_x = self.lora(x, add_residual=False)
            residual = x

            x = self.mlp(self.ln_2(x))
            x = x + adapt_x
            x = residual + x

        elif self.mode == 'lorass_qkv' or self.mode == 'lorass_qkvattn':
            # if return_attention:
            #     return self.attention(ssf_ada(self.ln_1(x), self.ssf_scale_1, self.ssf_shift_1))[1]

            attention_res = self.attention(ssf_ada(self.ln_1(x), self.ssf_scale_1, self.ssf_shift_1))
            attn_weights, attn = attention_res[0], attention_res[1]

            # attn_weights = self.attention(ssf_ada(self.ln_1(x), self.ssf_scale_1, self.ssf_shift_1))

            x = x + attn_weights
            adapt_x = self.lora(x, add_residual=False)
            residual = x

            x = self.mlp(ssf_ada(self.ln_2(x), self.ssf_scale_2, self.ssf_shift_2))
            x = x + adapt_x
            x = residual + x

        elif self.mode == 'ss_qkvsb':

            attention_res = self.attention(ssf_ada(self.ln_1(x), self.ssf_scale_1, self.ssf_shift_1))
            attn_weights, attn = attention_res[0], attention_res[1]

            x = x + attn_weights

            x = x + self.mlp(ssf_ada(self.ln_2(x), self.ssf_scale_2, self.ssf_shift_2))

        elif self.mode == 'qkvsb':

            attention_res = self.attention(ssf_ada(self.ln_1(x), self.ssf_scale_1, self.ssf_shift_1))
            attn_weights, attn = attention_res[0], attention_res[1]

            x = x + attn_weights

            x = x + self.mlp(self.ln_2(x))

        # print('return_attn_weight', return_attn_weight)
        # print('return_attention', return_attention)
        if return_attn_weight and not return_attention:
            # print('return return_attn_weight')
            return x, attn_weights
        elif return_attention and not return_attn_weight:
            # print('return return_attention')
            return x, attn
        elif return_attn_weight and return_attention:
            # print('return both')
            # print(attn_weights.shape)
            # print(attn.shape)
            return x, attn_weights, attn
        else:
            return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mode: str or None, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, mode, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, ssmode: str):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, ssmode)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

    # def get_last_selfattention(self, x):
    #     x = self.conv1(x)  # shape = [*, width, grid, grid]
    #     x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    #     x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    #     x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
    #     x = x + self.positional_embedding.to(x.dtype)
    #     x = self.ln_pre(x)
    #
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #
    #     for i, blk in enumerate(self.transformer.resblocks):
    #         if i < len(self.blocks) - 1:
    #             x = blk(x)
    #         else:
    #             # return attention of the last block
    #             return blk(x, return_attention=True)


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 mode: str or None,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                ssmode=mode
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            mode=None,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, is_finetune=True, ssmode=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim, ssmode,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    if is_finetune:
        print(f"Finetune!!! Loading CLIP state_dict!!!")
        msg = model.load_state_dict(state_dict, strict=False)
        print('build_model msg: ', msg)
    else:
        print(f"Scratch!!! NOT Loading CLIP state_dict!!!")
    return model.eval()
