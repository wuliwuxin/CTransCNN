# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import DROPOUT_LAYERS
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule

from ..builder import ATTENTION
from .helpers import to_2tuple


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 init_cfg=None):

        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        super(WindowMSA, self).init_weights()

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor, Optional): mask with shape of (num_windows, Wh*Ww,
                Wh*Ww), value should be between (-inf, 0].
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


@ATTENTION.register_module()
class ShiftWindowMSA(BaseModule):
    """Shift Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Defaults to True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults to None.
        attn_drop (float, optional): Dropout ratio of attention weight.
            Defaults to 0.0.
        proj_drop (float, optional): Dropout ratio of output. Defaults to 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults to dict(type='DropPath', drop_prob=0.).
        pad_small_map (bool): If True, pad the small feature map to the window
            size, which is common used in detection and segmentation. If False,
            avoid shifting window and shrink the window size to the size of
            feature map, which is common used in classification.
            Defaults to False.
        init_cfg (dict, optional): The extra config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 pad_small_map=False,
                 input_resolution=None,
                 auto_pad=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        if input_resolution is not None or auto_pad is not None:
            warnings.warn(
                'The ShiftWindowMSA in new version has supported auto padding '
                'and dynamic input shape in all condition. And the argument '
                '`auto_pad` and `input_resolution` have been deprecated.',
                DeprecationWarning)

        self.shift_size = shift_size
        self.window_size = window_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.drop = build_dropout(dropout_layer)
        self.pad_small_map = pad_small_map

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, f"The query length {L} doesn't match the input " \
                           f'shape ({H}, {W}).'
        query = query.view(B, H, W, C)

        window_size = self.window_size
        shift_size = self.shift_size

        if min(H, W) == window_size:
            # If not pad small feature map, avoid shifting when the window size
            # is equal to the size of feature map. It's to align with the
            # behavior of the original implementation.
            shift_size = shift_size if self.pad_small_map else 0
        elif min(H, W) < window_size:
            # In the original implementation, the window size will be shrunk
            # to the size of feature map. The behavior is different with
            # swin-transformer for downstream tasks. To support dynamic input
            # shape, we don't allow this feature.
            assert self.pad_small_map, \
                f'The input shape ({H}, {W}) is smaller than the window ' \
                f'size ({window_size}). Please set `pad_small_map=True`, or ' \
                'decrease the `window_size`.'

        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))

        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if shift_size > 0:
            query = torch.roll(
                query, shifts=(-shift_size, -shift_size), dims=(1, 2))

        attn_mask = self.get_attn_mask((H_pad, W_pad),
                                       window_size=window_size,
                                       shift_size=shift_size,
                                       device=query.device)

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(query, window_size)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, window_size ** 2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, window_size, window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad,
                                        window_size)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if H != H_pad or W != W_pad:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)

        return x

    @staticmethod
    def window_reverse(windows, H, W, window_size):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows

    @staticmethod
    def get_attn_mask(hw_shape, window_size, shift_size, device=None):
        if shift_size > 0:
            img_mask = torch.zeros(1, *hw_shape, 1, device=device)
            h_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size,
                                                      -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = ShiftWindowMSA.window_partition(
                img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        return attn_mask


class MultiheadAttention(BaseModule):
    """Multi-head Attention Module.

    This module implements multi-head attention that supports different input
    dims and embed dims. And it also supports a shortcut from ``value``, which
    is useful if input dims is not the same with embed dims.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MultiheadAttention, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims  # input_dims:None  embed_dims: 384
        self.embed_dims = embed_dims  # embed_dims: 384
        self.num_heads = num_heads  # num_head: 6
        self.v_shortcut = v_shortcut  # v_shortcut: False

        self.head_dims = embed_dims // num_heads  # head_dims: 64
        self.scale = qk_scale or self.head_dims ** -0.5  # qk_scale: None  scale: 64**-0.5=0.125
        # Linear(in_features=384, out_features=1152, bias=True)
        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)  # (384, 384*3=1152, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout(p=0.0, inplace=False)
        self.proj = nn.Linear(embed_dims, embed_dims,
                              bias=proj_bias)  # Linear(in_features=384, out_features=384, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout(p=0.0, inplace=False)
        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape  # B: 32, N: 197, _: 384  x: 32, 197, 384
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)  # qkv: (3, 32, 6, 197, 64)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q: (32, 6, 197, 64)  k: (32, 6, 197, 64)  v: (32, 6, 197, 64)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # attn: (32, 6, 197, 197)
        attn = torch.sigmoid(attn)  # attn: (32, 6, 197, 197)
        attn = attn.softmax(dim=-1)  # attn: (32, 6, 197, 197)
        # attn = torch.sigmoid(attn)
        attn = self.attn_drop(attn)  # attn: (32, 6, 197, 197)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)  # x: 32, 197, 384
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class MSS_block(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 init_cfg=None):
        super(MSS_block, self).__init__(init_cfg=init_cfg)

        self.input_dims = input_dims or embed_dims  # input_dims:None  embed_dims: 384
        self.embed_dims = embed_dims  # embed_dims: 384
        self.num_heads = num_heads  # num_head: 6
        self.v_shortcut = v_shortcut  # v_shortcut: False

        self.head_dims = embed_dims // num_heads  # head_dims: 64
        self.scale = qk_scale or self.head_dims ** -0.5  # qk_scale: None  scale: 64**-0.5=0.125
        # Linear(in_features=384, out_features=1152, bias=True)
        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)  # (384, 384*3=1152, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)  # Dropout(p=0.0, inplace=False)
        self.proj = nn.Linear(embed_dims, embed_dims,
                              bias=proj_bias)  # Linear(in_features=384, out_features=384, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)  # Dropout(p=0.0, inplace=False)
        self.out_drop = DROPOUT_LAYERS.build(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape  # B 32 N 197 _ 384
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dims).permute(2, 0, 3, 1, 4)  # qkv: [3, 32, 6, 197, 64]
        q, k, v = qkv[0], qkv[1], qkv[2]  # q k v [32, 6, 197, 64]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # att [32, 6, 197, 197]
        attn = torch.sigmoid(attn)
        attn = attn.softmax(dim=-1)
        # attn = torch.sigmoid(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)  # v [32, 6, 197, 64]  att [32, 6, 197, 197]  x [32, 197, 384]
        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

# class MSP_block(BaseModule):
#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  input_dims=None,
#                  attn_drop=0.,
#                  proj_drop=0.,
#                  dropout_layer=dict(type='Dropout', drop_prob=0.),
#                  qkv_bias=True,
#                  qk_scale=None,
#                  proj_bias=True,
#                  v_shortcut=False,
#                  init_cfg=None):
#         super(MSP_block, self).__init__(init_cfg=init_cfg)
#
#         self.input_dims = input_dims or embed_dims  # input_dims:None  embed_dims: 384
#         self.embed_dims = embed_dims  # embed_dims: 384
#         self.num_heads = num_heads  # num_head: 6
#         self.v_shortcut = v_shortcut  # v_shortcut: False
#
#         self.head_dims = embed_dims // num_heads  # head_dims: 64
#         self.scale = qk_scale or self.head_dims ** -0.5  # qk_scale: None  scale: 64**-0.5=0.125
#         # Linear(in_features=384, out_features=1152, bias=True)
#         self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)  # (384, 384*3=1152, bias=True)
#         self.attn_drop = nn.Dropout(attn_drop)  # Dropout(p=0.0, inplace=False)
#         self.proj = nn.Linear(embed_dims, embed_dims,
#                               bias=proj_bias)  # Linear(in_features=384, out_features=384, bias=True)
#         self.proj_drop = nn.Dropout(proj_drop)  # Dropout(p=0.0, inplace=False)
#         self.out_drop = DROPOUT_LAYERS.build(dropout_layer)
#
#     def forward(self, x):  # x [32, 197, 384]
#         B, N, _ = x.shape  # B 32 N 197 _ 384
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
#                                   self.head_dims).permute(2, 0, 3, 1, 4)  # qkv: [3, 32, 6, 197, 64]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # q k v [32, 6, 197, 64]
#
#         input_size = q.shape  # [32, 6, 197, 64]
#         # 定义不同尺寸的池化尺寸列表
#         pool_sizes = [(2, 2), (3, 3), (4, 4)]
#         # 初始化空列表用于存储池化后的结果
#         pooled_q = []
#         pooled_k = []
#         pooled_v = []
#         # 针对每个池化尺寸进行池化操作
#         for pool_size in pool_sizes:
#             # 对 q 进行池化操作，并调整尺寸为统一尺寸
#             q_pooled = F.adaptive_max_pool2d(q, pool_size)
#             q_pooled = F.interpolate(q_pooled, size=input_size[2:], mode='bilinear')  # [32, 6, 197, 64]
#             # 对 k 进行池化操作，并调整尺寸为统一尺寸
#             k_pooled = F.adaptive_max_pool2d(k, pool_size)
#             k_pooled = F.interpolate(k_pooled, size=input_size[2:], mode='bilinear')  # [32, 6, 197, 64]
#             # 对 v 进行池化操作，并调整尺寸为统一尺寸
#             v_pooled = F.adaptive_max_pool2d(v, pool_size)
#             v_pooled = F.interpolate(v_pooled, size=input_size[2:], mode='bilinear')  # [32, 6, 197, 64]
#
#             # 将池化后的结果添加到列表中
#             pooled_q.append(q_pooled.unsqueeze(0))
#             pooled_k.append(k_pooled.unsqueeze(0))
#             pooled_v.append(v_pooled.unsqueeze(0))
#
#         # torch.stack(pooled_q, dim=0).sum(dim=0)
#
#         # 将列表中的张量拼接起来
#         # pooled_q = torch.cat(pooled_q)  # [3, 32, 6, 197, 64]
#         # pooled_k = torch.cat(pooled_k)  # [3, 32, 6, 197, 64]
#         # pooled_v = torch.cat(pooled_v)  # [3, 32, 6, 197, 64]
#         # 先将列表中的张量按维度0堆叠成一个新的张量
#         # stacked_q = torch.stack(pooled_q, dim=0)  # shape: [num_pool_sizes, batch_size, num_channels, height, width]
#         # # 按维度0求和，即对列表中的所有张量按元素相加
#         # pooled_q = torch.sum(stacked_q, dim=0)
#         #
#         # # 先将列表中的张量按维度0堆叠成一个新的张量
#         # stacked_k = torch.stack(pooled_k, dim=0)  # shape: [num_pool_sizes, batch_size, num_channels, height, width]
#         # # 按维度0求和，即对列表中的所有张量按元素相加
#         # pooled_k = torch.sum(stacked_k, dim=0)
#         #
#         # # 先将列表中的张量按维度0堆叠成一个新的张量
#         # stacked_v = torch.stack(pooled_v, dim=0)  # shape: [num_pool_sizes, batch_size, num_channels, height, width]
#         # # 按维度0求和，即对列表中的所有张量按元素相加
#         # pooled_v = torch.sum(stacked_v, dim=0)
#
#         pooled_q = torch.prod(torch.stack(pooled_q), dim=0).sum(dim=0)  # 对列表中的元素进行相乘   # [32, 6, 197, 64]
#         pooled_k = torch.prod(torch.stack(pooled_k), dim=0).sum(dim=0) # 对列表中的元素进行逐元素相乘 # [32, 6, 197, 64]
#         pooled_v = sum(pooled_v).sum(dim=0)  # [32, 6, 197, 64]
#
#         attn = (pooled_q @ pooled_k.transpose(-2, -1)) * self.scale  # att [32, 6, 197, 197]
#         # attn = (q @ k.transpose(-2, -1)) * self.scale  # att [32, 12, 28, 28]
#         # attn = torch.sigmoid(attn)
#         attn = attn.softmax(dim=-1)  # v [32, 6, 197, 197]
#         # attn = torch.sigmoid(attn)
#         attn = self.attn_drop(attn)  # att [32, 6, 197, 197]
#         # v [32, 6, 197, 64]  att [32, 6, 197, 197]  x [32, 197, 384]
#         x = torch.add(attn @ pooled_v, pooled_q)  # pooled_v [32, 6, 197, 64]  x [32, 197, 384]
#         # x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)  # x [32, 28, 768]
#         # x = torch.add(x, pooled_q)
#         x = x.transpose(1, 2).reshape(B, N, self.embed_dims)
#         x = self.proj(x)
#         x = self.out_drop(self.proj_drop(x))
#
#         if self.v_shortcut:
#             x = v.squeeze(1) + x
#         return x
#
#
#
# class MultiScaleTransformer(nn.Module):
#     def __init__(self, num_layers, embed_dims, num_heads, scales, input_dims=None, attn_drop=0., proj_drop=0.,
#                  dropout_layer=dict(type='Dropout', drop_prob=0.), qkv_bias=True, qk_scale=None, proj_bias=True,
#                  v_shortcut=False, init_cfg=None):
#         super(MultiScaleTransformer, self).__init__()
#
#         self.num_layers = num_layers
#         self.scales = scales
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             self.layers.append(
#                 MSP_block(embed_dims=embed_dims, num_heads=num_heads, input_dims=input_dims,
#                                            attn_drop=attn_drop, proj_drop=proj_drop, dropout_layer=dropout_layer,
#                                            qkv_bias=qkv_bias, qk_scale=qk_scale, proj_bias=proj_bias,
#                                            v_shortcut=v_shortcut, init_cfg=init_cfg)
#             )
#         self.pooling = nn.ModuleList([nn.AdaptiveAvgPool1d(scale) for scale in scales])
#
#         self.conv = nn.Conv1d(embed_dims * len(scales), embed_dims, kernel_size=1)  # 添加卷积层
#
#     def forward(self, x):
#         pooled_results = []
#         for pool in self.pooling:
#             pooled = pool(x.transpose(1, 2)).transpose(1, 2)
#             pooled_results.append(pooled)
#
#         x = torch.cat(pooled_results, dim=1)
#
#         for layer in self.layers:
#             x = layer(x)
#
#         x = x.transpose(1, 2)  # 将维度转换为 (B, C, N)
#         x = self.conv(x)  # 使用卷积层进行特征转换
#         x = x.transpose(1, 2)  # 将维度转换回 (B, N, C)
#
#         return x
