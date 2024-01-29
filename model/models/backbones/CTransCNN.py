from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import DropPath
from mmcv.cnn.bricks.transformer import AdaptivePadding
from mmcv.cnn.utils.weight_init import trunc_normal_

from model.utils import get_root_logger
from ..builder import BACKBONES
from .base_backbone import BaseBackbone, BaseModule
from .my_transformer import TransformerDecoderLayer


class ConvBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 drop_path_rate=0.,
                 with_residual_conv=False,
                 norm_cfg=dict(type='BN', eps=1e-6),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(ConvBlock, self).__init__(init_cfg=init_cfg)

        expansion = 4
        mid_channels = out_channels // expansion

        # 1x1 Conv-BN ReLU
        self.conv1 = nn.Conv2d(
            in_channels,  # 64
            mid_channels,  # 16
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn1 = build_norm_layer(norm_cfg, mid_channels)[1]
        self.act1 = build_activation_layer(act_cfg)

        self.conv_identity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False)
        self.bn_out = build_norm_layer(norm_cfg, out_channels)[1]

        # 3x3 Conv-BN ReLU
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            groups=groups,
            padding=1,
            bias=False)
        self.bn2 = build_norm_layer(norm_cfg, mid_channels)[1] 
        self.act2 = build_activation_layer(act_cfg) 

        # 1x1 Conv-BN ReLU
        self.conv3 = nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)
        self.bn3 = build_norm_layer(norm_cfg, out_channels)[1]
        self.act3 = build_activation_layer(act_cfg)

        self.conv_64_16 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        if with_residual_conv:
            # 1x1 Conv-BN ReLU
            self.residual_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False)
            self.residual_bn = build_norm_layer(norm_cfg, out_channels)[1]

        self.with_residual_conv = with_residual_conv
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, fusion_features=None, out_conv2=True):
        identity = x 

        # 1x1 Conv-BN ReLU
        x = self.conv1(x)
        x = self.bn1(x)

        # method2 add 1x1 bn
        x_identity_bn = self.conv_identity(identity)
        x_identity_bn = self.bn_out(x_identity_bn)

        x = self.act1(x)

        x_1 = self.conv2(x) if fusion_features is None else self.conv2(
            x + fusion_features)
        x = self.bn2(x_1)

        # add rep method1
        x_conv1_bn = self.conv_64_16(x_1) 
        x_bn = self.bn2(x_1)
        x = x + x_conv1_bn + x_bn

        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.with_residual_conv:
            identity = self.residual_conv(identity)
            identity = self.residual_bn(identity)

        x = x + identity + x_identity_bn

        x = self.act3(x)

        if out_conv2:  # out_conv2 为 True
            return x, x2
        else:
            return x


# CNN feature maps -> Transformer patch embeddings
class C2T(BaseModule):

    def __init__(self,
                 in_channels,  # in_channels： 16
                 out_channels,  # out_channels;384
                 down_stride,  # down_stride：4
                 with_cls_token=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 init_cfg=None):
        super(C2T, self).__init__(
            init_cfg=init_cfg)
        self.down_stride = down_stride
        self.with_cls_token = with_cls_token

        # 1x1 Conv-AvgPooling
        self.conv_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(
            kernel_size=down_stride, stride=down_stride)

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # LayerNorm-GELU
        self.ln = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

    def forward(self, x, x_t):
        # 1x1 Conv-AvgPooling-LayerNorm-GELU
        x = self.conv_project(x)  # [N, C, H, W]
        # [N, C, H, W]-->[N, C, H`, W`]-->[N, C, H` * W`]-->[N, H` * W`, C]
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.sigmoid(x1)
        x1 = self.conv2(x1)
        x = x1 + x
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)

        x = self.ln(x)  # LayerNorm
        x = self.act(x)  # GELU

        if self.with_cls_token:
            x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


# Transformer patch embeddings -> CNN feature maps
class T2C(BaseModule):

    def __init__(self,
                 in_channels,  # in_channels: 384
                 out_channels,  # out_channels:1 6
                 up_stride,  # up_stride: 4
                 with_cls_token=True,
                 norm_cfg=dict(type='BN', eps=1e-6),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super(T2C, self).__init__(
            init_cfg=init_cfg)

        self.up_stride = up_stride
        self.with_cls_token = with_cls_token

        # 1x1 Conv-BN-ReLU
        self.conv_project = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = build_activation_layer(act_cfg)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        if self.with_cls_token: 
            x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        else:
            x_r = x.transpose(1, 2).reshape(B, C, H, W)
        # 1x1 Conv-BN-ReLU
        x_r = self.act(self.bn(self.conv_project(x_r)))

        x = F.interpolate(
            x_r, size=(H * self.up_stride, W * self.up_stride), mode='bilinear')

        x1 = self.conv2(x)
        x1 = self.relu2(x1)

        x1 = self.conv3(x1)
        x1 = self.sigmoid(x1)

        out = x + x1
        return out

class IIMBlock(BaseModule):

    def __init__(self,
                 in_channels,  # in_channels: 64
                 out_channels,  # out_channels: 64
                 embed_dims,  # embed_dims: 384
                 conv_stride=1,
                 groups=1,
                 with_residual_conv=False,
                 down_stride=4,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 with_cls_token=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 last_fusion=False,
                 init_cfg=None):
        super(IIMBlock, self).__init__(init_cfg=init_cfg)
        expansion = 4

        self.cnn_block = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            with_residual_conv=with_residual_conv,
            stride=conv_stride,
            groups=groups)

        if last_fusion:  # last_fusion: True
            self.fusion_block = ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=2,
                with_residual_conv=True,
                groups=groups,
                drop_path_rate=drop_path_rate)
        else:  # last_fusion: False
            self.fusion_block = ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                groups=groups,
                drop_path_rate=drop_path_rate)

        self.squeeze_block = C2T(
            in_channels=out_channels // expansion,
            out_channels=embed_dims,
            down_stride=down_stride,
            with_cls_token=with_cls_token)

        self.expand_block = T2C(
            in_channels=embed_dims,
            out_channels=out_channels // expansion,
            up_stride=down_stride,
            with_cls_token=with_cls_token)

        # Transformer
        self.trans_block = TransformerDecoderLayer(
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=int(embed_dims * mlp_ratio),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            qkv_bias=qkv_bias,
            norm_cfg=dict(type='LN', eps=1e-6))

        self.down_stride = down_stride
        self.embed_dim = embed_dims
        self.last_fusion = last_fusion

    def forward(self, cnn_input, trans_input):
        x, x_conv2 = self.cnn_block(cnn_input, out_conv2=True)

        _, _, H, W = x_conv2.shape

        # Convert the feature map of conv2 to transformer embedding
        # and concat with class token.
        conv2_embedding = self.squeeze_block(x_conv2, trans_input)

        trans_output = self.trans_block(conv2_embedding + trans_input)

        # Convert the transformer output embedding to feature map
        trans_features = self.expand_block(trans_output, H // self.down_stride,
                                           W // self.down_stride)
        x = self.fusion_block(
            x, fusion_features=trans_features, out_conv2=False)

        return x, trans_output


@BACKBONES.register_module()
class my_hybird_CTransCNN(BaseBackbone):
    arch_zoo = {
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': 384,
                         'channel_ratio': 1,
                         'num_heads': 6,
                         'depths': 12
                         })
    }

    _version = 1

    def __init__(self,
                 arch='tiny',
                 patch_size=16,
                 base_channels=64,
                 mlp_ratio=4.,  # mlp_ratio:4.0
                 qkv_bias=True,
                 with_cls_token=True,
                 drop_path_rate=0.,  # drop_path_rate: 0.1
                 norm_eval=True,
                 frozen_stages=0,
                 out_indices=-1,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):  # arch为tiny
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'depths', 'num_heads', 'channel_ratio'
            }
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.num_features = self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths'] 
        self.num_heads = self.arch_settings['num_heads']
        self.channel_ratio = self.arch_settings['channel_ratio'] 

        if isinstance(out_indices, int): 
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices): 
            if index < 0:
                out_indices[i] = self.depths + index + 1 
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices  # out_indices = [12]

        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages

        self.with_cls_token = with_cls_token
        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        self.trans_dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, self.depths)
        ]

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1)

        assert patch_size % 16 == 0, 'The patch size of CTransCNN must ' \
                                     'be divisible by 16.'
        trans_down_stride = patch_size // 4

        self.auto_pad = AdaptivePadding(trans_down_stride,
                                        trans_down_stride)

        stage1_channels = int(base_channels * self.channel_ratio)
        self.conv_1 = ConvBlock(
            in_channels=64,
            out_channels=stage1_channels,
            with_residual_conv=True,
            stride=1)
        self.trans_patch_conv = nn.Conv2d(
            64,
            self.embed_dims,
            kernel_size=trans_down_stride,
            stride=trans_down_stride,
            padding=0)

        self.trans_1 = TransformerDecoderLayer(
            embed_dims=self.embed_dims,
            num_heads=self.num_heads,
            feedforward_channels=int(self.embed_dims * mlp_ratio),
            drop_path_rate=self.trans_dpr[0],
            qkv_bias=qkv_bias,
            norm_cfg=dict(type='LN', eps=1e-6))

        init_stage = 2
        fin_stage = self.depths // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module(
                f'conv_trans_{i}',
                IIMBlock(
                    in_channels=stage1_channels,
                    out_channels=stage1_channels,
                    embed_dims=self.embed_dims,
                    conv_stride=1,
                    with_residual_conv=False,
                    down_stride=trans_down_stride,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=self.trans_dpr[i - 1],
                    with_cls_token=self.with_cls_token))

        stage2_channels = int(base_channels * self.channel_ratio * 2)
        init_stage = fin_stage
        fin_stage = fin_stage + self.depths // 3
        for i in range(init_stage, fin_stage):
            if i == init_stage:
                conv_stride = 2
                in_channels = stage1_channels
            else:
                conv_stride = 1
                in_channels = stage2_channels

            with_residual_conv = True if i == init_stage else False
            self.add_module(
                f'conv_trans_{i}',
                IIMBlock(
                    in_channels=in_channels,
                    out_channels=stage2_channels,
                    embed_dims=self.embed_dims,
                    conv_stride=conv_stride,
                    with_residual_conv=with_residual_conv,
                    down_stride=trans_down_stride // 2,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=self.trans_dpr[i - 1],
                    with_cls_token=self.with_cls_token))

        stage3_channels = int(base_channels * self.channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + self.depths // 3  # 13
        for i in range(init_stage, fin_stage):
            if i == init_stage:
                conv_stride = 2
                in_channels = stage2_channels
                with_residual_conv = True
            else:
                conv_stride = 1
                in_channels = stage3_channels
                with_residual_conv = False

            last_fusion = (i == self.depths)

            self.add_module(
                f'conv_trans_{i}',
                IIMBlock(
                    in_channels=in_channels,
                    out_channels=stage3_channels,
                    embed_dims=self.embed_dims,
                    conv_stride=conv_stride,
                    with_residual_conv=with_residual_conv,
                    down_stride=trans_down_stride // 4,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path_rate=self.trans_dpr[i - 1],
                    with_cls_token=self.with_cls_token,
                    last_fusion=last_fusion))
        self.fin_stage = fin_stage

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.trans_norm = nn.LayerNorm(self.embed_dims)

        if self.with_cls_token:
            trunc_normal_(self.cls_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

        if hasattr(m, 'zero_init_last_bn'):
            m.zero_init_last_bn()

    def init_weights(self):
        super(my_hybird_CTransCNN, self).init_weights()
        logger = get_root_logger()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress default init if use pretrained model.
            return
        else:
            logger.info(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)

    def forward(self, x):
        output = []
        B = x.shape[0]
        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)

        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        x_base = self.auto_pad(x_base)

        x = self.conv_1(x_base, out_conv2=False)
        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        if self.with_cls_token:
            x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)

        for i in range(2, self.fin_stage):
            stage = getattr(self, f'conv_trans_{i}')
            x, x_t = stage(x, x_t)
            if i in self.out_indices:
                if self.with_cls_token:
                    output.append([
                        self.pooling(x).flatten(1),
                        self.trans_norm(x_t)[:, 0]
                    ])
                else:
                    output.append([
                        self.pooling(x).flatten(1),
                        self.trans_norm(x_t).mean(dim=1)
                    ])

        return tuple(output)
