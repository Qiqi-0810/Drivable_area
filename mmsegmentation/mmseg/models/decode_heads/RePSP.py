import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import numpy as np
import torch.nn.functional as F


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels[3]
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""      
        ppm_outs = []
        for ppm in self:
            ppm_out =(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)##upsampled_ppm_outHW和x一样
            ppm_outs.append(upsampled_ppm_out)
            #ppm_outs = self.gamma(ppm_outs)##
        return ppm_outs


@HEADS.register_module()
class REPSPHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.
    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(REPSPHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.channels*2 + len(pool_scales) * 2048,##z这个怎么搞得反过来了？？？
            ####在cfg是list，是不是和index配合，这里没有是list的可能性？也不对把？？？？？不行就全变成数
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_info0 = ConvModule(
            256,
            1024,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.global_info1 = ConvModule(
            512,
            1024,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.alpha = Scale(0)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)##选哪个阶段的backbone作为input
        psp_outs = []##
        psp_outs.extend(self.psp_modules(x[3]))##执行pooling和双线性插值然后存放1de-2de-3de-6de
        a = resize(self.global_info0(F.adaptive_max_pool2d(x[0], 1)), size=x[2].shape[2:])
        b = resize(self.global_info1(F.adaptive_max_pool2d(x[1], 1)), size=x[2].shape[2:])
        y = self.alpha(x[2]+a+b)##和x大小一样,α[x+g(x)]，若加别的层，注意算bottleneck的channel
        psp_outs.append(y)
        psp_outs = torch.cat(psp_outs, dim=1)##四维的，dim=1是C
        output = self.bottleneck(psp_outs)##x和g(x)是相加，大小不变，上面的定义中channel不变
        output = self.cls_seg(output)
        return output