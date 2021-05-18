import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import nn

from torch.nn import init#
import functools#
from torch.autograd import Variable#
import numpy as np#

from mmseg.core import add_prefix
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead
from mmseg.ops import resize


class PAM(_SelfAttentionBlock):
    """Position Attention Module (PAM)
    Args:
        in_channels (int): Input channels of key//query feature.
        channels (int): Output channels of key//query transform.
    """

    def __init__(self, in_channels, channels):
        super(PAM, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            key_query_norm=False,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=False,
            with_out=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

        self.gamma = Scale(0)## This layer scales the input by a learnable factor. It multiplies a learnable scale parameter of shape (1,) with input of any shape. 就是里面的α

    def forward(self, x):
        """Forward function."""
        out = super(PAM, self).forward(x, x)

        out = self.gamma(out) + x
        return out
        
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(input_channels, output_channels//4, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(output_channels//4, output_channels//4, 3, stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels//4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_channels//4, output_channels, 1, 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(input_channels, output_channels , 1, stride, bias = False)
        
    def forward(self, x):
        residual = x
        out_c1 = self.conv1(x)
        out_b1 = self.bn1(out_c1)
        out_r1 = self.relu(out_b1)
        out_c2 = self.conv2(out_r1)
        out_b2 = self.bn2(out_c2)
        out_r2 = self.relu(out_b2)
        out_c3 = self.conv3(out_r2)
        out_b3 = self.bn3(out_c3)
        out_r3 = self.relu(out_b3)
        if (self.input_channels != self.output_channels) or (self.stride !=1 ):
            residual = self.conv4(out_b1)
        out = out_r3+residual
        return out

class AttentionModule(nn.Module):##in和out改成一样的，size变成具体的数
    def __init__(self, in_channels, height, width):
        super(AttentionModule, self).__init__()

        self.first_residual_blocks = ResidualBlock(in_channels, in_channels)

        self.mpool1 = nn.AdaptiveMaxPool2d((height//2,width//2))

        self.softmax1_blocks = ResidualBlock(in_channels, in_channels)

        self.skip1_connection_residual_block = ResidualBlock(in_channels, in_channels)

        self.mpool2 = nn.AdaptiveMaxPool2d((height//4,width//4))

        self.softmax2_blocks = ResidualBlock(in_channels, in_channels)

        self.skip2_connection_residual_block = ResidualBlock(in_channels, in_channels)

        self.mpool3 = nn.AdaptiveMaxPool2d((height//8,width//8))

        self.softmax3_blocks = nn.Sequential(
            ResidualBlock(in_channels, in_channels),
            ResidualBlock(in_channels, in_channels)
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=(height//4,width//4))

        self.softmax4_blocks = ResidualBlock(in_channels, in_channels)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=(height//2,width//2))

        self.softmax5_blocks = ResidualBlock(in_channels, in_channels)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=(height,width))

        '''self.softmax6_blocks = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels , kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels , kernel_size = 1, stride = 1, bias = False),
            nn.Sigmoid()
        )'''

        self.last_blocks = ResidualBlock(in_channels, in_channels)

    def forward(self, x):
        x = self.first_residual_blocks(x)
        #print(x.shape)
        out_mpool1 =  self.mpool1(x)
        #print('out_mpool1',out_mpool1.shape)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        #print('out_softmax1',out_softmax1.shape)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)#
        #print('out_skip1_connection',out_skip1_connection.shape)
        out_mpool2 = self.mpool2(out_softmax1)
        #print('out_mpool2',out_mpool2.shape)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        #print('out_softmax2',out_softmax2.shape)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)#
        #print('out_skip2_connection',out_skip2_connection.shape)
        out_mpool3 = self.mpool3(out_softmax2)
        #print('out_mpool3.shape',out_mpool3.shape)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        #print('out_softmax3.shape',out_softmax3.shape)
        out_interp3 = self.interpolation3(out_softmax3)
        #print('out_skip2_connection.shape',out_skip2_connection.shape)
        #print('out_interp3.shape',out_interp3.shape)
        out = out_interp3 + out_skip2_connection#
        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = self.interpolation2(out_softmax4)
        out1 = out_interp2 + out_skip1_connection#
        out_softmax5 = self.softmax5_blocks(out1)
        out_interp1 = self.interpolation1(out_softmax5)
        out2 = self.last_blocks(out_interp1)
        #out_softmax6 = self.softmax6_blocks(out_interp1)###attention的输出
        return out2


class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self, in_channels, conv_cfg, norm_cfg, act_cfg):##
        super(CAM, self).__init__()
        self.gamma = Scale(0)##This layer scales the input by a learnable factor. It multiplies a learnable scale parameter of shape (1,) with input of any shape. 就是里面的β
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.input_redu_conv = ConvModule(##
            self.in_channels,
            self.in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
            
        self.attention_module = AttentionModule(self.in_channels, 56, 56)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()#C*H*W###?
        y = self.input_redu_conv(x)###
        
        z = self.attention_module(x)###
        a = resize(z, size=(height, width))###
        proj_query = y.view(batch_size, channels, -1) ###batch_size1 *C*(hw)=batch_size1*C*N
        proj_key = a.view(batch_size, channels, -1).permute(0, 2, 1)##batch_size1*(hw)*C
        energy = torch.bmm(proj_query, proj_key)#3维矩阵乘法，如size（2，2，5）的cc和size（2，5，2）dd，torch.bmm(cc,dd)后size=（2，2，2）。这里batch_size*C*C
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy#torch.max()[0]， 只返回最大值的每个数,troch.max()[1]， 只返回最大值的每个索引
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)

        out = torch.bmm(attention, proj_value)
        out1 = out.view(batch_size, channels, height, width)

        out2 = self.gamma(out1) + x
        return out2


@HEADS.register_module()
class REDAHead(BaseDecodeHead):
    """Dual Attention Network for Scene Segmentation.
    This head is the implementation of `DANet
    <https:////arxiv.org//abs//1809.02983>`_.
    Args:
        pam_channels (int): The channels of Position Attention Module(PAM).
    """

    def __init__(self, pam_channels, **kwargs):###
        super(REDAHead, self).__init__(**kwargs)
        self.pam_channels = pam_channels
        self.pam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam = PAM(self.channels, pam_channels)
        self.pam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

        self.cam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam = CAM(
            self.channels, 
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)##
        self.cam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

    def pam_cls_seg(self, feat):
        """PAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.pam_conv_seg(feat)
        return output

    def cam_cls_seg(self, feat):
        """CAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.cam_conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        pam_feat = self.pam_in_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_feat = self.pam_out_conv(pam_feat)
        pam_out = self.pam_cls_seg(pam_feat)

        cam_feat = self.cam_in_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_out = self.cam_cls_seg(cam_feat)

        feat_sum = pam_feat + cam_feat
        pam_cam_out = self.cls_seg(feat_sum)

        return pam_cam_out, pam_out, cam_out

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``pam_cam`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label):
        """Compute ``pam_cam``, ``pam``, ``cam`` loss."""
        pam_cam_seg_logit, pam_seg_logit, cam_seg_logit = seg_logit
        loss = dict()
        loss.update(
            add_prefix(
                super(REDAHead, self).losses(pam_cam_seg_logit, seg_label),
                'pam_cam'))
        loss.update(
            add_prefix(
                super(REDAHead, self).losses(pam_seg_logit, seg_label), 'pam'))
        loss.update(
            add_prefix(
                super(REDAHead, self).losses(cam_seg_logit, seg_label), 'cam'))
        return loss