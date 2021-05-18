_base_ = '../pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'

norm_cfg = dict(type = 'BN', requires_grad=True)
model = dict(
    backbone=dict(
        type='ResNetV1c',
        norm_cfg=norm_cfg,
        ),
    decode_head=dict(
        type='REPSPHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=(0, 1, 2, 3),
        channels=512,
        input_transform='multiple_select',
        num_classes=3,
        norm_cfg=norm_cfg,
        ),
    auxiliary_head=dict(
        type='FCNHead',
        num_classes=3,
        norm_cfg=norm_cfg,
        ))