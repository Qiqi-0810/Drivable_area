_base_ = '../danet/danet_r50-d8_512x1024_40k_cityscapes.py'

norm_cfg = dict(type = 'BN', requires_grad=True)
model = dict(
    backbone=dict(
        type='ResNetV1c',
        norm_cfg=norm_cfg,
        ),
    decode_head=dict(
        type='REDAHead',
        num_classes=3,
        norm_cfg=norm_cfg,
        ),
    auxiliary_head=dict(
        type='FCNHead',
        num_classes=3,
        norm_cfg=norm_cfg,
        ))