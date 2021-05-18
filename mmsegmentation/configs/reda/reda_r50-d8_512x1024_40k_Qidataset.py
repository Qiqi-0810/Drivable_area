_base_ = [
    '../_base_/models/danet_r50-d8.py', '../_base_/datasets/Qidataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type = 'BN', requires_grad=True)
model = dict(
    backbone=dict(
        dilations=(1, 2, 5, 1),
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='REDAHead',
        num_classes=3,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.916721204, 1, 1.793533931])),
    auxiliary_head=dict(
        type='FCNHead',
        num_classes=3,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.916721204, 1, 1.793533931]))
        )