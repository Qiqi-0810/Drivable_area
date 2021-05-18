_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/Qidataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type = 'BN', requires_grad=True)
model = dict(
    decode_head=dict(
        type='REPSPHead',
        in_channels=[256, 512, 1024, 2048],
        in_index=(0, 1, 2, 3),
        channels=512,
        input_transform='multiple_select',
        num_classes=3,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.916721204, 1, 1.793533931]),
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000)
        ),
    auxiliary_head=dict(
        type='FCNHead',
        num_classes=3,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.916721204, 1, 1.793533931]))
    )