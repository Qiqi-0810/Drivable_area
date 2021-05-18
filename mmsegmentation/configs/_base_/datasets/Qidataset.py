# dataset settings
dataset_type = 'QiqiDataset'
data_root = '/content/drive/MyDrive/Colab Notebooks/bdd100k/'
img_norm_cfg = dict(
    mean=[75.32, 78.63, 77.82], std=[65.15, 69.25, 71.95], to_rgb=True)
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pipeline=train_pipeline,
        split='splits/train480_all.txt'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline,
        split='splits/val160_all.txt'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline,
        split='splits/test_all.txt'))
