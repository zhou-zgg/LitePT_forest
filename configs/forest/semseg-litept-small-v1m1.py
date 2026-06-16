_base_ = ["../_base_/default_runtime.py"]

# ===================== 常改配置 =====================
# --- 预处理加速 ---
# True: GPU 加速 (155x), 需手动将 num_worker 降到 1~2，否则多 worker 占显存 OOM
# False: CPU 原有预处理
use_gpu_transform = True

# --- 裁切方式 ---
# "cylinder": Z轴不切，适合森林竖直结构; "sphere": 球形采样
crop_mode = "cylinder"
crop_type = (crop_mode.capitalize() + "CropCUDA") if use_gpu_transform else (crop_mode.capitalize() + "Crop")
crop_point_max = 500000     # CylinderCrop: 500000, SphereCrop: 102400

# --- 体素大小 ---
grid_size = 0.02            # 训练体素大小 (m)

# --- 类别映射 ---
class_mapping = {7: -1}     # 忽略 class 7 噪声点

# --- 数据增强开关 ---
enable_scale = False         # 等比缩放，断树场景建议关闭，保留相对尺度参照
scale_range = [0.9, 1.1]    # enable_scale=True 时生效

# --- 预训练权重（在 local/server 配置中指定路径）---
# weight = None
resume = False              # True: 从 model_last.pth 恢复训练

# ===================== 训练配置 =====================
epoch = 600
eval_epoch = 60
save_path = "exp/forest/semseg-litept-small-v1m1"


enable_wandb = False
batch_size = 4
num_worker = 20
mix_prob = 0.8
empty_cache = False
enable_amp = True
clip_grad = 1.0

# ===================== 模型 =====================
model = dict(
    type="DefaultSegmentorV2",
    num_classes=7,
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 2048, 2048),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# ===================== 优化器 & 调度器 =====================
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# ===================== 数据集 =====================
dataset_type = "ForestDataset"
data_root = "data/forest"

data = dict(
    num_classes=7,
    ignore_index=-1,
    names=[
        "terrain",
        "foliage",
        "CWD",
        "trunk",
        "branch",
        "snag",
        "non-tree-cyl",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_mapping=class_mapping,
        transform=[
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=True),
            dict(
                type="RandomDropoutCUDA" if use_gpu_transform else "RandomDropout",
                dropout_ratio=0.2, dropout_application_ratio=0.2,
            ),
            dict(type="RandomRotateCUDA" if use_gpu_transform else "RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotateCUDA" if use_gpu_transform else "RandomRotate", angle=[-0.21, 0.21], axis="x", p=0.5),
            dict(type="RandomRotateCUDA" if use_gpu_transform else "RandomRotate", angle=[-0.21, 0.21], axis="y", p=0.5),
        ]
        + ([dict(type="RandomScaleCUDA" if use_gpu_transform else "RandomScale", scale=scale_range)] if enable_scale else [])
        + [
            dict(type="RandomFlipCUDA" if use_gpu_transform else "RandomFlip", p=0.5),
            dict(type="RandomJitterCUDA" if use_gpu_transform else "RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSampleCUDA" if use_gpu_transform else "GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="ElasticDistortionCUDA" if use_gpu_transform else "ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type=crop_type, point_max=crop_point_max, mode="random"),
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=False),
            dict(type="ToTensorCUDA" if use_gpu_transform else "ToTensor"),
            dict(type="UpdateCUDA" if use_gpu_transform else "Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="CollectCUDA" if use_gpu_transform else "Collect",
                keys=("coord", "grid_coord", "segment", "grid_size"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        class_mapping=class_mapping,
        transform=[
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=True),
            dict(type="CopyCUDA" if use_gpu_transform else "Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSampleCUDA" if use_gpu_transform else "GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=False),
            dict(type="ToTensorCUDA" if use_gpu_transform else "ToTensor"),
            dict(
                type="CollectCUDA" if use_gpu_transform else "Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSampleCUDA" if use_gpu_transform else "GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=False),
                dict(type="ToTensorCUDA" if use_gpu_transform else "ToTensor"),
                dict(
                    type="CollectCUDA" if use_gpu_transform else "Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord",),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngleCUDA" if use_gpu_transform else "RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=3),
]