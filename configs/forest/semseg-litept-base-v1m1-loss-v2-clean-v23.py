_base_ = ["semseg-litept-base-v1m1-loss-v2-clean-v22.py"]

save_path = "/root/autodl-tmp/exp/forest/semseg-litept-base-v1m1-loss-v2-clean-v23"
data_root = "/root/autodl-tmp/dataset/tree/forest"
weight = None
resume = False

epoch = 50
eval_epoch = 5
batch_size = 1
num_worker = 0
crop_point_max = 450000
empty_cache_per_epoch = True

scheduler = dict(
    _delete_=True,
    type="CosineAnnealingLR",
    eta_min=1e-6,
)

model = dict(
    backbone=dict(
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 2048, 2048),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 108, 216, 432),
        dec_num_head=(4, 6, 12, 24),
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
)

data = dict(
    block_xy=40,
    overlap=10,
    train=dict(
        oversample=True,
        oversample_exclude_classes=[2],
        transform=[
            dict(type="CenterShiftCUDA", apply_z=True),
            dict(type="RandomDropoutCUDA", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotateCUDA", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomFlipCUDA", p=0.5),
            dict(type="RandomJitterCUDA", sigma=0.005, clip=0.02),
            dict(type="GridSampleCUDA", grid_size=0.025, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CylinderCropCUDA", point_max=450000, mode="random"),
            dict(type="CenterShiftCUDA", apply_z=False),
            dict(type="ToTensorCUDA"),
            dict(type="UpdateCUDA", keys_dict={"grid_size": 0.025}),
            dict(type="CollectCUDA", keys=("coord", "grid_coord", "segment", "grid_size"), feat_keys=("coord",)),
        ]
    ),
    val=dict(
        transform=[
            dict(type="CenterShiftCUDA", apply_z=True),
            dict(type="CopyCUDA", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSampleCUDA", grid_size=0.025, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="CenterShiftCUDA", apply_z=False),
            dict(type="ToTensorCUDA"),
            dict(type="CollectCUDA", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"), feat_keys=("coord",)),
        ],
    ),
    test=dict(
        test_cfg=dict(
            voxelize=dict(
                type="GridSampleCUDA",
                grid_size=0.025,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
        ),
    ),
)
