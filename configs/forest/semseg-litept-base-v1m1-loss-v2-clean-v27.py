_base_ = ["semseg-litept-base-v1m1-loss-v2-clean-v24.py"]

save_path = "/root/autodl-tmp/exp/forest/semseg-litept-base-v1m1-loss-v2-clean-v27"
data_root = "/root/autodl-tmp/dataset/tree/forest"
weight = "/root/autodl-tmp/exp/forest/semseg-litept-base-v1m1-loss-v2-clean-v24/model/model_best.pth"
resume = False

epoch = 100
eval_epoch = 5

scheduler = dict(
    _delete_=True,
    type="OneCycleLR",
    max_lr=[0.0003, 0.00003],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
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
            dict(type="RandomRotateCUDA", angle=[0, 0.12], axis="x", p=0.5),
            dict(type="RandomRotateCUDA", angle=[0, 0.12], axis="y", p=0.5),
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
)
