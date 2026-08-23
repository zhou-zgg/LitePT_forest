_base_ = ["semseg-litept-small-v1m1-loss-v2.py"]

save_path = "exp/forest/semseg-litept-small-v1m1-loss-v2-clean"

data_root = "data/forest"
weight = "exp/forest/pretrained/semseg-litept-small-v1m1-loss-v2-clean-best.pth"
resume = False

epoch = 100
eval_epoch = 5
batch_size = 2
num_worker = 1
crop_point_max = 1000000
empty_cache_per_epoch = True

data = dict(
    block_xy=40,
    overlap=10,
    train=dict(
        transform=[
            dict(type="CenterShiftCUDA", apply_z=True),
            dict(type="RandomDropoutCUDA", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotateCUDA", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # ===== 山地坡度增强 (toggle below: comment out when enough real mountain data) =====
            dict(type="RandomRotateCUDA", angle=[0, 0.12], axis="x", p=0.5),
            dict(type="RandomRotateCUDA", angle=[0, 0.12], axis="y", p=0.5),
            # ===== 山地坡度增强 end =====
            dict(type="RandomFlipCUDA", p=0.5),
            dict(type="RandomJitterCUDA", sigma=0.005, clip=0.02),
            dict(type="GridSampleCUDA", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CylinderCropCUDA", point_max=1000000, mode="random"),
            dict(type="CenterShiftCUDA", apply_z=False),
            dict(type="ToTensorCUDA"),
            dict(type="UpdateCUDA", keys_dict={"grid_size": 0.02}),
            dict(type="CollectCUDA", keys=("coord", "grid_coord", "segment", "grid_size"), feat_keys=("coord",)),
        ]
    ),
)
