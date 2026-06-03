_base_ = ["../_base_/default_runtime.py", "../_base_/schedules/semseg_200e.py"]

model = dict(
    type="LitePointNet",
    in_channels=3,
    num_classes=2,
)

data = dict(
    train=dict(
        type="ForestDataset",
        data_root="data/forest/stage2_train",
        split="stage2_train",
    ),
    val=dict(
        type="ForestDataset",
        data_root="data/forest/stage2_train",
        split="stage2_train",
    ),
    test=dict(
        type="ForestDataset",
        data_root="data/forest/stage2_train",
        split="stage2_train",
    ),
    num_classes=2,
    names=["trunk", "branch"],
    ignore_index=255,
)

batch_size = 4
num_worker = 4

epoch = 200
eval_epoch = 20

enable_wandb = False

save_path = "exp/forest/semseg-litept-small-v1m1-stage2"
