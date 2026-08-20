_base_ = ["semseg-litept-base-v1m1-loss-v2-clean-v24.py"]

save_path = "/root/autodl-tmp/exp/forest/semseg-litept-base-v1m1-loss-v2-clean-v25"
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

model = dict(
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, label_smoothing=0.1, ignore_index=-1,
             weight=[1.0, 1.0, 1.0, 2.0, 2.0, 5.0, 10.0]),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        dict(type="DiceLoss", loss_weight=1.0, ignore_index=-1),
    ],
)
