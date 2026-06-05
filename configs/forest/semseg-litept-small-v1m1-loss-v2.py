_base_ = ["semseg-litept-small-v1m1.py"]

save_path = "exp/forest/semseg-litept-small-v1m1-loss-v2"

epoch = 30
eval_epoch = 5

model = dict(
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, label_smoothing=0.1, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        dict(type="DiceLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]
