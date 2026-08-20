_base_ = ["semseg-litept-base-v1m1-loss-v2-clean-v24.py"]

ROOT = "/root/autodl-tmp/dataset/tree/forest"

def _load_cw():
    import json, numpy as np
    with open(f"{ROOT}/train_class_hist.json") as f:
        hist = json.load(f)["hist"]
    gc = np.zeros(7, dtype=np.float64)
    for fh in hist.values():
        for k, v in fh.items():
            c = int(k)
            if 0 <= c < 7 and c != 2:
                gc[c] += v
    fr = gc / gc.sum()
    w = np.ones(7)
    m = fr > 0
    w[m] = 1.0 / np.sqrt(fr[m])
    w[2] = 1.0
    w /= w[0]
    return [round(x, 3) for x in w.tolist()]

_class_weight = _load_cw()
del _load_cw

save_path = "/root/autodl-tmp/exp/forest/semseg-litept-base-v1m1-loss-v2-clean-v26"
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
        dict(type="CrossEntropyLoss", loss_weight=1.0, label_smoothing=0.1,
             ignore_index=-1, weight=_class_weight),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
        dict(type="DiceLoss", loss_weight=1.0, ignore_index=-1),
    ],
)
