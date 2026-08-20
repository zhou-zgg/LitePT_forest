_base_ = ["semseg-litept-base-v1m1-loss-v2-clean-v23.py"]

save_path = "/root/autodl-tmp/exp/forest/semseg-litept-base-v1m1-loss-v2-clean-v24"
weight = "/root/autodl-tmp/exp/forest/semseg-litept-base-v1m1-loss-v2-clean-v23/model/model_best.pth"
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
