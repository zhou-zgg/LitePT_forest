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
)
