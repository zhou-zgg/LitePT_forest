_base_ = ["semseg-litept-small-v1m1-loss-v2.py"]

data_root = "data/forest"
save_path = "exp/forest/semseg-litept-small-v1m1-loss-v2"
weight = "exp/forest/semseg-litept-small-v1m1/model/model_best.pth"
resume = False

batch_size = 1
crop_point_max = 150000
num_worker = 1

data = dict(
    block_xy=20,
    overlap=5,
)
