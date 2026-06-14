"""Server config: 800K crop + patch2048 + Loss-v2 + clean data + no Elastic on 48GB GPU.

继承 loss-v2 base（enc_patch_size 2048 + CE/Lovasz/Dice），
关闭 Elastic Distortion（保护竖直圆柱特征），
crop_point_max=800K（48GB 显存支持更大感受野）。

服务器启动：
  cd /root/workshop/LitePT_forest
  export PYTHONPATH=./
  nohup setsid python tools/train.py \
    --config-file configs/forest/semseg-litept-small-v1m1-loss-v2-1m-server.py \
    --num-gpus 1 \
    > /root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-1m/train.log 2>&1 < /dev/null &
"""
_base_ = ["semseg-litept-small-v1m1-loss-v2.py"]

save_path = "/root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-1m"

data_root = "/root/autodl-tmp/dataset/tree/forest"
weight = "/root/autodl-tmp/exp/forest/pretrained/semseg-litept-small-v1m1-loss-v2-clean-best.pth"
resume = False

epoch = 60
eval_epoch = 10

batch_size = 1
num_worker = 1
crop_point_max = 800000
empty_cache_per_epoch = True

data = dict(
    block_xy=40,
    overlap=10,
)
