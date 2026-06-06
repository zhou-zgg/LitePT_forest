"""Server config: 1M crop + patch2048 + Loss-v2 + clean data on RTX 4080 32GB.

继承 loss-v2 base（enc_patch_size 2048 + CE/Lovasz/Dice），
覆盖 crop_point_max=1M、batch_size=4、epoch=60、服务器绝对路径。

起点：/root/autodl-tmp/exp/forest/pretrained/semseg-litept-small-v1m1-loss-v2-clean-best.pth
（从本地 scp 上传，本地 mIoU 0.6040）

服务器启动：
  cd /root/workshop/LitePT_forest
  export PYTHONPATH=./
  bash scripts/run_loss_v2_1m_server.sh
"""
_base_ = ["semseg-litept-small-v1m1-loss-v2.py"]

save_path = "/root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-1m"

data_root = "/root/autodl-tmp/forest"
weight = "/root/autodl-tmp/exp/forest/pretrained/semseg-litept-small-v1m1-loss-v2-clean-best.pth"
resume = True

epoch = 60
eval_epoch = 5

batch_size = 4
num_worker = 4
crop_point_max = 1000000
empty_cache_per_epoch = True

data = dict(
    block_xy=40,
    overlap=10,
)
