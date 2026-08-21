"""V28: Small 模型在新数据集上从头训练（服务器 3090 48G）。

配方 = V17（Small + 1M crop + grid 0.02）+ 新数据集 + oversample。
与本地 local16g（Small, crop 100k, 无过采样, batch 1）对照，唯一实质差异：
  crop_point_max（100k vs 1M）与 oversample（无 vs True）。

继承 semseg-litept-small-v1m1-loss-v2-clean.py（注意：该链不含 V22-V27 依赖的
v21 根配置，v21 已不存在于仓库），已包含：
  - 无坡度增强（与本地一致）
  - crop_point_max=1000000 / grid_size=0.02（顶层，运行时注入所有 split）
  - block_xy=40 / overlap=10
  - OneCycleLR max_lr=[0.001, 0.0001] pct_start=0.05、CE+Lovasz+Dice

服务器启动：
  cd /root/autodl-tmp/workshop/cz/LitePT_forest
  conda activate forest
  export PYTHONPATH=./
  mkdir -p /root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-clean-v28
  nohup setsid python tools/train.py \
    --config-file configs/forest/semseg-litept-small-v1m1-loss-v2-clean-v28.py \
    --num-gpus 1 \
    > /root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-clean-v28/train.out 2>&1 < /dev/null &
"""
_base_ = ["semseg-litept-small-v1m1-loss-v2-clean.py"]

save_path = "/root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-clean-v28"
data_root = "/root/autodl-tmp/dataset/forest/forest"
weight = None
resume = False

epoch = 100
eval_epoch = 5
# GPU 上已有其他任务（~0.4G），batch 1 降低激活显存峰值（batch 3 约为 3 倍），
# 避免互相挤占 OOM；速度变慢可接受。crop_point_max=1M 是与本地对照的实验变量，保持不动。
batch_size = 1
num_worker = 1
# crop_point_max=1000000 与 grid_size=0.02 继承自 clean.py，此处不重复声明
empty_cache_per_epoch = True
# val 评估：与本地 local16g 完全相同的口径（300k 圆柱分块，该路径已在
# 16G 卡上完整验证：最大 1.6M 点文件切 18 块评估通过）。超过 30 万体素的
# 文件全部走分块，单块上限 45 万点（hard_cap=1.5x），远低于训练 1M crop
# 的显存包络，评估不可能 OOM。
val_crop_point_max = 300000

data = dict(
    # block_xy=40 / overlap=10 继承自 clean.py
    train=dict(
        # 新数据集 NTC 仅 0.08%（抽样实测），必须过采样
        oversample=True,
        oversample_exclude_classes=[2],
        # CWD(2) 不作为学习目标：标签映射为 ignore_index(-1)，训练损失与
        # val 指标均跳过该类；模型输出仍是 7 类，类别位置保留（推理兼容）。
        # 与 local16g 一致。需配套同步 datasets/forest.py 和
        # engines/hooks/evaluator.py（valid-class 均值），否则崩溃或指标口径不一致。
        ignore_classes=[2],
    ),
    val=dict(
        ignore_classes=[2],
    ),
)
