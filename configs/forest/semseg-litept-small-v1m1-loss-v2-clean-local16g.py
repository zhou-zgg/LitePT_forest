_base_ = ["semseg-litept-small-v1m1-loss-v2-clean.py"]

# 本机 16G 专用冒烟配置：数据路径切到新数据集
data_root = "data/forest_new"
save_path = "exp/forest/semseg-litept-small-v1m1-clean-local16g"
weight = None
resume = False

epoch = 100
eval_epoch = 5
# val 整图评估已在 evaluator.py 加了 empty_cache 释放显存，可正常自动评估
evaluate = True
batch_size = 1
num_worker = 0
crop_point_max = 100000
# val 评估用圆柱形空间分块（与训练 CylinderCropCUDA 同分布）：
# 大文件不再整图前向，而是按点密度自适应半径切成多个圆柱块全覆盖，
# softmax 概率累加合并后统一算指标。评估无梯度/优化器开销，
# 块上限可取训练 crop_point_max 的 3~5 倍。
val_crop_point_max = 300000
empty_cache_per_epoch = True

data = dict(
    block_xy=20,
    overlap=5,
    # CWD(2) 不作为学习目标：标签映射为 ignore_index(-1)，训练损失与 val 指标
    # 均跳过该类；模型输出仍是 7 类，类别位置保留，推理/下游项目兼容。
    train=dict(ignore_classes=[2]),
    # val grid 与训练一致（0.02）。大文件的整图 OOM 问题已由 evaluator.py 的
    # 圆柱分块评估（val_crop_point_max）解决，不再需要放粗网格。
    val=dict(
        ignore_classes=[2],
        transform=[
            dict(type="CenterShiftCUDA", apply_z=True),
            dict(type="CopyCUDA", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSampleCUDA", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="CenterShiftCUDA", apply_z=False),
            dict(type="ToTensorCUDA"),
            dict(type="CollectCUDA", keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"), feat_keys=("coord",)),
        ]
    ),
)