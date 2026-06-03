"""
Stage2 数据预处理脚本

从 Stage1 推理结果中筛选 woody 点（预测为 class 3），并只保留原始标签为 trunk(3) 或 branch(6) 的点。
生成新的 LAS 文件用于 Stage2 训练。

用法:
    # 1. 先用 Stage1 对训练集推理（生成 *_pred.npy）:
    PYTHONPATH=./ python tools/test.py \
      --config-file configs/forest/semseg-litept-small-v1m1-stage1.py \
      --num-gpus 1 \
      --options save_path=exp/forest/semseg-litept-small-v1m1-stage1 \
      weight=exp/forest/semseg-litept-small-v1m1-stage1/model/model_best.pth \
      test.type=SemSegTester test.aug_transform=[] \
      data.test.split=train

    # 2. 运行本脚本生成 Stage2 训练数据:
    PYTHONPATH=./ python tools/prepare_stage2_data.py
"""

import os
import glob
import numpy as np
import laspy

STAGE1_RESULT_ROOT = "exp/forest/semseg-litept-small-v1m1-stage1/result"
TRAIN_DATA_ROOT = "data/forest/train"
OUTPUT_ROOT = "data/forest/stage2_train"

STAGE1_WOODY_CLASS = 3


def prepare_stage2_data():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    las_files = glob.glob(os.path.join(TRAIN_DATA_ROOT, "*.las"))
    print(f"Found {len(las_files)} LAS files in {TRAIN_DATA_ROOT}")

    for las_path in las_files:
        las_name = os.path.basename(las_path)
        pred_path = os.path.join(STAGE1_RESULT_ROOT, f"{las_name}_pred.npy")

        if not os.path.exists(pred_path):
            print(f"  [SKIP] No prediction found: {pred_path}")
            continue

        out_path = os.path.join(OUTPUT_ROOT, las_name)
        if os.path.exists(out_path):
            print(f"  [SKIP] Already exists: {out_path}")
            continue

        las = laspy.read(las_path)
        pred = np.load(pred_path)
        labels = las["label"]

        woody_mask = (pred == STAGE1_WOODY_CLASS)
        trunk_mask = (labels == 3)
        branch_mask = (labels == 6)

        stage2_mask = woody_mask & (trunk_mask | branch_mask)

        stage2_labels = labels[stage2_mask]
        if len(stage2_labels) == 0:
            print(f"  [SKIP] No woody trunk/branch points: {las_name}")
            continue

        new_labels = np.where(stage2_labels == 3, 0, 1).astype(np.float64)

        filtered_points = las.points[stage2_mask].copy()
        new_las = laspy.LasData(header=las.header, points=filtered_points)
        new_las["label"] = new_labels

        new_las.write(out_path)

        trunk_n = np.sum(stage2_labels == 3)
        branch_n = np.sum(stage2_labels == 6)
        print(f"  [OK] {las_name}: {len(stage2_labels):,} pts (trunk={trunk_n:,}, branch={branch_n:,}) -> {out_path}")

    print(f"\nStage2 data prepared in {OUTPUT_ROOT}")
    print(f"Class mapping: trunk(3) -> 0, branch(6) -> 1")


if __name__ == "__main__":
    prepare_stage2_data()