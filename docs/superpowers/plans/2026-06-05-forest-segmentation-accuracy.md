# 森林点云分割精度提升 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 Loss 增强 + 后处理提升森林点云分割 mIoU，减少碎片化分割

**Architecture:** 在 LitePT 现有 pipeline 上，Phase 1 修改 config 增加 DiceLoss + label_smoothing 并 fine-tune；Phase 1b 新建独立后处理脚本做连通域滤波和 KNN 平滑。所有实验从 best 权重 fine-tune 30 epoch，后台运行。

**Tech Stack:** Python, PyTorch, laspy, numpy, scipy (连通域)

**Spec:** `docs/superpowers/specs/2026-06-05-forest-segmentation-accuracy-design.md`

---

## File Structure

| 操作 | 文件 | 职责 |
|---|---|---|
| Create | `configs/forest/semseg-litept-small-v1m1-loss-v2.py` | Phase 1: Loss 增强实验 config |
| Create | `configs/forest/semseg-litept-small-v1m1-loss-v2-local.py` | Phase 1: 本地 16GB 覆盖配置 |
| Create | `scripts/run_loss_v2.sh` | Phase 1: 一键后台训练脚本 |
| Create | `tools/postprocess.py` | Phase 1b: 推理后处理（碎片滤波 + KNN 平滑） |
| Modify | `FOREST_ADAPTATION.md` | 记录每次实验的 mIoU 结果和配置变更 |

---

## Task 1: 创建 Phase 1 Loss 增强实验 config

**Files:**
- Create: `configs/forest/semseg-litept-small-v1m1-loss-v2.py`
- Create: `configs/forest/semseg-litept-small-v1m1-loss-v2-local.py`

- [ ] **Step 1: 创建 loss-v2 基础 config**

创建 `configs/forest/semseg-litept-small-v1m1-loss-v2.py`，继承 base config 并修改 criteria、optimizer、scheduler：

```python
_base_ = ["../_base_/default_runtime.py"]

use_gpu_transform = True
crop_mode = "cylinder"
crop_type = (crop_mode.capitalize() + "CropCUDA") if use_gpu_transform else (crop_mode.capitalize() + "Crop")
crop_point_max = 500000
grid_size = 0.02
class_mapping = {7: -1}

enable_scale = False
scale_range = [0.9, 1.1]
resume = False

epoch = 30
eval_epoch = 5
save_path = "exp/forest/semseg-litept-small-v1m1-loss-v2"

enable_wandb = False
batch_size = 4
num_worker = 20
mix_prob = 0.8
empty_cache = False
enable_amp = True
clip_grad = 1.0

model = dict(
    type="DefaultSegmentorV2",
    num_classes=7,
    backbone_out_channels=72,
    backbone=dict(
        type="LitePT",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
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

dataset_type = "ForestDataset"
data_root = "data/forest"

data = dict(
    num_classes=7,
    ignore_index=-1,
    names=[
        "terrain",
        "foliage",
        "CWD",
        "trunk",
        "branch",
        "snag",
        "non-tree-cyl",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_mapping=class_mapping,
        transform=[
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=True),
            dict(
                type="RandomDropoutCUDA" if use_gpu_transform else "RandomDropout",
                dropout_ratio=0.2, dropout_application_ratio=0.2,
            ),
            dict(type="RandomRotateCUDA" if use_gpu_transform else "RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotateCUDA" if use_gpu_transform else "RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotateCUDA" if use_gpu_transform else "RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
        ]
        + ([dict(type="RandomScaleCUDA" if use_gpu_transform else "RandomScale", scale=scale_range)] if enable_scale else [])
        + [
            dict(type="RandomFlipCUDA" if use_gpu_transform else "RandomFlip", p=0.5),
            dict(type="RandomJitterCUDA" if use_gpu_transform else "RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSampleCUDA" if use_gpu_transform else "GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ElasticDistortionCUDA" if use_gpu_transform else "ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type=crop_type, point_max=crop_point_max, mode="random"),
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=False),
            dict(type="ToTensorCUDA" if use_gpu_transform else "ToTensor"),
            dict(type="UpdateCUDA" if use_gpu_transform else "Update", keys_dict={"grid_size": grid_size}),
            dict(
                type="CollectCUDA" if use_gpu_transform else "Collect",
                keys=("coord", "grid_coord", "segment", "grid_size"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=True),
            dict(type="CopyCUDA" if use_gpu_transform else "Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSampleCUDA" if use_gpu_transform else "GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=False),
            dict(type="ToTensorCUDA" if use_gpu_transform else "ToTensor"),
            dict(
                type="CollectCUDA" if use_gpu_transform else "Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSampleCUDA" if use_gpu_transform else "GridSample",
                grid_size=0.2,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShiftCUDA" if use_gpu_transform else "CenterShift", apply_z=False),
                dict(type="ToTensorCUDA" if use_gpu_transform else "ToTensor"),
                dict(
                    type="CollectCUDA" if use_gpu_transform else "Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord",),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngleCUDA" if use_gpu_transform else "RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=3),
    dict(type="PreciseEvaluator", test_last=False),
]
```

- [ ] **Step 2: 创建本地覆盖 config**

创建 `configs/forest/semseg-litept-small-v1m1-loss-v2-local.py`：

```python
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
```

- [ ] **Step 3: 验证 config 可被正确加载**

```bash
conda activate litept
export PYTHONPATH=./
python -c "from configs.forest.semseg_litept_small_v1m1_loss_v2_local import *; print('criteria:', [c['type'] for c in model['criteria']]); print('epoch:', epoch); print('lr:', optimizer['lr'])"
```

Expected: `criteria: ['CrossEntropyLoss', 'LovaszLoss', 'DiceLoss']`, `epoch: 30`, `lr: 0.001`

- [ ] **Step 4: Commit**

```bash
git add configs/forest/semseg-litept-small-v1m1-loss-v2.py configs/forest/semseg-litept-small-v1m1-loss-v2-local.py
git commit -m "feat: add loss-v2 config (CE+label_smooth+Lovasz+Dice, fine-tune from best)"
```

---

## Task 2: 创建一键后台训练脚本

**Files:**
- Create: `scripts/run_loss_v2.sh`

- [ ] **Step 1: 创建训练脚本**

创建 `scripts/run_loss_v2.sh`：

```bash
#!/bin/bash
# Phase 1: Loss 增强实验 (CE + label_smoothing + Lovasz + Dice)
# 从 best 权重 fine-tune 30 epoch

cd "$(dirname "$0")/.." || exit

EXP_NAME="semseg-litept-small-v1m1-loss-v2"
LOG_FILE="exp/forest/${EXP_NAME}/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "exp/forest/${EXP_NAME}"

echo "========== Phase 1: Loss-v2 实验 ==========" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "配置: CE(0.1 smooth) + Lovasz + Dice, lr=0.001, 30 epoch" | tee -a "$LOG_FILE"
echo "基线 mIoU: 0.5529" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

conda activate litept
export PYTHONPATH=./

nohup python tools/train.py \
    --config-file configs/forest/semseg-litept-small-v1m1-loss-v2-local.py \
    --num-gpus 1 \
    --options save_path="exp/forest/${EXP_NAME}" \
    weight="exp/forest/semseg-litept-small-v1m1/model/model_best.pth" \
    >> "$LOG_FILE" 2>&1 &

echo "训练已在后台启动, PID: $!"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
```

- [ ] **Step 2: 验证脚本可执行**

```bash
chmod +x scripts/run_loss_v2.sh
bash -n scripts/run_loss_v2.sh
```

Expected: 无语法错误

- [ ] **Step 3: 启动训练**

```bash
bash scripts/run_loss_v2.sh
```

Expected: 输出 PID 和日志路径，训练在后台运行

- [ ] **Step 4: 验证训练已启动**

```bash
ps aux | grep train.py | grep -v grep
tail -20 exp/forest/semseg-litept-small-v1m1-loss-v2/train_*.log
```

Expected: 看到 train.py 进程，日志中有 "Start Training" 字样

- [ ] **Step 5: Commit**

```bash
git add scripts/run_loss_v2.sh
git commit -m "feat: add background training script for loss-v2 experiment"
```

---

## Task 3: 创建后处理脚本

**Files:**
- Create: `tools/postprocess.py`

- [ ] **Step 1: 创建后处理脚本**

创建 `tools/postprocess.py`，包含碎片滤波和 KNN 平滑两种方法，带修正统计输出：

```python
import os
import argparse
import numpy as np
import laspy
from collections import Counter
from scipy.spatial import cKDTree


def knn_smooth(coord, pred, labels, k=20):
    smoothed = pred.copy()
    tree = cKDTree(coord)
    total_changed = 0
    change_detail = Counter()

    _, indices = tree.query(coord, k=k + 1)
    for i in range(len(pred)):
        neighbor_labels = pred[indices[i, 1:]]
        votes = Counter(neighbor_labels.tolist())
        winner = votes.most_common(1)[0][0]
        if winner != pred[i]:
            old_label = labels[pred[i]] if pred[i] < len(labels) else str(pred[i])
            new_label = labels[winner] if winner < len(labels) else str(winner)
            change_detail[f"{old_label} -> {new_label}"] += 1
            smoothed[i] = winner
            total_changed += 1

    return smoothed, total_changed, change_detail


def fragment_filter(coord, pred, labels, min_points=50, knn_k=20):
    filtered = pred.copy()
    tree = cKDTree(coord)
    total_changed = 0
    change_detail = Counter()

    unique_classes = np.unique(pred)
    for cls in unique_classes:
        mask = pred == cls
        cls_indices = np.where(mask)[0]
        if len(cls_indices) < min_points:
            continue

        cls_coord = coord[cls_indices]
        cls_tree = cKDTree(cls_coord)
        visited = set()
        components = []

        for i in range(len(cls_coord)):
            if i in visited:
                continue
            queue = [i]
            component = []
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                neighbors = cls_tree.query_ball_point(cls_coord[node], r=0.1)
                for n in neighbors:
                    if n not in visited:
                        queue.append(n)
            components.append(component)

        for comp in components:
            if len(comp) >= min_points:
                continue
            global_indices = cls_indices[comp]
            _, neighbor_indices = tree.query(coord[global_indices], k=knn_k)
            for gi, ni in zip(global_indices, neighbor_indices):
                neighbor_labels = pred[ni]
                votes = Counter(neighbor_labels.tolist())
                winner = votes.most_common(1)[0][0]
                if winner != pred[gi]:
                    old_label = labels[pred[gi]] if pred[gi] < len(labels) else str(pred[gi])
                    new_label = labels[winner] if winner < len(labels) else str(winner)
                    change_detail[f"{old_label} -> {new_label}"] += 1
                    filtered[gi] = winner
                    total_changed += 1

    return filtered, total_changed, change_detail


def print_stats(name, total, changed, change_detail):
    pct = changed / total * 100 if total > 0 else 0
    print(f"[{name}] 修正统计: {changed}/{total} 点被修改 ({pct:.3f}%)")
    if change_detail:
        for transition, count in change_detail.most_common():
            print(f"  {transition}: {count}")


def main():
    parser = argparse.ArgumentParser(description="点云分割后处理")
    parser.add_argument("--result_dir", required=True, help="预测结果目录 (包含 *_pred.npy)")
    parser.add_argument("--data_root", required=True, help="原始 LAS 文件目录")
    parser.add_argument("--method", choices=["knn_smooth", "fragment_filter", "both"], default="both")
    parser.add_argument("--min_points", type=int, default=50)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--output_suffix", default="_smoothed", help="输出文件后缀")
    args = parser.parse_args()

    labels = ["terrain", "foliage", "CWD", "trunk", "branch", "snag", "non-tree-cyl"]

    npy_files = sorted([f for f in os.listdir(args.result_dir) if f.endswith("_pred.npy")])

    for pred_name in npy_files:
        base_name = pred_name.replace("_pred.npy", "")
        coord_path = os.path.join(args.data_root, base_name)
        pred_path = os.path.join(args.result_dir, pred_name)

        if not os.path.exists(coord_path):
            print(f"[{base_name}] 跳过: 找不到原始文件 {coord_path}")
            continue

        las = laspy.read(coord_path)
        coord = np.vstack([las.X, las.Y, las.Z]).T.astype(np.float32) * 0.001
        pred = np.load(pred_path)

        total_points = len(pred)
        print(f"\n{'='*60}")
        print(f"[{base_name}] 总点数: {total_points}, 方法: {args.method}")

        result = pred.copy()

        if args.method in ("fragment_filter", "both"):
            result, changed, detail = fragment_filter(
                coord, result, labels, min_points=args.min_points, knn_k=args.knn_k
            )
            print_stats(base_name + " (碎片滤波)", total_points, changed, detail)

        if args.method in ("knn_smooth", "both"):
            result, changed, detail = knn_smooth(
                coord, result, labels, k=args.knn_k
            )
            print_stats(base_name + " (KNN平滑)", total_points, changed, detail)

        out_name = pred_name.replace("_pred.npy", f"_pred{args.output_suffix}.npy")
        out_path = os.path.join(args.result_dir, out_name)
        np.save(out_path, result)
        print(f"  -> 已保存: {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本可运行（dry run 检查语法）**

```bash
conda activate litept
python -c "import tools.postprocess" 2>&1 || python tools/postprocess.py --help
```

Expected: 显示 argparse help 信息，无 import 错误

- [ ] **Step 3: 等训练完成后运行后处理**

训练完成后（或用现有 result 测试）：

```bash
PYTHONPATH=./ python tools/postprocess.py \
    --result_dir exp/forest/semseg-litept-small-v1m1-loss-v2/result \
    --data_root data/forest/val \
    --method both \
    --min_points 50 \
    --knn_k 20
```

Expected: 每个场景输出修正统计，按 `(原标签 -> 新标签)` 分组计数

- [ ] **Step 4: Commit**

```bash
git add tools/postprocess.py
git commit -m "feat: add postprocess script (fragment filter + KNN smooth with stats)"
```

---

## Task 4: 记录实验结果

**Files:**
- Modify: `FOREST_ADAPTATION.md`

- [ ] **Step 1: 在 FOREST_ADAPTATION.md 末尾追加 Phase 1 实验记录**

在训练完成后，从日志中提取 mIoU，追加记录：

```markdown
### 实验 4: Loss-v2（CE+smooth+Lovasz+Dice, 30 epoch fine-tune）
- **训练时间**: 2026-06-05 ~
- **基线 mIoU**: 0.5529 (实验 1, epoch 51)
- **改动**:
  - criteria: CE(label_smoothing=0.1) + Lovasz + Dice (原 CE + Lovasz)
  - optimizer: lr 0.006 → 0.001, param_dicts lr 0.0006 → 0.0001
  - 从 best 权重 fine-tune, resume=False
- **Best mIoU**: (填入结果) (epoch XX)
- **各类 IoU**: terrain XX, foliage XX, CWD XX, trunk XX, branch XX, snag XX, non-tree XX
- **结论**: (有效/无效，是否继续下一阶段)
```

- [ ] **Step 2: 记录后处理效果**

```markdown
### 后处理结果（基于实验 4 预测）
- 方法: 碎片滤波(min_points=50) + KNN平滑(k=20)
- (粘贴 postprocess.py 输出的修正统计)
```

- [ ] **Step 3: Commit**

```bash
git add FOREST_ADAPTATION.md
git commit -m "docs: record loss-v2 experiment results"
```
