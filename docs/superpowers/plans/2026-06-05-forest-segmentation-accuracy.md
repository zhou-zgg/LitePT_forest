# 森林点云分割精度提升 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 Loss 增强 + 后处理 + grid_size 调整提升森林点云分割 mIoU，减少碎片化分割

**Architecture:** Phase 1 在 LitePT config 层面增加 DiceLoss + label_smoothing 并 fine-tune；Phase 1b 用体素化连通域 + 向量化 KNN 做离线后处理并输出 LAS；Phase 2 调大 grid_size 扩大感受野。所有实验从 best 权重 fine-tune 30 epoch，后台运行。

**Tech Stack:** Python, PyTorch, laspy, numpy, scipy (ndimage.label, cKDTree)

**Spec:** `docs/superpowers/specs/2026-06-05-forest-segmentation-accuracy-design.md`

---

## File Structure

| 操作 | 文件 | 职责 |
|---|---|---|
| Create | `configs/forest/semseg-litept-small-v1m1-loss-v2.py` | Phase 1: Loss 增强实验 config（继承 base） |
| Create | `configs/forest/semseg-litept-small-v1m1-loss-v2-local.py` | Phase 1: 本地 16GB 覆盖配置（继承 loss-v2） |
| Create | `configs/forest/semseg-litept-small-v1m1-grid04.py` | Phase 2: grid_size 0.04 config（继承 loss-v2） |
| Create | `configs/forest/semseg-litept-small-v1m1-grid04-local.py` | Phase 2: 本地 16GB 覆盖配置 |
| Create | `scripts/run_loss_v2.sh` | Phase 1: 一键后台训练脚本 |
| Create | `scripts/run_grid04.sh` | Phase 2: 一键后台训练脚本 |
| Create | `tools/postprocess.py` | Phase 1b: 后处理（体素化碎片滤波 + 向量化 KNN + LAS 输出） |
| Modify | `FOREST_ADAPTATION.md` | 记录每次实验的 mIoU、loss、配置变更 |

---

## Task 1: 创建 Phase 1 Loss 增强实验 config

**Files:**
- Create: `configs/forest/semseg-litept-small-v1m1-loss-v2.py`
- Create: `configs/forest/semseg-litept-small-v1m1-loss-v2-local.py`

- [ ] **Step 1: 创建 loss-v2 基础 config（继承 base config）**

创建 `configs/forest/semseg-litept-small-v1m1-loss-v2.py`，只覆盖差异字段：

```python
_base_ = ["semseg-litept-small-v1m1.py"]

save_path = "exp/forest/semseg-litept-small-v1m1-loss-v2"

epoch = 30
eval_epoch = 5

model = dict(
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

- [ ] **Step 3: 验证 config 继承加载正确**

```bash
conda activate litept
export PYTHONPATH=./
python -c "
from utils.config import Config
cfg = Config.fromfile('configs/forest/semseg-litept-small-v1m1-loss-v2-local.py')
print('criteria:', [c['type'] for c in cfg.model['criteria']])
print('label_smoothing:', cfg.model['criteria'][0].get('label_smoothing'))
print('epoch:', cfg.epoch)
print('lr:', cfg.optimizer['lr'])
print('grid_size:', cfg.grid_size)
print('num_classes:', cfg.model['num_classes'])
"
```

Expected:
```
criteria: ['CrossEntropyLoss', 'LovaszLoss', 'DiceLoss']
label_smoothing: 0.1
epoch: 30
lr: 0.001
grid_size: 0.02
num_classes: 7
```

- [ ] **Step 4: Commit**

```bash
git add configs/forest/semseg-litept-small-v1m1-loss-v2.py configs/forest/semseg-litept-small-v1m1-loss-v2-local.py
git commit -m "feat: add loss-v2 config inheriting base (CE+smooth+Lovasz+Dice)"
```

---

## Task 2: 创建后台训练脚本

**Files:**
- Create: `scripts/run_loss_v2.sh`

- [ ] **Step 1: 创建训练脚本（conda 环境自动获取 Python）**

创建 `scripts/run_loss_v2.sh`：

```bash
#!/bin/bash
cd "$(dirname "$0")/.." || exit

CONDA_ENV="litept"
PYTHON=$(conda run -n "$CONDA_ENV" which python 2>/dev/null || echo "python")

EXP_NAME="semseg-litept-small-v1m1-loss-v2"
LOG_FILE="exp/forest/${EXP_NAME}/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "exp/forest/${EXP_NAME}"

echo "========== Phase 1: Loss-v2 实验 ==========" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "配置: CE(0.1 smooth) + Lovasz + Dice, lr=0.001, 30 epoch" | tee -a "$LOG_FILE"
echo "基线 mIoU: 0.5529" | tee -a "$LOG_FILE"
echo "Python: $PYTHON" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

export PYTHONPATH=./

nohup "$PYTHON" tools/train.py \
    --config-file configs/forest/semseg-litept-small-v1m1-loss-v2-local.py \
    --num-gpus 1 \
    --options save_path="exp/forest/${EXP_NAME}" \
    weight="exp/forest/semseg-litept-small-v1m1/model/model_best.pth" \
    >> "$LOG_FILE" 2>&1 &

echo "训练已在后台启动, PID: $!"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
```

- [ ] **Step 2: 启动训练并验证**

```bash
chmod +x scripts/run_loss_v2.sh
bash scripts/run_loss_v2.sh
sleep 30 && tail -30 exp/forest/semseg-litept-small-v1m1-loss-v2/train_*.log
```

Expected: 日志中有 "Start Training"，loss 数值在 1~3 范围

- [ ] **Step 3: Commit**

```bash
git add scripts/run_loss_v2.sh
git commit -m "feat: add background training script with auto conda python"
```

---

## Task 3: 创建后处理脚本

**Files:**
- Create: `tools/postprocess.py`

- [ ] **Step 1: 创建后处理脚本**

创建 `tools/postprocess.py`，包含：
- **体素化碎片滤波**：将点云体素化到 `voxel_size` 网格，用 `scipy.ndimage.label` 做 3D 连通域检测，小碎片用 KNN 投票修正
- **向量化 KNN 平滑**：用 numpy 广播 + bincount 代替逐点循环
- **LAS 输出**：`--save_las` 选项直接输出 smoothed LAS
- **修正统计输出**：每个场景打印 `(原标签 -> 新标签)` 分组计数

完整代码见 `tools/postprocess.py`（已创建）。

- [ ] **Step 2: 验证脚本可运行**

```bash
python tools/postprocess.py --help
```

Expected: 显示所有参数（包括 `--save_las`, `--voxel_size`）

- [ ] **Step 3: 用现有 baseline result 测试后处理**

```bash
PYTHONPATH=./ python tools/postprocess.py \
    --result_dir exp/forest/semseg-litept-small-v1m1/result \
    --data_root data/forest/val \
    --method both \
    --save_las
```

Expected: 每个场景输出修正统计 + 生成 smoothed npy 和 las 文件

- [ ] **Step 4: Commit**

```bash
git add tools/postprocess.py
git commit -m "feat: optimized postprocess (voxel fragment filter + vectorized KNN + LAS output)"
```

---

## Task 4: 记录 Phase 1 实验结果

**Files:**
- Modify: `FOREST_ADAPTATION.md`

- [ ] **Step 1: 等训练完成，从日志提取 mIoU 和 loss**

```bash
grep -E "mIoU|loss" exp/forest/semseg-litept-small-v1m1-loss-v2/train_*.log | tail -50
```

- [ ] **Step 2: 在 FOREST_ADAPTATION.md 末尾追加**

```markdown
### 实验 4: Loss-v2（CE+smooth+Lovasz+Dice, 30 epoch fine-tune）
- **训练时间**: 2026-06-05 ~
- **基线 mIoU**: 0.5529 (实验 1, epoch 51)
- **改动**:
  - criteria: CE(label_smoothing=0.1) + Lovasz + Dice (原 CE + Lovasz)
  - optimizer: lr 0.006 → 0.001, param_dicts lr 0.0006 → 0.0001
  - 从 best 权重 fine-tune, resume=False
- **Best mIoU**: (填入) (epoch XX)
- **各类 IoU**: terrain XX, foliage XX, CWD XX, trunk XX, branch XX, snag XX, non-tree XX
- **Loss 趋势**: (填入 epoch 5/10/15/20/25/30 的 loss 值)
- **结论**: (有效/无效，是否继续下一阶段)
```

- [ ] **Step 3: 记录后处理效果**

```markdown
### 后处理结果（基于实验 4 预测）
- 方法: 碎片滤波(voxel_size=0.1, min_points=50) + KNN平滑(k=20)
- (粘贴 postprocess.py 输出的修正统计)
```

- [ ] **Step 4: Commit**

```bash
git add FOREST_ADAPTATION.md
git commit -m "docs: record loss-v2 experiment results and postprocess stats"
```

---

## Task 5: 创建 Phase 2 grid_size 0.04 实验 config

**前提**: Phase 1 训练完成，已记录结果。

**Files:**
- Create: `configs/forest/semseg-litept-small-v1m1-grid04.py`
- Create: `configs/forest/semseg-litept-small-v1m1-grid04-local.py`

- [ ] **Step 1: 创建 grid04 基础 config（继承 loss-v2）**

创建 `configs/forest/semseg-litept-small-v1m1-grid04.py`：

```python
_base_ = ["semseg-litept-small-v1m1-loss-v2.py"]

grid_size = 0.04
save_path = "exp/forest/semseg-litept-small-v1m1-grid04"
```

仅覆盖 grid_size 和 save_path。从 Phase 1 best 权重 fine-tune。

- [ ] **Step 2: 创建本地覆盖 config**

创建 `configs/forest/semseg-litept-small-v1m1-grid04-local.py`：

```python
_base_ = ["semseg-litept-small-v1m1-grid04.py"]

data_root = "data/forest"
save_path = "exp/forest/semseg-litept-small-v1m1-grid04"
weight = "exp/forest/semseg-litept-small-v1m1-loss-v2/model/model_best.pth"
resume = False

batch_size = 1
crop_point_max = 150000
num_worker = 1

data = dict(
    block_xy=20,
    overlap=5,
)
```

- [ ] **Step 3: 验证 config**

```bash
python -c "
from utils.config import Config
cfg = Config.fromfile('configs/forest/semseg-litept-small-v1m1-grid04-local.py')
print('grid_size:', cfg.grid_size)
print('criteria:', [c['type'] for c in cfg.model['criteria']])
print('label_smoothing:', cfg.model['criteria'][0].get('label_smoothing'))
print('save_path:', cfg.save_path)
"
```

Expected: `grid_size: 0.04`, criteria 三项，label_smoothing 0.1

- [ ] **Step 4: Commit**

```bash
git add configs/forest/semseg-litept-small-v1m1-grid04.py configs/forest/semseg-litept-small-v1m1-grid04-local.py
git commit -m "feat: add grid04 config (grid_size 0.04 for larger receptive field)"
```

---

## Task 6: 创建 Phase 2 训练脚本并运行

**Files:**
- Create: `scripts/run_grid04.sh`

- [ ] **Step 1: 创建训练脚本**

创建 `scripts/run_grid04.sh`：

```bash
#!/bin/bash
cd "$(dirname "$0")/.." || exit

CONDA_ENV="litept"
PYTHON=$(conda run -n "$CONDA_ENV" which python 2>/dev/null || echo "python")

EXP_NAME="semseg-litept-small-v1m1-grid04"
LOG_FILE="exp/forest/${EXP_NAME}/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "exp/forest/${EXP_NAME}"

echo "========== Phase 2: grid_size=0.04 实验 ==========" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "配置: grid_size 0.02→0.04, 其余同 Phase 1" | tee -a "$LOG_FILE"
echo "Phase 1 best mIoU: (填入)" | tee -a "$LOG_FILE"
echo "Python: $PYTHON" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

export PYTHONPATH=./

nohup "$PYTHON" tools/train.py \
    --config-file configs/forest/semseg-litept-small-v1m1-grid04-local.py \
    --num-gpus 1 \
    --options save_path="exp/forest/${EXP_NAME}" \
    weight="exp/forest/semseg-litept-small-v1m1-loss-v2/model/model_best.pth" \
    >> "$LOG_FILE" 2>&1 &

echo "训练已在后台启动, PID: $!"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
```

- [ ] **Step 2: 启动训练**

```bash
chmod +x scripts/run_grid04.sh
bash scripts/run_grid04.sh
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_grid04.sh
git commit -m "feat: add grid04 background training script"
```

---

## Task 7: 记录 Phase 2 实验结果

**Files:**
- Modify: `FOREST_ADAPTATION.md`

- [ ] **Step 1: 等训练完成，记录结果**

```markdown
### 实验 5: grid_size 0.04（从 Phase 1 best fine-tune）
- **训练时间**: 2026-06-XX ~
- **Phase 1 best mIoU**: (填入)
- **改动**: grid_size 0.02 → 0.04
- **Best mIoU**: (填入) (epoch XX)
- **各类 IoU**: terrain XX, foliage XX, CWD XX, trunk XX, branch XX, snag XX, non-tree XX
- **Loss 趋势**: (填入)
- **结论**: (有效/无效)
```

- [ ] **Step 2: 对 Phase 2 结果运行后处理**

```bash
PYTHONPATH=./ python tools/postprocess.py \
    --result_dir exp/forest/semseg-litept-small-v1m1-grid04/result \
    --data_root data/forest/val \
    --method both \
    --save_las
```

- [ ] **Step 3: Commit**

```bash
git add FOREST_ADAPTATION.md
git commit -m "docs: record grid04 experiment results and postprocess stats"
```
