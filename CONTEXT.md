# Forest Point Cloud Segmentation — Domain Context

## Domain

Semantic segmentation of forest point clouds using deep learning (LitePT transformer).

## Classes (7 output slots, CWD ignored in training)

| ID | Name | Description |
|----|------|-------------|
| 0 | terrain | Ground surface |
| 1 | foliage | Leaves, canopy |
| 2 | CWD | Coarse woody debris. **Ignored**: `ignore_classes=[2]` maps labels to -1 in train and val; metrics skip it. Slot is kept in `num_classes`/names for downstream compatibility — never remove it |
| 3 | trunk | Tree trunks |
| 4 | branch | Branches (2-5cm, thin structures) |
| 5 | snag | Standing dead trees |
| 6 | NTC | Non-tree-cyl (utility poles). Most challenging class |
| 7 | — | Noise (`class_mapping={7:-1}`, inherited from base config) |

mIoU is averaged over valid classes only (GT count > 0), see `SemSegEvaluator`. With CWD ignored the denominator is 6, not 7 — old 7-class mIoU numbers are not directly comparable.

## Training Data

**Current dataset (forest_new)**: local `data/forest_new/` (server counterpart under `/root/autodl-tmp/...`). Old dataset archived at `data/forest/old/` + `old_dataset.tar.gz`.

**Train**: 90 .las files — 14 pole files (contain NTC), 12 mountain tiles (steep terrain), rest forest scans. CWD ≈ 0 in this dataset.

**Val**: 13 .las files — 6 dated scans (20240618/20260326/20260401...), 4 mountain (`moutain-cz*.las`, note the filename typo), 3 pole (`pole.las`, `pole1.las`, `pole2.las`).

**Format**: .las files with labels in `label` extra dimension (not `classification` field). Read coordinates via `las.header.scales` (not hardcoded 0.001).

## Data Preprocessing Pipeline (current, config chain clean.py → local16g)

```
.las file
  → CenterShiftCUDA (shift to origin, apply_z=True)
  → RandomDropoutCUDA (20% dropout, 20% probability)
  → RandomRotateCUDA (z-axis, ±1°, 50% probability)
  → RandomRotateCUDA (x-axis, +0.12 rad, 50%)   ← slope augmentation
  → RandomRotateCUDA (y-axis, +0.12 rad, 50%)   ← slope augmentation
  → RandomFlipCUDA (50% probability)
  → RandomJitterCUDA (sigma=0.005, clip=0.02)
  → GridSampleCUDA (voxelize at 0.02m grid)
  → CylinderCropCUDA (350K points on local 16G; 1M on server)
  → CenterShiftCUDA (shift to XY center, apply_z=False)
  → ToTensorCUDA
  → CollectCUDA
```

**Validation**: block-based cylindrical cropping (`val_crop_point_max=300000`), grid 0.02 matching training; softmax probabilities accumulated across blocks. Evaluator calls `torch.cuda.empty_cache()` **before** evaluation to avoid OOM during worker init.

**16G local constraint**: `num_worker=0` (CUDA transforms + workers leak GPU memory).

## NTC Problem & Solution

**Problem**: Utility poles (NTC, class 6) are densely scanned thin cylinders. At 0.02m grid:
- 1.32M raw NTC points → 14K voxels (99% loss)
- Only 14/90 training files contain NTC (~15% of batches)
- Result: model NEVER predicted class 6 (V21)

**Solutions tried**:
1. GridSample/CylinderCrop class_weight sorting — **ineffective** (NTC voxels are 100% NTC, sorting doesn't change selection)
2. Weighted CrossEntropyLoss — **rejected** (over-penalized, hurt snag and trunk)
3. File-level oversampling via WeightedRandomSampler — **effective** ✅

**Current solution (oversampling)**: adaptive per-file weights from JSON-cached class histograms; pole files sampled ~4x more frequently.

## Evaluation Grouping

`SemSegEvaluator` groups per-file confusion matrices and prints **mountain vs other** separately (mountain has stricter terrain annotation → lower terrain IoU is expected). Grouping matches both `moutain` (typo in filenames) and `mountain` spellings, see `_acc_group` in `engines/hooks/evaluator.py`.

## Current Best Model

**L1-local (c350-slope)**: LitePT-Small, trained locally on 16G GPU.
- Config: `configs/forest/semseg-litept-small-v1m1-loss-v2-clean-local16g.py`
- grid_size 0.02, crop 350K, slope aug ±0.12 rad, OneCycleLR 60 epochs, warm start from c350 best (`resume=False`)
- **mIoU 0.7582** (best@ep30; mAcc 0.8237, allAcc 0.8915)
- Per-class IoU: terrain 0.741, foliage 0.853, trunk 0.797, branch 0.571, snag 0.711, NTC 0.877 (CWD ignored)
- Mountain vs other: terrain 0.52 vs 0.83, trunk 0.78 vs 0.84, branch 0.55 vs 0.63
- Weight: `exp/forest/semseg-litept-small-v1m1-clean-local16g-c350-slope/model/model_best.pth` (local)

**Local series (forest_new)**: 0.5979 (crop 100K) → 0.6569 (crop 350K) → 0.7582 (crop 350K + slope aug). Recipe: warm start + short OneCycle beats scratch + long OneCycle.

**Server best (old dataset)**: V27 Base, mIoU 0.6850. V28 (scratch, 1M crop, forest_new) abandoned at 0.6261 — recurrent OOM on the shared 48G GPU.

## Server Connection

```bash
ssh serve
# Target: root@connect.westd.seetacloud.com -p 39631
# Repo: /root/autodl-tmp/workshop/caozhou/LitePT_forest
```

Server is source of truth for the old-dataset experiments; local 16G machine now trains Small-model runs on forest_new.
