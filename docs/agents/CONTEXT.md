# Forest Point Cloud Segmentation — Domain Context

## Domain

Semantic segmentation of forest point clouds using deep learning (LitePT transformer).

## Classes (7 + ignored)

| ID | Name | Description |
|----|------|-------------|
| 0 | terrain | Ground surface |
| 1 | foliage | Leaves, canopy |
| 2 | CWD | Coarse woody debris. Present in data, accuracy not tracked (too few points) |
| 3 | trunk | Tree trunks |
| 4 | branch | Branches (2-5cm, thin structures) |
| 5 | snag | Standing dead trees |
| 6 | NTC | Non-tree-cyl (utility poles). Most challenging class |
| 7 | — | Noise (class_mapping={7:-1}, ignored) |

## Training Data

**Root**: `/root/autodl-tmp/dataset/tree/forest/`

**Train**: 91 files split into:
- **Pole files (14)**: pole_05_1_2.las through pole_05_16.las, pole3.las. Contain NTC labels. Key for rare class learning.
- **Mountain files (15)**: mountain-cz, mountain-lz, mountain-yxm, mountain-zzr (each split into multiple tiles). Steep terrain scenes.
- **Forest files (62)**: Standard forest scans. 58 contain snag labels.

**Val**: 11 files (pole.las, pole1.las, pole2.las, snag/forest scans, mountain-zzr_1.las, mountain-cz_1.las)

**Format**: .las files with labels in `label` extra dimension (not `classification` field).

## Data Preprocessing Pipeline

```
.las file
  → CenterShiftCUDA (shift to origin, apply_z=True)
  → RandomDropoutCUDA (20% dropout, 20% probability)
  → RandomRotateCUDA (z-axis, ±1°, 50% probability)
  → RandomFlipCUDA (50% probability)
  → RandomJitterCUDA (sigma=0.005, clip=0.02)
  → GridSampleCUDA (voxelize at 0.025m grid)
  → CylinderCropCUDA (take 450K nearest points in XY plane)
  → CenterShiftCUDA (shift to XY center, apply_z=False)
  → ToTensorCUDA
  → CollectCUDA
```

## NTC Problem & Solution

**Problem**: Utility poles (NTC, class 6) are densely scanned thin cylinders. At 0.02m grid:
- 1.32M raw NTC points → 14K voxels (99% loss)
- NTC occupies 2.1% of voxels post-GridSample
- Only 14/91 training files contain NTC (15% of batches)
- Effective NTC training signal: ~0.3% of total gradient
- Result: model NEVER predicted class 6 (V21)

**Solutions tried**:
1. GridSample/CylinderCrop class_weight sorting — **ineffective** (NTC voxels are 100% NTC, sorting doesn't change selection)
2. Weighted CrossEntropyLoss — **rejected** (over-penalized, hurt SNG and trunk)
3. File-level oversampling via WeightedRandomSampler — **effective** ✅

**Current solution (oversampling)**:
- Adaptive per-file weights based on class rarity across files
- Pole files sampled 4x more frequently (42.5% of batches vs 15%)

## Current Best Model

**V24**: LitePT-Base (45M params)
- grid_size: 0.025m
- enc_patch_size: (1024, 1024, 1024, 2048, 2048)
- enc_channels: (54, 108, 216, 432, 576)
- Crop: 450K points
- mIoU: 0.6635
- Weight: `/root/autodl-tmp/exp/forest/semseg-litept-base-v1m1-loss-v2-clean-v24/model/model_best.pth`

## Server Connection

```bash
ssh serve
# Target: root@connect.westd.seetacloud.com -p 39631
```
