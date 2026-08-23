# LitePT — Forest Point Cloud Segmentation

## Project

LitePT-based semantic segmentation for forest point clouds. 7 classes: terrain(0), foliage(1), CWD(2), trunk(3), branch(4), snag(5), non-tree-cyl/NTC(6).

**All training and inference runs on the server. Local is for code editing only.**

## Server Connection

```bash
# Local ~/.ssh/config:
# Host serve
#     HostName connect.westd.seetacloud.com
#     Port 39631
#     User root

ssh serve
cd /root/autodl-tmp/workshop/caozhou/LitePT_forest

# Sync files to server
scp -P 39631 /local/path root@connect.westd.seetacloud.com:/root/autodl-tmp/...
```

```bash
# Local: edit code, commit, sync to server
# Server: train, infer

# Server setup (once)
conda activate forest
export PYTHONPATH=./
cd /root/autodl-tmp/workshop/caozhou/LitePT_forest

# CUDA extensions (once, match GPU arch in setup.py)
cd libs/pointrope && python setup.py install && cd ../..
cd libs/pointops && python setup.py install && cd ../..
```

## Training

```bash
# On server
PYTHONPATH=. python3 tools/train.py \
  --config-file configs/forest/semseg-litept-base-v1m1-loss-v2-clean-vXX.py \
  --num-gpus 1
```

## Inference

```bash
# Single file with block_inference.py
PYTHONPATH=. python3 tools/block_inference.py \
  --input /path/to/file.pcd \
  --output /path/to/output.las \
  --weight /path/to/model_best.pth \
  --grid-size 0.025

# Batch inference
PYTHONPATH=. python3 tools/batch_infer_v24.py
```

## Version History

| Ver | Model | mIoU | Key Change |
|-----|-------|------|------------|
| V17 | Small, 1M crop, grid 0.02 | **0.6166** | Best small model; terrain=0.8525, NTC=0.7089 |
| V18-V20 | Base, various | <0.58 | Early base experiments, NTC=0 |
| V21 | Base, class_weight attempt | 0.5277 | NTC=0 (GridSample decimated poles) |
| V22 | Base, **oversampling** | 0.6007 | NTC=0.7363; WeightedRandomSampler on pole files |
| V23 | Base, **grid 0.025 + patch 2048** | 0.6329 | terrain=0.799; 56% more spatial coverage |
| V24 | Base, resume V23 best + OneCycleLR | **0.6635** | Previous best (superseded by V27); trunk=0.866, snag=0.768 |
| V25/V26 | Base, weighted CE | 0.6320 | Failed; weighted loss hurt snag (-0.21) |
| V27 | Base, resume V24 best + continue | **0.6850** | New best base; same config as V24 (train grid 0.02 / infer grid 0.025, CE+Lovasz+Dice, OneCycleLR) |
| L1-local | Small, **crop 350k + slope aug ±0.12** | **0.7582** | Local 16G best; resume best + slope aug, 60ep, best@ep30; new forest_new dataset (CWD ignored). trunk=0.797, snag=0.711, NTC=0.877 |

## Key Code

| File | Purpose |
|------|---------|
| `datasets/forest.py` | ForestDataset, oversampling, JSON class histogram cache |
| `datasets/transform_gpu.py` | GridSampleCUDA, CylinderCropCUDA, augmentations |
| `engines/train.py` | Trainer, WeightedRandomSampler hook |
| `models/losses/misc.py` | CrossEntropyLoss (supports `weight` param) |
| `tools/block_inference.py` | 3-step block inference: split → infer → merge LAS |
| `tools/batch_infer_v24.py` | V24 batch inference (Base model, grid 0.025) |
| `tools/batch_infer_v17.py` | V17 batch inference (Small model, grid 0.02) |
| `configs/forest/` | V17-V27 experiment configs |

## Critical Notes

- **Server is source of truth** — local code is behind. All commits are on server.
- **grid_size must match** training and inference (0.02 for V17, 0.025 for V22+)
- **enc_patch_size must match** model config (Small: all 1024 in top stages; Base V22: 1024; Base V23+: 2048 in stages 3-4)
- **NTC root cause fixed**: GridSample at 0.02m decimates pole points 99%; oversampling compensates
- **Weighted CE rejected**: amplifies NTC at cost of snag/trunk; oversampling alone is sufficient
- **CWD (class 2) is NOT a learning target** — user does not care about CWD accuracy. Labels are mapped to ignore_index(-1) in training and val (`ignore_classes=[2]` in config, handled by `ForestDataset`). The model still outputs 7 classes: the CWD slot MUST be kept for inference compatibility with downstream projects. Never remove the class from num_classes/names. mIoU is averaged over valid classes only (GT count > 0), see `SemSegEvaluator`.
