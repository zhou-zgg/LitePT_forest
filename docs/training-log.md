# 森林点云语义分割训练记录

**数据**: `/root/autodl-tmp/dataset/tree/forest/`, .las 格式
**GPU**: RTX 3090 (49GB), RTX 4080 (16GB, 本地)
**标签**: 7类 (GND=0, FOL=1, CWD=2, TRK=3, BRC=4, SNG=5, NTC=6), class_mapping {7: -1} 忽略噪声

---

## 实验总览

| 版本 | 模型 | 数据 | crop | 权重来源 | scheduler | best mIoU | 状态 |
|---|---|---|---|---|---|---|---|
| V8 | S | 77+9 | 800K | scratch | OneCycle | 0.6056 | 完成 |
| V9 | S | 77+9 | 800K | V8 best | OneCycle | 0.6097 | 完成 |
| V10 | S | 77+9 | 800K | V9 best | OneCycle, lr=5e-4 | **0.6234** | 完成 |
| V11 | S | 77+9 | 1M | pretrained | OneCycle | 0.6225 | 完成 |
| V12 | S | 77+9 | ? | pretrained | OneCycle | — | OOM 无 eval |
| V13 | S | 77+9 | 1M | V11 best | OneCycle, loop=30 | 0.5311 | OOM |
| V14 | S | 80+9 | 1M | V11 best | OneCycle, loop=30 | 0.5368 | OOM |
| V15 | S | 87+10 | 500K | V11 best | OneCycle, loop=30 | — | 配置错误停 |
| V16 | S | 87+10 | 1M | scratch | OneCycle | 0.5370 | OOM |
| V17 | S | 87+10 | 1M | V16 best | CosineAnnealing | **0.6166** | 完成 |
| V18 | B(自制) | 87+10 | 350K | scratch | CosineAnnealing | 0.5143 | 完成 |
| V19 | B(官方) | 87+10 | 350K | scratch | CosineAnnealing | — | 进行中 |

---

## V8

| 项目 | 值 |
|---|---|
| 配置 | 已丢失 (wd config) |
| 模型 | LitePT-S (12.7M) |
| 数据 | 77 train + 9 val |
| scheduler | OneCycleLR |
| epoch | 100, eval_epoch=10 |
| crop_point_max | 800,000 |
| 训练方式 | from scratch (weight="") |
| best mIoU | **0.6056** |

---

## V9

| 项目 | 值 |
|---|---|
| 配置 | 已丢失 |
| 模型 | LitePT-S (12.7M) |
| 数据 | 77 train + 9 val, train_loop=6 |
| weight | V8 best (0.6056) |
| scheduler | OneCycleLR |
| epoch | 30, eval_epoch=5 |
| crop_point_max | 800,000 |
| best mIoU | **0.6097** |
| 最终 mIoU | 0.5578 (过拟合下降) |

---

## V10

| 项目 | 值 |
|---|---|
| 配置 | 已丢失 |
| 模型 | LitePT-S (12.7M) |
| 数据 | 77 train + 9 val, train_loop=8 |
| weight | V9 best (0.6097) |
| scheduler | OneCycleLR, **lr=0.0005** (降学习率) |
| epoch | 40, eval_epoch=5 |
| crop_point_max | 800,000 |
| best mIoU | **0.6234** |

---

## V11

| 项目 | 值 |
|---|---|
| 配置 | 已丢失 |
| 模型 | LitePT-S (12.7M) |
| 数据 | 77 train + 9 val, train_loop=6 |
| weight | pretrained |
| scheduler | OneCycleLR, lr=0.001 |
| epoch | 30, eval_epoch=5 |
| crop_point_max | 1,000,000 |
| best mIoU | **0.6225** |
| 最终 mIoU | 0.6225 |

---

## V12

| 项目 | 值 |
|---|---|
| 配置 | 已丢失 |
| 模型 | LitePT-S (12.7M) |
| 数据 | 77 train + 9 val, train_loop=6 |
| weight | pretrained |
| best mIoU | **无 eval** (OOM/crash 在前几个 epoch) |

---

## V13

| 项目 | 值 |
|---|---|
| 配置 | `semseg-litept-small-v1m1-loss-v2-clean-v13.py` (已丢失) |
| 模型 | LitePT-S (12.7M) |
| 数据 | 77 train + 9 val, train_loop=30 (eval_epoch=1 导致) |
| weight | V11 best (0.6225) |
| scheduler | OneCycleLR |
| epoch | 30, eval_epoch=1 |
| crop_point_max | 1,000,000 |
| best mIoU | **0.5311** (epoch 19, OOM) |
| 问题 | train_loop=30 → 2310 iter/epoch，eval_epoch=1 引爆 |

---

## V14

| 项目 | 值 |
|---|---|
| 配置 | `semseg-litept-small-v1m1-loss-v2-clean-v14.py` (已丢失) |
| 模型 | LitePT-S (12.7M) |
| 数据 | 80 train + 9 val, train_loop=30 (eval_epoch=1 导致) |
| weight | V11 best (0.6225) |
| 坡度增强 | **开启** (x/y rotate) |
| crop_point_max | 1,000,000 |
| best mIoU | **0.5368** (epoch 11, OOM at epoch 12) |
| 问题 | 坡度增强 + eval_epoch=1 双重打击 |

---

## V15

| 项目 | 值 |
|---|---|
| 配置 | `semseg-litept-small-v1m1-loss-v2-clean-v15.py` |
| 模型 | LitePT-S (12.7M) |
| 数据 | 87 train + 10 val, train_loop=30 (eval_epoch=1 导致) |
| weight | V11 best (0.6225) |
| epoch | 30, eval_epoch=1 |
| crop_point_max | 500,000 (实际运行值) |
| num_worker | 1 |
| empty_cache_per_epoch | False |
| best mIoU | **无** (配置错误 + 无顶层 crop_point_max=1M，实际 500K) |

---

## V16 (从 scratch，无坡度增强)

| 项目 | 值 |
|---|---|
| 配置 | `semseg-litept-small-v1m1-loss-v2-clean-v16.py` |
| 模型 | LitePT-S (12.7M) |
| 数据 | 87 train (含 12 山地) |
| scheduler | OneCycleLR (pct_start=0.05) |
| epoch | 50, eval_epoch=5 |
| crop_point_max | 1,000,000 |
| num_worker | 1 |
| empty_cache | True |
| 坡度增强 | **关闭** |
| 训练方式 | from scratch (weight=None) |
| best mIoU | **0.5370** (epoch 4) |
| epoch 7 | 0.4461 (OOM 后死掉) |

**问题**: OneCycleLR 在 epoch 4-7 LR ~0.001，导致过拟合振荡；empty_cache 在 eval 之后才执行。

---

## V17 (V16 best 权重 + CosineAnnealingLR + OOM 修复)

| 项目 | 值 |
|---|---|
| 配置 | `semseg-litept-small-v1m1-loss-v2-clean-v17.py` |
| 模型 | LitePT-S (12.7M) |
| weight | V16 best (0.5370) |
| scheduler | **CosineAnnealingLR** (eta_min=1e-6) |
| epoch | 50, eval_epoch=5 |
| crop_point_max | 1,000,000 |
| num_worker | 0 (修复 OOM) |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True |
| empty_cache 顺序修复 | eval 前先清缓存 |
| 坡度增强 | **关闭** |
| best mIoU | **0.6166** (epoch 19) |
| epoch 38 | 0.6159 (plateau，手动停) |

**改动 vs V16**:
- OneCycleLR → CosineAnnealingLR (平滑衰减，避免高 LR 振荡)
- num_worker 1→0 (消除 DataLoader worker 2+GB 显存占用)
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
- 修改 engines/train.py after_epoch 空 empty_cache 在 eval 前

**类 best (epoch 19)**:
| Class | IoU |
|---|---|
| terrain | 0.8525 |
| foliage | 0.9017 |
| CWD | 0.0000 |
| trunk | 0.7709 |
| branch | 0.5827 |
| snag | 0.4991 |
| NTC | 0.7089 |

---

## V18 (LitePT-B, 自制参数, 400K→300K→350K)

| 项目 | 值 |
|---|---|
| 配置 | `semseg-litept-base-v1m1-loss-v2-clean-v18.py` |
| 模型 | LitePT-B (45M) |
| enc_patch_size | (1024,1024,1024,**2048,2048**) ← 非官方，从 V17 loss-v2 继承 |
| enc_conv/enc_attn | **缺失** ← 非官方 |
| scheduler | CosineAnnealingLR |
| crop_point_max | 300K→350K (显存限制) |
| num_worker | 0 |
| 训练方式 | from scratch |
| best mIoU | **0.5143** (epoch 29) |
| epoch 35 | 0.4975 (手动停) |

**问题**: 大模型 (45M) + 小 crop (350K) + 87 文件 = 过拟合，表现不如 V17 small。enc_patch_size 2048 非官方参数，多耗显存。

---

## V20 (LitePT-B, 官方参数+class_weight, 450K, V19 best 权重)

| 项目 | 值 |
|---|---|
| 配置 | `semseg-litept-base-v1m1-loss-v2-clean-v20.py` |
| 模型 | LitePT-B (45M), 官方完整参数 |
| weight | **V19 best (epoch 9, 0.5613)** |
| scheduler | CosineAnnealingLR (eta_min=1e-6, 新初始化) |
| epoch | 50, eval_epoch=5 |
| crop_point_max | 450,000 |
| 采样 | **class_weight=[0.1,0.1,0.05,0.2,0.2,0.5,1.0], prob=0.5** |
| num_worker | 0 |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True |
| GPU | ~38 GiB |

**改动 vs V19**:
- 新增 class-aware center 采样（CylinderCropCUDA 修改）
- 加载 V19 best 权重而非 from scratch
- class_weight_prob=0.5: 50% 加权, 50% 随机

**进行中** (PID 767938)
