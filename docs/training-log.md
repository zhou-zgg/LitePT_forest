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

---

# 新数据集实验（forest_new，2026-08-21 起）

## 背景变更

- **新数据集**（`data/forest_new`，本地）/ 服务器同源，与老数据集不同；
- 新增山地场景（mountain-cz 等，val 中有 172 万点的大文件）；
- **CWD(2) 不再是学习目标**：CWD 在新数据集中点数极少（train 仅 ~2182 点、val 为 0），由 `ignore_classes=[2]` 在加载时映射为 ignore_index(-1)，损失与指标全部跳过；模型仍保留 7 通道输出，保证下游工程兼容。mIoU 只看有 GT 点的有效类。

## 汇总表

| 运行 | 模型 | 起点 | scheduler | epoch | crop | 坡度增强 | best mIoU | 状态 |
|---|---|---|---|---|---|---|---|---|
| local16g train_k | Small | scratch | OneCycleLR | 100 | 100K | 无 | (早期) | 中途废弃 |
| 服务器 **V28** | Small | scratch | OneCycleLR | 100 | 1M | 无 | **0.6261** | 废弃(2026-08-23 00:52) |
| 本地 **c350-slope** | Small | warm start | OneCycleLR | 60 | 350K | **有** | **0.7582** | **当前最优** |

## 服务器 V28（scratch + 长调度 → 判定失败）

| 项目 | 值 |
|---|---|
| 目的 | 验证 scratch + 100ep 长调度 + 1M crop + NTC oversample 能否自爬高位 |
| best mIoU | **0.6261**（epoch 5 早期打出，之后 45+ 个高 lr epoch 震荡在 0.5~0.62，不再刷新） |
| 结局 | 2026-08-23 00:52 看门狗 `MAX_RESTARTS=10` 放弃 |
| 失败原因 | ① 中途 OOM 反复（共享 48G 卡 + 1M crop，训练中段峰值顶爆，即使已修 eval 时序）；② scratch 长调度在 lr 退火前一直低效震荡 |
| 留档 | `model_best.pth`(0.6261) 留存 /root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-clean-v28/model/ |

## 本地 c350-slope（warm start + 短调度 + 坡度增强 → 当前最优）

| 项目 | 值 |
|---|---|
| 起点 | 加载 c350 run 的 model_best（warm start） |
| 坡度增强 | **开启**（x/y 轴 ±0.12 rad 旋转），帮助泛化到山地 |
| crop | 350K |
| scheduler | OneCycleLR |
| 进度曲线 | ep19:0.7355 → ep30:0.7582(新Best) → 后段 0.63~0.74 波动，未被超越 |
| best mIoU | **0.7582**（epoch 30, mAcc 0.8237, allAcc 0.8915, F1 0.8585, FWIoU 0.8044） |

### c350-slope 最佳(ep30)逐类指标

| 类 | IoU | Acc | Prec | Rec | 分组(mountain/other) IoU |
|----|-----|-----|------|-----|--------------------------|
| terrain | 0.741 | 0.818 | 0.888 | 0.818 | **0.52 / 0.83** |
| foliage | 0.853 | 0.947 | 0.896 | 0.947 | 0.85 / 0.88 |
| CWD | 0（忽略）| | | | |
| trunk | **0.797** | 0.879 | 0.896 | 0.879 | **0.78 / 0.84** |
| branch | 0.571 | 0.650 | 0.824 | 0.650 | **0.55 / 0.63** |
| snag | 0.711 | 0.772 | 0.900 | 0.772 | **0.71 / 0.71** |
| non-tree-cyl | 0.877 | 0.877 | 1.000 | 0.877 | —(mountain无)/ 0.80 |

**山地 vs 平地**：terrain 仍是最短板(0.52 vs 0.83，但已较早期 0.14 大幅回升)；snag 两组完全持平；trunk/branch 山地仅回落 0.06/0.08，符合预期。

## 关键结论

1. **warm start + 短调度(60ep) ≫ scratch + 长调度(100ep)**：本地 0.7582 vs 服务器 0.6261。与老数据 V24/V27（热启动+OneCycle）的经验一致。
2. **坡度增强这次有效**：c350-slope 加了坡度增强后超过 c350(0.6569)，与老数据集"坡度增强无效/有害"的结论相反——因新数据集含更多山地。山地增强的收益随山地数据比例增大而增大。
3. **跨口径、跨数据集 mIoU 只能比量级**：去掉 CWD 使分母 7→6，最坏 `M7=6×M6/7`，差约 0.03~0.07（非 0.15~0.2）；且新旧数据集本身不同（新含山地、CWD≈0），严格不可比。
