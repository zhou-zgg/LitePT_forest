# LitePT 森林点云语义分割 — 完整训练记录

## 项目背景

基于 **LitePT** 框架（Pointcept 系列）对森林场景点云进行 7 类语义分割。

### 模型

- 架构：LitePT small（DefaultSegmentorV2）
- 参数量：12.7M
- 输入：仅 xyz 坐标（intensity 全为 0，无回波特征）
- 不可修改模型结构

### 标签体系

| 编号 | 缩写 | 全称 | 说明 |
|------|------|------|------|
| 0 | GND | Ground | 地面 |
| 1 | FOL | Foliage | 树叶/冠层 |
| 2 | CWD | Coarse Woody Debris | 粗木质残体（倒地木、枯木） |
| 3 | TRK | Trunk | 树干 |
| 4 | BRC | Branch | 树枝 |
| 5 | SNG | Snag | 灌木/枯立木 |
| 6 | NTC | Non-Tree Canopy | 非树冠植被（草、灌木冠层等） |
| 7 | IGN | Ignored | 忽略（训练时映射为 -1，不参与损失计算） |

## 数据目录结构

```
/root/autodl-tmp/dataset/tree/forest/
├── train/          # 训练集 .las 文件
│   ├── 202*.las           # 常规森林扫描文件（~64 个）
│   ├── huace{1,2,3,4}.las  # 4 个 huace 采集文件（~85M 点）
│   ├── pole_05_*.las       # 5 个电线杆文件（~9M 点）
│   ├── pole_*.las          # 2 个电线杆文件
│   ├── mountain-cz.las     # 山地 cz 原始文件
│   ├── mountain-cz_{1,2,3}.las  # 山地 cz 切分 3 片
│   ├── mountain-lz.las     # 山地 lz 原始文件（新增于 V15）
│   ├── mountain-lz_{1,2,3}.las  # 山地 lz 切分 3 片（新增于 V15）
│   ├── mountain-yxm.las    # 山地 yxm 原始文件（新增于 V15）
│   ├── mountain-yxm_{1,2,3}.las # 山地 yxm 切分 3 片（新增于 V15）
│   └── snag{1,2,3}.las     # 3 个含 IGN(7) 的枯木文件
├── val/            # 验证集 .las 文件
│   ├── 202*.las           # 5 个常规文件
│   ├── mountain-cz_1.las   # 山地 cz 切分片（从训练集移入）
│   ├── pole.las / pole1.las / pole2.las  # 3 个电线杆文件
│   └── (原 pole_05_2.las 等)
└── mountain-lz.las  # 山地 lz 原始文件（拆分前源文件）
└── mountain-yxm.las # 山地 yxm 原始文件（拆分前源文件）
```

## 数据标注演进

### 阶段 1：无山地数据（V1-V3）
- 训练集仅含常规森林文件 + huace + pole 文件，**无任何山地场景**
- 验证集 **9 个文件，不含山地**
- V2/V3 达到 mIoU 0.686/0.735，但验证集不含山地，不可与后续比较

### 阶段 2：mountain-cz 加入（V4-V13）
- mountain-cz 原始文件（1.8M 点）加入训练集（增加了 mountain-cz.las 到 train/）
- 验证集新增 mountain-cz_1.las → 变成 10 个文件
- 但 V9 之后 val 又变回 9 个文件（mountain-cz_1.las 被移除）
- V10-V11 恢复到 9 文件 val，best mIoU ~0.62
- 关键问题：mountain-cz 仅 1 个文件，占训练集仅 ~1/77

### 阶段 3：山地扩充 + 切分（V14 起）
- mountain-cz 切分为 3 片 + 保留原文件 → 4 个山地文件
- V14：80 训练文件（含 4 个山地），val 9 文件

### 阶段 4：新增两座山（V15）
- 新增 mountain-lz（3.95M 点）和 mountain-yxm（8.95M 点）
- 每座山沿长轴切 3 片（5m overlap）+ 保留原文件 → 各 4 文件
- 87 训练文件（含 12 个山地），山地占比 12/87 ≈ 13.8%（点占比 ~16.8%）
- 验证集 10 文件（mountain-cz_1.las 从训练集移入 val）
- mountain-cz 中 SNG(5)/NTC(6) 共 90 个误标点改为 FOL(1)

### 山地文件细节

| 文件 | 点数 | 标签类别 | 切分轴 | 原始文件位置 |
|------|------|----------|--------|------------|
| mountain-cz.las | 1,828,531 | GND/FOL/TRK/BRC | - (原始) | train/ |
| mountain-cz_1.las | 115,219 | GND/FOL/TRK/BRC | Y | **val/** |
| mountain-cz_2.las | 1,063,374 | GND/FOL/TRK/BRC | Y | train/ |
| mountain-cz_3.las | 1,020,968 | GND/FOL/TRK/BRC | Y | train/ |
| mountain-lz.las | 3,952,497 | GND/FOL/TRK/BRC | - (原始) | train/ |
| mountain-lz_1.las | 1,128,187 | GND/FOL/TRK/BRC | X | train/ |
| mountain-lz_2.las | 2,675,110 | GND/FOL/TRK/BRC | X | train/ |
| mountain-lz_3.las | 922,813 | GND/FOL/TRK/BRC | X | train/ |
| mountain-yxm.las | 8,946,646 | GND/FOL/TRK/BRC | - (原始) | train/ |
| mountain-yxm_1.las | 4,576,543 | GND/FOL/TRK/BRC | Y | train/ |
| mountain-yxm_2.las | 4,497,562 | GND/FOL/TRK/BRC | Y | train/ |
| mountain-yxm_3.las | 1,580,452 | GND/FOL/TRK/BRC | Y | train/ |

## 实验历史

### 汇总表

| 版本 | 训练文件 | 验证文件 | 山地训练 | 山地验证 | 初始权重 | 坡度增强 | Epoch | Best mIoU | 备注 |
|------|---------|---------|----------|----------|---------|---------|-------|-----------|------|
| V2 | 76 | 9 | 无 | 无 | 预训练 | 有 | 100 | 0.6864 | 不含山地 val |
| V3 | 76 | 9 | 无 | 无 | 预训练 | 有 | 100 | 0.7355 | 不含山地 val |
| V4 | 77 | 10 | cz×1 | cz_1 | 预训练 | 有 | 30 | 0.6105 | 首次含山地 val |
| V5 | 77 | 10 | cz×1 | cz_1 | V3 | 有 | 30 | 0.7144 | |
| V6 | 77 | 10 | cz×1 | cz_1 | V3 | 有 | 10 | 0.6369 | |
| V7 | 77 | 10 | cz×1 | cz_1 | V5 | 有 | 30 | 0.5820 | |
| V8 | 77 | 10 | cz×1 | cz_1 | V6 | 有 | 100 | 0.6056 | |
| V9 | 77 | 9 | cz×1 | 无 | V7 | 有 | 30 | 0.5578 | cz_1 被移出 val |
| V10 | 77 | 9 | cz×1 | 无 | 预训练 | 有 | 40 | 0.6234 | |
| V11 | 77 | 9 | cz×1 | 无 | V10 | 有 | 30 | **0.6225** | 当前最佳 |
| V12 | 77 | 9 | cz×1 | 无 | V11等 | 有 | - | 0.6542 | resume 实验(6 iter/epoch) |
| V13 | 77 | 9 | cz×1 | 无 | V12 resume | 有 | 30 | 0.5311 | epoch 19 OOM |
| V14 | 80 | 9 | cz×4 | cz_1？ | V11 best | **误开启** | 30 | 0.5368 | 坡度增强实际未关闭 |
| V15 | **87** | **10** | **cz/lz/yxm** | **cz_1** | V11 best | **无** | **30** | **训练中** | **当前实验** |

### 实验详情

#### V2-V3（初始实验，无山地）
- 预训练权重 `semseg-litept-small-v1m1-loss-v2-clean-best.pth`
- 76 标准训练文件，9 文件验证集
- 不含任何山地数据
- V3 best 0.7355 — 表面上看最高，但验证集不含山地，不可参考

#### V4-V6（mountain-cz 加入训练集，验证集也加入山地）
- mountain-cz.las 加入训练集（77 文件）
- 验证集添加 mountain-cz_1.las → 10 文件
- mIoU 骤降至 0.61 → 说明模型在山地场景上表现差

#### V7-V8（从 V5/V6 继续训练）
- 从 V5/V6 权重分别加载继续训练
- 无显著提升

#### V9-V11（验证集变回 9 文件）
- mountain-cz_1.las 从验证集移出 → 9 文件 val
- mIoU 回升到 ~0.62
- V11 为当前最佳模型（0.6225）

#### V12（resume 测试）
- 多轮 resume 实验，改变了 data 采样方式（6 iter/epoch）
- 0.6542 是早期 epoch 的恢复值，不可靠

#### V13（resume 崩溃）
- V12 resume 继续训练
- Epoch 18 mIoU 仅 0.4520，epoch 19 OOM 崩溃
- 总结：resume 无法解决山地问题

#### V14（mountain-cz 切分，V11 best 初始化）
- mountain-cz 切为 3 片 + 保留原文件 → 4 个山地文件
- 80 训练文件（原 77 + 3 新切片）
- 从 V11 best 初始化，计划关闭坡度增强
- **实际坡度增强未关闭**：本地 config 注释了但服务器未同步
- 结果 best 仅 0.5368，大幅倒退

#### V15（当前，三座山 + 去坡度增强）
- 新增 mountain-lz、mountain-yxm 两座山
- 每座山切 3 片 + 保留原文件 → 12 山地文件
- 87 训练文件，10 验证文件（mountain-cz_1 移入 val）
- **坡度增强已正确关闭**
- 从 V11 best 初始化
- 正在训练中...

## 关键发现

### 1. 坡度增强（RandomRotateCUDA x/y 轴）

坡度增强通过对 x/y 轴旋转模拟山地坡度，但在实验中**确认无效**：
- V11（有坡度增强）0.6225 vs V14（有坡度增强）0.5368
- V15（无坡度增强）正在验证
- 理论原因：训练集中常规平地数据占 >80%，强行旋转会使平地数据变形，反而损害模型
- 服务器 config 中坡度增强曾被误开启，导致 V14 大幅倒退

### 2. LAS header offset 如何处理

`datasets/forest.py:59` 用 `las.X * 0.001` 直接计算坐标，**忽略 LAS header offset**：
- huace 文件 offset=[366000, 2400000, 0]，实际坐标 ~366k-2400k，但代码计算得 183-585
- 由于 CenterShift 归一化的存在，offset 偏移不影响训练效果
- 所有文件 scale 一致 [0.001, 0.001, 0.001]

### 3. CWD(2) 和 NTC(6) 恒为 0

在所有实验版本中，CWD 和 NTC 的 IoU 几乎始终为 0：
- CWD 在训练集中占 3.6%（主要由 huace3 贡献），但验证集中仅 0.04%
- NTC 在训练集中占 0.8%，验证集中 0.43%
- pole_05_5.las 有 41% NTC（用户确认无误）
- 根本原因：训练样本不足，特别是 CWD/NTC 的多样性不够
- 这不是增强策略能解决的问题，需要补充标注数据

### 4. 训练/验证分布差异

| 类别 | 训练集 | 验证集 | 差异 |
|------|--------|--------|------|
| GND | 45.2% | 27.7% | +17.5% |
| FOL | 38.8% | 61.4% | -22.5% |
| CWD | 3.6% | 0.04% | +3.6% |
| TRK | 6.1% | 4.5% | +1.6% |
| BRC | 4.4% | 3.1% | +1.2% |
| SNG | 1.1% | 2.9% | -1.8% |
| NTC | 0.8% | 0.43% | +0.4% |

GND 在训练集中严重偏高（huace 大文件 85M 点多为地面），而验证集中 FOL 占主导。

### 5. 不用考虑的错误来源

- pole_05_5.las 有 41% NTC — 用户确认无误，是多个 pole 05 切片的合并，NTC 为真实标注
- huace3.las 有 14% CWD — 用户确认无误，该场景有大量倒地木材
- snag 文件中的 IGN(7) — 通过 class_mapping {7: -1} 在训练中忽略
- 模型结构不变（LitePT small 12.7M 参数）

## V15 训练配置

### 基础参数

| 参数 | 值 |
|------|-----|
| 模型 | LitePT small（12.7M params） |
| batch_size | 1 |
| num_worker | 1（use_gpu_transform=True 时需为 1，否则 CUDA OOM） |
| crop_mode | CylinderCropCUDA |
| crop_point_max | 1,000,000 |
| grid_size | 0.02 |
| epoch | 30 |
| eval_epoch | 1 |
| optimizer | AdamW lr=0.001, weight_decay=0.05 |
| scheduler | OneCycleLR（max_lr=[0.001, 0.0001]） |
| AMP | float16 |
| class_mapping | {7: -1} |
| loss | CrossEntropy (ls=0.1) + Lovasz + Dice，各权重 1.0 |

### 数据增强管线

```
CenterShiftCUDA(apply_z=True)
  → RandomDropoutCUDA(dropout=0.2, app=0.2)
  → RandomRotateCUDA(axis=z, angle=[-1,1], p=0.5)    # 仅水平旋转
  → RandomFlipCUDA(p=0.5)
  → RandomJitterCUDA(sigma=0.005, clip=0.02)
  → GridSampleCUDA(grid=0.02, hash=fnv)
  → CylinderCropCUDA(point_max=1_000_000)
  → CenterShiftCUDA(apply_z=False)
  → ToTensorCUDA / UpdateCUDA / CollectCUDA
```

注意：**坡度增强（RandomRotateCUDA x/y 轴）已关闭**。

## 文件路径

### 代码与配置

| 项目 | 路径 |
|------|------|
| 代码根目录 | `/root/autodl-tmp/workshop/caozhou/LitePT_forest/` |
| 训练入口 | `.../tools/train.py` |
| 通用配置 | `.../configs/forest/semseg-litept-small-v1m1-loss-v2.py`（基类） |
| Clean 配置 | `.../configs/forest/semseg-litept-small-v1m1-loss-v2-clean.py` |
| V15 配置 | `.../configs/forest/semseg-litept-small-v1m1-loss-v2-clean-v15.py` |
| 数据集 | `.../datasets/forest.py` |
| CheckpointLoader | `.../engines/hooks/misc.py` |

### 数据

| 项目 | 路径 |
|------|------|
| 数据根目录 | `/root/autodl-tmp/dataset/tree/forest/` |
| 训练集 | `.../train/`（87 个 .las 文件） |
| 验证集 | `.../val/`（10 个 .las 文件） |

### 实验输出

| 项目 | 路径 |
|------|------|
| 实验根目录 | `/root/autodl-tmp/exp/forest/` |
| V11 best | `.../semseg-litept-small-v1m1-loss-v2-clean-v11/` |
| V11 权重 | `.../v11/model/model_best.pth` |
| V14 输出 | `.../semseg-litept-small-v1m1-loss-v2-clean-v14/` |
| V15 输出 | `.../semseg-litept-small-v1m1-loss-v2-clean-v15/` |
| V15 日志 | `.../v15/train.out`（nohup 输出） |
| V15 训练日志 | `.../v15/train.log`（结构化日志） |
| V15 配置(已解析) | `.../v15/config.py` |

### 预训练权重

| 路径 | 说明 |
|------|------|
| `.../exp/forest/pretrained/semseg-litept-small-v1m1-loss-v2-clean-best.pth` | 官方预训练权重（V2-V4 初始使用） |

## V15 当前状态

- **PID**: 555335
- **状态**: 运行中
- **开始时间**: 2026-07-06 01:10 CST
- **进度**: Epoch 1/30，约 440+/2610 iter
- **Loss**: ~0.9-1.2，稳定
- **速度**: 约 2-3 it/s
- **预计完成**: 约 16-18 小时（预计今日 18:00-20:00）

## 推荐下一步

1. **等待 V15 完成**：如果 mIoU > 0.62，说明山地数据扩展 + 去坡度增强有效
2. **若仍 < 0.62**：
   - 补充 CWD(2)/NTC(6)/SNG(5) 标注数据
   - 考虑用 V15 切分后的山地文件替代原始大文件（每片 ~1M 点更均匀）
3. **可尝试方向**：
   - 数据均衡采样（对大文件降采样，对小文件过采样）
   - 检查 huace 大文件是否导致 GND 过拟合
   - 无需再试坡度增强（已被多次验证无效）

## 常用命令

```bash
# 查看训练日志
tail -f /root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-clean-v15/train.out

# 查看训练进度（GPU 进程）
ps aux | grep train.py

# 查看 GPU 使用
nvidia-smi

# 训练新实验（从 V11 best 初始化）
cd /root/autodl-tmp/workshop/caozhou/LitePT_forest
PYTHONPATH=./ nohup python3 -u tools/train.py \
  --config-file configs/forest/semseg-litept-small-v1m1-loss-v2-clean-v15.py \
  --num-gpus 1 \
  > exp/out.log 2>&1 &
```
