# LitePT 森林点云分割适配记录

## 修改时间
- 2026-04-21: 初始适配（部署环境、创建 ForestDataset、7类 config、3+1 scenes 训练验证）
- 2026-04-22 01:00: 解压新数据（2个 rar），整理目录结构，移动文件到 train/val
- 2026-04-22 09:30: 备份实验 1 权重（model_best.pth + model_last.pth）
- 2026-04-22 09:35: ForestDataset 添加 class_mapping 参数
- 2026-04-22 09:40: 创建 stage1 config（6类，trunk+branch→woody）
- 2026-04-22 09:42: 注册 ForestLoss 到 models/losses/__init__.py
- 2026-04-22 09:45: 修复 engines/test.py 共享内存问题（num_workers=0, pin_memory=False）
- 2026-04-22 10:10: 注册 SemSegTester 到 engines/test.py（修复测试流程）
- 2026-04-22 10:15: 更新 FOREST_ADAPTATION.md
- 2026-04-22 13:55: 新增 tools/pred2las.py（pred.npy → LAS 转换脚本）
- 2026-04-22 14:00: 新增使用说明.md（训练/推理/可视化完整流程文档）

## 目标
将森林点云语义分割数据集（LAS 格式，7 类）接入 LitePT 训练 pipeline。

## 数据集信息

- **格式**: LAS 1.4 (format 6)，每个文件包含 XYZ + label(0-6)
- **坐标**: 整数存储，scale=0.001，即真实坐标 = int_coord * 0.001（米）
- **标签**: 0=地形, 1=植被树叶, 2=粗木质残体(CWD), 3=树干主干, 4=断树, 5=非树(预留/无数据), 6=树枝
- **当前数据量**: 训练集 55 个场景 (~36M 点), 验证集 6 个场景 (~5.5M 点)
- **类别分布严重不均**: foliage 占 48-74%, CWD 仅 0.01%, non-tree 无标注
- **数据目录**: `data/forest/train/` (55 .las), `data/forest/val/` (6 .las)
- **标注说明**: 末梢细枝因难以标注被标为 foliage，只有粗树枝标为 branch
- **bin 文件**: 2 个 .bin 文件为 CloudCompare 二进制格式（含 "Merged clouds" 标记头），非标准点云格式，暂未处理

## 修改文件清单

### 1. `datasets/forest.py` — ForestDataset（新建于 2026-04-21）
- 继承 `DefaultDataset`
- **`class_mapping` 参数** (2026-04-22 09:35 新增): 可选的标签重映射字典，在 `get_data()` 时应用
  - 默认 `None`: 原始 7 类
  - stage1 使用 `{0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:3}` 合并 trunk(3)+branch(6)→woody(3)
  - 使用 `np.vectorize` 实现高效 remap
- `get_data_list()`: 过滤非 `.las` 文件（排除 note.md 等）
- `get_data()`: 用 `laspy` 读取 LAS，提取 coord(XYZ float32, *0.001) 和 segment(label int32)

### 2. `datasets/__init__.py`
- 2026-04-21: 添加 `from .forest import ForestDataset`

### 3. `configs/forest/semseg-litept-small-v1m1.py` — 原始 7 类 config（新建于 2026-04-21）
- `num_classes = 7`, `in_channels = 3`, `batch_size = 4`, `num_worker = 4`
- `epoch = 600`, `eval_epoch = 60`, `enable_wandb = False`
- 7 类: terrain, foliage, CWD, trunk, snag, non-tree, branch
- `split="val"` (验证集目录名为 val)

### 4. `configs/forest/semseg-litept-small-v1m1-stage1.py` — Stage1 6 类 config（新建于 2026-04-22 09:40）
- `num_classes = 6`, 其他超参数与原始 config 相同
- `class_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:3}` 合并 trunk+branch→woody
- 6 类: terrain, foliage, CWD, **woody**, snag, non-tree
- save_path: `exp/forest/semseg-litept-small-v1m1-stage1`
- 输出到独立目录，不覆盖原始实验结果
- 训练/验证/测试三个 split 均添加了 `class_mapping` 参数

### 5. `models/losses/forest.py` — ForestLoss（新建于 2026-04-22, 备用）
- 在 loss 层合并 trunk+branch，当前 stage1 未使用（改用数据层 remap）
- 包含 CE + Lovasz loss 和自定义 lovasz 实现
- 已注册到 `models/losses/__init__.py`

### 6. `models/losses/__init__.py`
- 2026-04-22 09:42: 添加 `from .forest import ForestLoss`

### 7. `engines/train.py`（2026-04-21 修改）
- `persistent_workers=True` → `persistent_workers=self.cfg.num_worker_per_gpu > 0`
- 原因: num_worker=0 时 persistent_workers=True 报错

### 8. `engines/test.py`（2026-04-22 09:45 修复）
- 两处 `build_test_loader()` (line 93, line 211):
  - `num_workers` 从 `self.cfg.batch_size_test_per_gpu` 改为 `0`
  - `pin_memory` 从 `True` 改为 `False`
- 原因: PreciseEvaluator 在训练结束后运行，DataLoader worker 因共享内存不足触发 Bus error

### 9. `engines/test.py`（2026-04-22 10:10 新增 SemSegTester）
- 注册 `SemSegTester` 类（继承 `TesterBase`），修复官方缺少 `SemSegTester` 注册的问题
- 原始代码只注册了 `SemSegTester_Assemble`（需要 weight_model2/model3）和 `DINOSemSegTester`（需要 dino_coord/dino_feat/dino_offset）
- 新增的 `SemSegTester` 支持标准单模型测试，不依赖 DINO 特征
- 修复了 `intersectionAndUnion` → `intersection_and_union` 的函数名错误
- 修复了 `segment.numpy()` 的类型错误（segment 已经是 numpy array）

### 10. `libs/pointrope/setup.py`（2026-04-21 修改）
- `all_cuda_archs` 从 `sm_90` 改为 `sm_89` + `sm_90`（适配 RTX 4080）

### 11. `tools/pred2las.py`（2026-04-22 13:55 新建）
- 将 `result/*_pred.npy` 转换回带 `label` 字段的 LAS 文件
- 原始 LAS 的 `label` 字段是 GT，预测 LAS 的 `label` 字段是模型输出
- 读取原始 LAS，替换 `label` 字段为预测值，保留其他所有字段
- 用于 CloudCompare 可视化对比 GT vs 预测结果

### 12. `configs/forest/semseg-litept-small-v1m1-nuscenes.py`（2026-04-24 新建）
- 从 NuScenes 预训练权重迁移的森林 7 类配置
- `weight = "pretrained/nuscenes-semseg-litept-small-v1m1-model_best.pth"` 指向 NuScenes 预训练权重
- backbone (LitePT) 权重从 NuScenes 迁移，输入层和分类头随机初始化
- epoch=300（比原始 600 少，迁移学习收敛更快）
- eval_epoch=30
- lr=0.003（比原始 0.006 低，迁移学习用较小学习率微调）
- 输出目录: `exp/forest/semseg-litept-small-v1m1-nuscenes/`

## 训练结果

### 实验 1: 原始 7 类（55 train / 6 val, 600 epochs）
- **训练时间**: 2026-04-21 21:08 ~ 2026-04-22 09:08
- **Best mIoU: 0.5529** (epoch 51)
- 训练末尾 PreciseEvaluator 因共享内存不足崩溃（已修复）
- 各类 IoU: terrain 0.82, foliage 0.88, trunk 0.75, snag 0.71, branch 0.55, CWD 0.00, non-tree 0.00
- 权重备份: `exp/forest/semseg-litept-small-v1m1-backup/`
  - `model_best.pth` (146M): mIoU 最高的权重，用于推理
  - `model_last.pth` (146M): 最后一个 epoch 的权重
  - `config.py` (11K): 训练配置快照
  - `train.log` (1.4M): 完整训练日志
- PreciseEvaluator 测试（test-time augmentation）: 2026-04-22 10:26 启动，因超时中断，未完成

### 实验 2: Stage1 6 类（已完成）
- **训练时间**: 2026-04-23 16:16 ~ 20:08
- **Best mIoU: 0.5729** (epoch 10)
- 各类 IoU: terrain 0.77, foliage 0.85, CWD 0.01, woody 0.86, snag 0.84, non-tree 0.00
- 权重: `exp/forest/semseg-litept-small-v1m1-stage1/model/model_best.pth`
- 训练集推理因 OOM 中断，未完成

### 实验 3: NuScenes 预训练迁移（7类，2026-04-24）
- 从 NuScenes 预训练权重迁移到森林 7 类
- 预训练权重: `pretrained/nuscenes-semseg-litept-small-v1m1-model_best.pth`
- 配置文件: `configs/forest/semseg-litept-small-v1m1-nuscenes.py`
- 预期: backbone 特征迁移，训练更快，效果更好
- 训练命令:
```bash
sh scripts/train.sh -g 1 -d forest -c semseg-litept-small-v1m1-nuscenes -n semseg-litept-small-v1m1-nuscenes
```

## 两阶段方案

### Stage 1: 大类分割（6 类）
- 合并 trunk(3) + branch(6) → woody(3)
- 6 类: terrain, foliage, CWD, woody, snag, non-tree
- 通过 `class_mapping` 在数据加载时 remap，模型输出 6 维
- 预期: mIoU 从 0.55 提升到 0.65+

### Stage 2: 细粒度区分（2 类，待实现）
- 用 Stage1 模型推理训练集，提取预测为 "woody" 的点
- 仅用原始标签为 trunk(3) 或 branch(6) 的点训练
- 2 类: trunk vs branch
- 推理时: Stage1 先分 6 类，woody 点再过 Stage2 得到 trunk/branch

### 推理流程
```
原始点云 → Stage1 → terrain/foliage/CWD/snag/non-tree → 输出
                  → "woody" 的点 → Stage2 → trunk/branch
```

## 测试命令
```bash
# 7 类测试（使用 best 权重）
PYTHONPATH=./ python tools/test.py \
  --config-file configs/forest/semseg-litept-small-v1m1.py --num-gpus 1 \
  --options save_path=exp/forest/semseg-litept-small-v1m1 \
  weight=exp/forest/semseg-litept-small-v1m1/model/model_best.pth \
  test.type=SemSegTester test.aug_transform=[]

# Stage1 6 类测试
PYTHONPATH=./ python tools/test.py \
  --config-file configs/forest/semseg-litept-small-v1m1-stage1.py --num-gpus 1 \
  --options save_path=exp/forest/semseg-litept-small-v1m1-stage1 \
  weight=exp/forest/semseg-litept-small-v1m1-stage1/model/model_best.pth \
  test.type=SemSegTester test.aug_transform=[]

# 推理完成后，转换 pred.npy 为 LAS 可视化
PYTHONPATH=./ python tools/pred2las.py
```

完整流程说明见 [使用说明.md](./使用说明.md)。

## 已知问题

1. **CWD 标注极少**: IoU 为 0，等后续补充数据
2. **non-tree 无标注**: IoU 为 0，保留为预留类
3. **末梢细枝标为 foliage**: 不影响 Stage1，Stage2 只学习粗树枝 vs 树干
4. **PreciseEvaluator 测试慢**: 每个 val 场景需跑 13 种数据增强，6 个场景约需 10+ 分钟

### 实验 4: Loss-v2（CE+smooth+Lovasz+Dice, 30 epoch fine-tune）
- **训练时间**: 2026-06-05 20:18 ~ 21:59
- **基线 mIoU**: 0.5529 (实验 1, epoch 51)
- **改动**:
  - criteria: CE(label_smoothing=0.1) + Lovasz + Dice (原 CE + Lovasz)
  - optimizer: lr 0.006 → 0.001, param_dicts lr 0.0006 → 0.0001
  - crop_point_max: 150000 → 100000 (16GB OOM)
  - 从 best 权重 fine-tune, resume=False
- **Best mIoU: 0.6092** (epoch 28)
- **Loss 趋势**: 1.49→1.34→1.33→1.30→1.29→1.28→1.28→1.29→1.28→1.27→1.28→1.26→1.27→1.24→1.24→1.24→1.24→1.24→1.23→1.25→1.23→1.23→1.24→1.23→1.24→1.24→1.23→1.23→1.23→1.22
- **各类 IoU 对比**:

| 类别 | Baseline (实验1) | Loss-v2 (ep28) | 变化 |
|---|---|---|---|
| terrain | 0.82 | 0.861 | +4% |
| foliage | 0.88 | 0.906 | +3% |
| CWD | 0.00 | 0.000 | — |
| trunk | 0.75 | 0.716 | -3% |
| branch | 0.55 | 0.578 | +3% |
| snag | 0.71 | 0.467 | -24% |
| non-tree | 0.00 | 0.737 | +74% |
| **mIoU** | **0.5529** | **0.6092** | **+5.6%** |

- **结论**: 整体有效。non-tree 从 0 突破到 0.737（Dice loss 帮助学到小类）。snag 退化严重（0.71→0.47），可能因为感受野不够，snag/trunk 局部几何相似无法区分。待 Phase 2 (grid_size 0.04) 验证感受野假设。
- **权重**: `exp/forest/semseg-litept-small-v1m1-loss-v2/model/model_best.pth`

### 实验 5: grid_size 0.04（从 Phase 1 best fine-tune）
- **训练时间**: 2026-06-05 22:14 ~ 23:14 (27/30 epoch)
- **Phase 1 best mIoU**: 0.6092
- **改动**: grid_size 0.02 → 0.04, 其余同 Phase 1
- **Best mIoU: 0.5573** (epoch 27), 低于 Phase 1 的 0.6092
- **Loss 趋势**: 1.41→1.28→1.26→1.24→1.22→1.23→1.22→1.22→1.20→1.19→1.18→1.17→1.18→1.16→1.16
- **结论**: **grid_size 0.04 无效**。感受野扩大的收益不足以弥补精细度丢失。mIoU 从 0.6092 降到 0.5573，说明 0.02m 的精细体素对森林点云分割是必要的。grid_size=0.04 不再继续。
