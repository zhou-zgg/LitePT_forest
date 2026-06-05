# 森林点云分割精度提升设计

## 问题

稀疏森林点云（55 train / 6 val 场景，~36M 训练点），存在噪声，类别极度不均（foliage 48-74%，CWD 0.01%）。当前 LitePT-S best mIoU = 0.5529，核心问题：

1. **分割碎片化**：同一物体被分成多个类别片段（树干中间出现一小段非树，树枝中出现断树等）
2. **类别混淆**：snag/trunk/branch/non-tree 局部几何相似，模型难以区分
3. **空间不一致性**：一根电线杆一半是非树一半是树枝

各类 IoU 现状：terrain 0.82, foliage 0.88, trunk 0.75, snag 0.71, branch 0.55, CWD 0.00, non-tree 0.00。

## 约束

- 不改模型结构
- 本地 RTX 4080 16GB 先验证，有效后再部署到服务器 4080 32GB
- 显存增加的策略放最后验证
- 从 best 权重 fine-tune，每次 30 epoch
- 点云无回波特征（return_number 字段不可用），intensity 全为 0
- 已验证简单反频率加权会导致模型到处预测小类，不可用
- CWD 类别保留（不影响其他类，未来可能补充数据）

## 方案概览

三个阶段，按显存消耗递增排列。每阶段从上一阶段的 best 权重 fine-tune。

---

## Phase 1: Loss 增强（零显存增加）

### 1.1 三 loss 联合 + Label Smoothing

将 `CrossEntropyLoss` + `LovaszLoss` 替换为：

```python
criteria=[
    dict(type="CrossEntropyLoss", loss_weight=1.0, label_smoothing=0.1, ignore_index=-1),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    dict(type="DiceLoss", loss_weight=1.0, ignore_index=-1),
]
```

设计理由：
- **不加 CE 权重**：简单反频率加权已验证会翻车（模型到处预测 snag/non-tree 获取奖励）
- **Lovasz** 直接优化 IoU，天然对类别数量不敏感
- **Dice** 归一化计算，对小类有更好的梯度信号，不会出现"到处预测小类"问题
- **Label smoothing 0.1**：减少模型对边界点的过度自信，缓解碎片化

### 1.2 Fine-tune 配置

从 `exp/forest/semseg-litept-small-v1m1/model/model_best.pth` 继续：

```python
weight = "exp/forest/semseg-litept-small-v1m1/model/model_best.pth"
resume = False
epoch = 30
eval_epoch = 5
```

优化器调整（fine-tune 用更小学习率）：
```python
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

### 1.3 涉及文件

- 新建 config: `configs/forest/semseg-litept-small-v1m1-loss-v2.py`（从现有 config 复制并修改 criteria、optimizer、weight）
- 无需修改模型/loss 代码（`DiceLoss` 和 `CrossEntropyLoss` 的 `label_smoothing` 已存在）

### 1.4 验收标准

- val mIoU > 0.5529（当前 best）
- 各类 IoU 不退化
- 30 epoch 内 mIoU 趋势向上

---

## Phase 1b: 推理后处理（离线运行，不占训练显存）

### 2.1 连通域碎片滤波

算法：
1. 对预测结果按类别分组
2. 对每个类别用体素连通域找连通区域
3. 小于 `min_points` 的孤立碎片 → KNN 投票替换为周围主流类别

参数：
- `min_points`：连通域最小点数阈值（默认 50）
- `knn_k`：KNN 投票邻居数（默认 20）

### 2.2 KNN 概率平滑

对每个点取 K 近邻，用邻域内预测概率的距离衰减加权平均修正预测：

```
smoothed_prob[i] = weighted_avg(近邻 softmax 概率, 距离衰减)
final_label[i] = argmax(smoothed_prob[i])
```

比连通域更平滑，但可能过度模糊边界。可独立使用或配合碎片滤波。

### 2.3 修正统计输出

脚本必须打印每个场景的修正统计，格式：

```
[163414] 修正统计: 1283/1903607 点被修改 (0.067%)
  non-tree → trunk:   842
  branch  → trunk:    231
  snag    → trunk:    105
  trunk   → snag:      58
  ...
[190082] 修正统计: 567/... 点被修改 (...)
```

按 `(原标签 → 新标签)` 分组计数，按修改数量降序排列。

### 2.4 涉及文件

- 新建: `tools/postprocess.py`
- 输入: `result/*_pred.npy` + 原始 LAS 文件
- 输出: `*_pred_smoothed.npy` + 修正统计到终端
- 独立于训练 pipeline

### 2.5 验收标准

- 碎片化明显减少（可视化对比）
- 修正统计清晰可读
- 不影响大类（terrain/foliage）的分割结果

---

## Phase 2: grid_size 调整（实际减少显存）

将 `grid_size` 从 0.02m 调整为 0.04m。

设计理由：
- 当前感受野约 2-3m，不足以区分 snag vs trunk（需要 5-10m 上下文）
- grid_size 翻倍 → 同样 1024 点 patch 覆盖空间范围翻倍 → 感受野 x2
- 体素数量减少 → 显存降低
- 风险：精细度可能下降（在细树枝等小结构上）

前提：Phase 1 已验证 loss 改动有效，从 Phase 1 best 权重继续 fine-tune。

验收标准：
- mIoU 不低于 Phase 1 结果
- snag/trunk 混淆减少
- 细小结构（细枝）IoU 可接受退化

---

## Phase 3: crop_point_max 增大（增加显存，服务器验证）

增大 `crop_point_max`，让模型一次看到更多场景上下文。

前提：Phase 2 已验证，仅在服务器 4080 32GB 上运行。

验收标准：mIoU > Phase 2 最佳结果。

---

## 不做的事情

- 不加简单反频率加权（已验证翻车）
- 不加 Focal Loss（与 Lovasz + Dice 功能重叠）
- 不加回波特征（数据无回波信息）
- 不做 CRF 后处理（实现复杂，收益有限）
- 不改模型结构
