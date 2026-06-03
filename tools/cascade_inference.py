"""
Stage1 + Stage2 串联推理脚本

流程:
1. Stage1 (6类) 推理 → terrain/foliage/CWD/woody/snag/non-tree
2. 取 Stage1 预测为 woody 的点
3. Stage2 (2类) 推理 → trunk/branch
4. 合并结果，生成带完整标签的 LAS 文件

用法:
    PYTHONPATH=./ python tools/cascade_inference.py \
      --stage1-weight exp/forest/semseg-litept-small-v1m1-stage1/model/model_best.pth \
      --stage2-weight exp/forest/semseg-litept-small-v1m1-stage2/model/model_best.pth \
      --input data/forest/val/20260401_111129.las \
      --output exp/forest/cascade/20260401_111129_pred.las
"""

import argparse
import os
import numpy as np
import laspy
import torch
import torch.nn.functional as F

from models import build_model
from utils.config import Config


def load_model(cfg_path, weight_path, num_classes):
    cfg = Config.fromfile(cfg_path)
    cfg.model["num_classes"] = num_classes
    model = build_model(cfg.model)
    model = model.cuda()
    checkpoint = torch.load(weight_path, weights_only=False)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def grid_sample_test_mode(coord, grid_size=0.02):
    """模拟 GridSample test mode 的分块逻辑，返回 fragment 列表和 inverse 映射"""
    scaled_coord = coord / grid_size
    grid_coord = np.floor(scaled_coord).astype(int)
    min_coord = grid_coord.min(0)
    grid_coord -= min_coord
    key = (grid_coord[:, 0] * 1000000 + grid_coord[:, 1] * 1000 + grid_coord[:, 2]).astype(int)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

    fragment_list = []
    inverse_map = np.zeros_like(inverse)
    inverse_map[idx_sort] = inverse

    for i in range(count.max()):
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
        idx_part = idx_sort[idx_select]
        fragment_list.append(idx_part)

    return fragment_list, inverse_map


def inference_single_model(model, coord, fragment_list, inverse_map, num_classes):
    """在完整点云上跑推理（GridSample test mode 方式）"""
    n_points = coord.shape[0]
    pred_logits = np.zeros((n_points, num_classes), dtype=np.float32)

    offset = np.array([len(f) for f in fragment_list])
    offset = np.cumsum(offset)

    batch_size = 16
    for i in range(0, len(fragment_list), batch_size):
        batch_frags = fragment_list[i : i + batch_size]
        batch_coords = coord[np.concatenate(batch_frags)]
        batch_offset = np.cumsum([len(f) for f in batch_frags])

        batch_tensor = torch.from_numpy(batch_coords).float().cuda()
        input_dict = {"coord": batch_tensor}

        with torch.no_grad():
            output = model(input_dict)
            logits = F.softmax(output["seg_logits"], dim=-1).cpu().numpy()

        start = 0
        for j, frag_idx in enumerate(batch_frags):
            end = start + len(frag_idx)
            pred_logits[frag_idx] += logits[start:end]
            start = end

    return pred_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-weight", required=True)
    parser.add_argument("--stage2-weight", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage1-config", default="configs/forest/semseg-litept-small-v1m1-stage1.py")
    parser.add_argument("--stage2-config", default="configs/forest/semseg-litept-small-v1m1-stage2.py")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading Stage1 model...")
    stage1_model = load_model(args.stage1_config, args.stage1_weight, num_classes=6)

    print("Loading Stage2 model...")
    stage2_model = load_model(args.stage2_config, args.stage2_weight, num_classes=2)

    print(f"Reading LAS: {args.input}")
    las = laspy.read(args.input)
    coord = np.stack([las.x, las.y, las.z], axis=1).astype(np.float32) * 0.001
    n_points = len(coord)

    print(f"Stage1 inference ({n_points:,} points, grid_size=0.02)...")
    fragment_list, inverse_map = grid_sample_test_mode(coord, grid_size=0.02)
    print(f"  {len(fragment_list)} fragments")

    stage1_logits = inference_single_model(stage1_model, coord, fragment_list, inverse_map, num_classes=6)
    stage1_pred = np.argmax(stage1_logits, axis=1)

    woody_mask = (stage1_pred == 3)
    print(f"  Stage1 result: woody={woody_mask.sum():,} / {n_points:,}")

    if woody_mask.sum() == 0:
        print("  No woody points found, using Stage1 result only")
        final_pred = stage1_pred.astype(np.float64)
    else:
        woody_coords = coord[woody_mask]
        print(f"Stage2 inference on {woody_mask.sum():,} woody points...")

        woody_fragment_list, woody_inverse_map = grid_sample_test_mode(woody_coords, grid_size=0.02)
        print(f"  {len(woody_fragment_list)} fragments")

        stage2_logits = inference_single_model(stage2_model, woody_coords, woody_fragment_list, woody_inverse_map, num_classes=2)
        stage2_pred = np.argmax(stage2_logits, axis=1)

        trunk_n = (stage2_pred == 0).sum()
        branch_n = (stage2_pred == 1).sum()
        print(f"  Stage2 result: trunk={trunk_n:,}, branch={branch_n:,}")

        final_pred = stage1_pred.copy().astype(np.float64)
        final_pred[woody_mask] = np.where(stage2_pred == 0, 3, 6)
        final_pred[~woody_mask] = final_pred[~woody_mask]

    final_pred = final_pred.astype(np.float64)
    print(f"Final distribution:")
    for i in range(7):
        n = (final_pred == i).sum()
        if n > 0:
            print(f"  class {i}: {n:,}")

    header = laspy.LasHeader(version="1.4", point_format=2)
    header.offsets = las.header.offsets
    header.scales = las.header.scales
    out_las = laspy.LasData(header)
    out_las.x = las.x
    out_las.y = las.y
    out_las.z = las.z
    out_las.intensity = las.intensity
    out_las.return_number = las.return_number
    out_las.number_of_returns = las.number_of_returns
    out_las["label"] = final_pred

    out_las.write(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
