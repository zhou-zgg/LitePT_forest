import os
import argparse
import numpy as np
import laspy
from collections import Counter
from scipy.spatial import cKDTree
from scipy.ndimage import label as ndimage_label


def knn_smooth(coord, pred, labels, k=20):
    tree = cKDTree(coord)
    _, indices = tree.query(coord, k=k + 1)
    neighbor_labels = pred[indices[:, 1:]]

    n_points = len(pred)
    n_classes = len(labels)
    votes = np.zeros((n_points, n_classes), dtype=np.int32)
    for c in range(n_classes):
        votes[:, c] = np.sum(neighbor_labels == c, axis=1)

    smoothed = np.argmax(votes, axis=1).astype(pred.dtype)

    changed_mask = smoothed != pred
    total_changed = int(changed_mask.sum())
    old_labels = pred[changed_mask]
    new_labels = smoothed[changed_mask]

    change_detail = Counter()
    for o, n in zip(old_labels, new_labels):
        old_name = labels[o] if o < len(labels) else str(o)
        new_name = labels[n] if n < len(labels) else str(n)
        change_detail[f"{old_name} -> {new_name}"] += 1

    return smoothed, total_changed, change_detail


def fragment_filter(coord, pred, labels, min_points=50, voxel_size=0.1):
    voxel_indices = np.floor(coord / voxel_size).astype(np.int32)

    offset = voxel_indices.min(axis=0)
    voxel_shifted = voxel_indices - offset
    grid_shape = voxel_shifted.max(axis=0) + 1

    filtered = pred.copy()
    total_changed = 0
    change_detail = Counter()

    unique_classes = np.unique(pred)
    for cls in unique_classes:
        cls_mask = pred == cls
        if cls_mask.sum() < min_points:
            continue

        voxel_grid = np.zeros(grid_shape, dtype=np.bool_)
        voxel_grid[voxel_shifted[cls_mask, 0],
                    voxel_shifted[cls_mask, 1],
                    voxel_shifted[cls_mask, 2]] = True

        labeled_array, num_features = ndimage_label(voxel_grid)

        voxel_to_label = {}
        for vi in range(len(voxel_shifted)):
            if cls_mask[vi]:
                key = tuple(voxel_shifted[vi])
                voxel_to_label[key] = labeled_array[key[0], key[1], key[2]]

        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0

        small_components = np.where(component_sizes < max(min_points // (int(voxel_size / 0.015) ** 3), 1))[0]
        small_components = small_components[small_components > 0]

        if len(small_components) == 0:
            continue

        small_mask = np.isin(labeled_array, small_components)
        small_voxels = set()
        for idx in np.argwhere(small_mask):
            small_voxels.add(tuple(idx))

        small_point_indices = []
        for vi in range(len(voxel_shifted)):
            if cls_mask[vi] and tuple(voxel_shifted[vi]) in small_voxels:
                small_point_indices.append(vi)
        small_point_indices = np.array(small_point_indices)

        if len(small_point_indices) == 0:
            continue

        all_labels = np.zeros(n_points := len(pred), dtype=np.int32)
        for c in range(len(labels)):
            all_labels[pred == c] = c

        cls_count = np.bincount(pred[small_point_indices].astype(np.int64),
                                 minlength=len(labels))

        small_coord = coord[small_point_indices]
        tree = cKDTree(coord)
        _, neighbor_indices = tree.query(small_coord, k=20)

        for i, gi in enumerate(small_point_indices):
            neighbor_labels = pred[neighbor_indices[i]]
            votes = np.bincount(neighbor_labels.astype(np.int64), minlength=len(labels))
            votes[pred[gi]] = 0
            if votes.sum() == 0:
                continue
            winner = int(np.argmax(votes))
            if winner != pred[gi]:
                old_name = labels[pred[gi]] if pred[gi] < len(labels) else str(pred[gi])
                new_name = labels[winner] if winner < len(labels) else str(winner)
                change_detail[f"{old_name} -> {new_name}"] += 1
                filtered[gi] = winner
                total_changed += 1

    return filtered, total_changed, change_detail


def save_smoothed_las(coord, pred, labels, las_path, out_path):
    las = laspy.read(las_path)
    if "label" in [d for d in las.point_format.dimension_names]:
        las["label"] = pred.astype(np.float64)
    else:
        las["classification"] = pred.astype(np.uint8)
    las.write(out_path)


def print_stats(name, total, changed, change_detail):
    pct = changed / total * 100 if total > 0 else 0
    print(f"[{name}] 修正统计: {changed}/{total} 点被修改 ({pct:.3f}%)")
    if change_detail:
        for transition, count in change_detail.most_common():
            print(f"  {transition}: {count}")


def main():
    parser = argparse.ArgumentParser(description="点云分割后处理")
    parser.add_argument("--result_dir", required=True, help="预测结果目录 (包含 *_pred.npy)")
    parser.add_argument("--data_root", required=True, help="原始 LAS 文件目录")
    parser.add_argument("--method", choices=["knn_smooth", "fragment_filter", "both"], default="both")
    parser.add_argument("--min_points", type=int, default=50)
    parser.add_argument("--knn_k", type=int, default=20)
    parser.add_argument("--voxel_size", type=float, default=0.1, help="连通域体素化大小 (m)")
    parser.add_argument("--save_las", action="store_true", help="同时输出 smoothed LAS 文件")
    parser.add_argument("--output_suffix", default="_smoothed", help="输出文件后缀")
    args = parser.parse_args()

    labels = ["terrain", "foliage", "CWD", "trunk", "branch", "snag", "non-tree-cyl"]

    npy_files = sorted([f for f in os.listdir(args.result_dir) if f.endswith("_pred.npy")])
    if not args.save_las:
        npy_files = [f for f in npy_files if not f.endswith(f"{args.output_suffix}.npy")]

    for pred_name in npy_files:
        base_name = pred_name.replace("_pred.npy", "")
        las_path = os.path.join(args.data_root, base_name)
        pred_path = os.path.join(args.result_dir, pred_name)

        if not os.path.exists(las_path):
            print(f"[{base_name}] 跳过: 找不到原始文件 {las_path}")
            continue

        las = laspy.read(las_path)
        coord = np.vstack([las.X, las.Y, las.Z]).T.astype(np.float32) * 0.001
        pred = np.load(pred_path)

        total_points = len(pred)
        print(f"\n{'='*60}")
        print(f"[{base_name}] 总点数: {total_points}, 方法: {args.method}")

        result = pred.copy()

        if args.method in ("fragment_filter", "both"):
            result, changed, detail = fragment_filter(
                coord, result, labels, min_points=args.min_points, voxel_size=args.voxel_size
            )
            print_stats(base_name + " (碎片滤波)", total_points, changed, detail)

        if args.method in ("knn_smooth", "both"):
            result, changed, detail = knn_smooth(
                coord, result, labels, k=args.knn_k
            )
            print_stats(base_name + " (KNN平滑)", total_points, changed, detail)

        out_npy_name = pred_name.replace("_pred.npy", f"_pred{args.output_suffix}.npy")
        out_npy_path = os.path.join(args.result_dir, out_npy_name)
        np.save(out_npy_path, result)
        print(f"  -> npy: {out_npy_path}")

        if args.save_las:
            out_las_name = base_name.replace(".las", f"{args.output_suffix}.las")
            out_las_path = os.path.join(args.result_dir, out_las_name)
            save_smoothed_las(coord, result, labels, las_path, out_las_path)
            print(f"  -> las: {out_las_path}")


if __name__ == "__main__":
    main()
