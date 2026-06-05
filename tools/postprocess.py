import os
import argparse
import numpy as np
import laspy
from collections import Counter
from scipy.spatial import cKDTree


def knn_smooth(coord, pred, labels, k=20):
    smoothed = pred.copy()
    tree = cKDTree(coord)
    total_changed = 0
    change_detail = Counter()

    _, indices = tree.query(coord, k=k + 1)
    for i in range(len(pred)):
        neighbor_labels = pred[indices[i, 1:]]
        votes = Counter(neighbor_labels.tolist())
        winner = votes.most_common(1)[0][0]
        if winner != pred[i]:
            old_label = labels[pred[i]] if pred[i] < len(labels) else str(pred[i])
            new_label = labels[winner] if winner < len(labels) else str(winner)
            change_detail[f"{old_label} -> {new_label}"] += 1
            smoothed[i] = winner
            total_changed += 1

    return smoothed, total_changed, change_detail


def fragment_filter(coord, pred, labels, min_points=50, knn_k=20):
    filtered = pred.copy()
    tree = cKDTree(coord)
    total_changed = 0
    change_detail = Counter()

    unique_classes = np.unique(pred)
    for cls in unique_classes:
        mask = pred == cls
        cls_indices = np.where(mask)[0]
        if len(cls_indices) < min_points:
            continue

        cls_coord = coord[cls_indices]
        cls_tree = cKDTree(cls_coord)
        visited = set()
        components = []

        for i in range(len(cls_coord)):
            if i in visited:
                continue
            queue = [i]
            component = []
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                neighbors = cls_tree.query_ball_point(cls_coord[node], r=0.1)
                for n in neighbors:
                    if n not in visited:
                        queue.append(n)
            components.append(component)

        for comp in components:
            if len(comp) >= min_points:
                continue
            global_indices = cls_indices[comp]
            _, neighbor_indices = tree.query(coord[global_indices], k=knn_k)
            for gi, ni in zip(global_indices, neighbor_indices):
                neighbor_labels = pred[ni]
                votes = Counter(neighbor_labels.tolist())
                winner = votes.most_common(1)[0][0]
                if winner != pred[gi]:
                    old_label = labels[pred[gi]] if pred[gi] < len(labels) else str(pred[gi])
                    new_label = labels[winner] if winner < len(labels) else str(winner)
                    change_detail[f"{old_label} -> {new_label}"] += 1
                    filtered[gi] = winner
                    total_changed += 1

    return filtered, total_changed, change_detail


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
    parser.add_argument("--output_suffix", default="_smoothed", help="输出文件后缀")
    args = parser.parse_args()

    labels = ["terrain", "foliage", "CWD", "trunk", "branch", "snag", "non-tree-cyl"]

    npy_files = sorted([f for f in os.listdir(args.result_dir) if f.endswith("_pred.npy")])

    for pred_name in npy_files:
        base_name = pred_name.replace("_pred.npy", "")
        coord_path = os.path.join(args.data_root, base_name)
        pred_path = os.path.join(args.result_dir, pred_name)

        if not os.path.exists(coord_path):
            print(f"[{base_name}] 跳过: 找不到原始文件 {coord_path}")
            continue

        las = laspy.read(coord_path)
        coord = np.vstack([las.X, las.Y, las.Z]).T.astype(np.float32) * 0.001
        pred = np.load(pred_path)

        total_points = len(pred)
        print(f"\n{'='*60}")
        print(f"[{base_name}] 总点数: {total_points}, 方法: {args.method}")

        result = pred.copy()

        if args.method in ("fragment_filter", "both"):
            result, changed, detail = fragment_filter(
                coord, result, labels, min_points=args.min_points, knn_k=args.knn_k
            )
            print_stats(base_name + " (碎片滤波)", total_points, changed, detail)

        if args.method in ("knn_smooth", "both"):
            result, changed, detail = knn_smooth(
                coord, result, labels, k=args.knn_k
            )
            print_stats(base_name + " (KNN平滑)", total_points, changed, detail)

        out_name = pred_name.replace("_pred.npy", f"_pred{args.output_suffix}.npy")
        out_path = os.path.join(args.result_dir, out_name)
        np.save(out_path, result)
        print(f"  -> 已保存: {out_path}")


if __name__ == "__main__":
    main()
