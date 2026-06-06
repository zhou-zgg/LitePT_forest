import os, sys, json
import numpy as np
import laspy

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

data_root = "data/forest"
save_dir = "exp/forest/semseg-litept-small-v1m1-patch2048/result"

with open(os.path.join(os.path.dirname(save_dir), "eval_val_result.json")) as f:
    result = json.load(f)

val_dir = os.path.join(data_root, "val")
files = sorted([f for f in os.listdir(val_dir) if f.endswith(".las")])

names = ["terrain", "foliage", "CWD", "trunk", "branch", "snag", "non-tree-cyl"]

stats = {
    "snag_correct": [],
    "snag_as_trunk": [],
    "snag_as_branch": [],
}

for fn in files:
    name = fn.replace(".las", "")
    pred_path = os.path.join(save_dir, f"{name}.las_pred.npy")
    if not os.path.exists(pred_path):
        print(f"skip {name}")
        continue

    las = laspy.read(os.path.join(val_dir, fn))
    coord = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    segment = np.array(las.label).astype(np.int32)
    pred = np.load(pred_path)

    n = min(len(segment), len(pred))
    segment = segment[:n]
    pred = pred[:n]
    coord = coord[:n]

    x_min, y_min = coord[:, 0].min(), coord[:, 1].min()
    z_min = coord[:, 2].min()

    terrain_mask = segment == 0
    if terrain_mask.sum() > 100:
        ground_z = np.percentile(coord[terrain_mask, 2], 5)
    else:
        ground_z = z_min

    height = coord[:, 2] - ground_z

    snag_gt = segment == 5
    snag_correct = snag_gt & (pred == 5)
    snag_as_trunk = snag_gt & (pred == 3)
    snag_as_branch = snag_gt & (pred == 4)

    for label, mask in [("snag_correct", snag_correct), ("snag_as_trunk", snag_as_trunk), ("snag_as_branch", snag_as_branch)]:
        count = mask.sum()
        if count > 0:
            h = height[mask]
            z = coord[mask, 2]
            stats[label].append({
                "scene": fn,
                "count": int(count),
                "z_min": float(z.min()),
                "z_max": float(z.max()),
                "z_mean": float(z.mean()),
                "z_std": float(z.std()),
                "h_min": float(h.min()),
                "h_max": float(h.max()),
                "h_mean": float(h.mean()),
                "h_std": float(h.std()),
                "h_p10": float(np.percentile(h, 10)),
                "h_p50": float(np.percentile(h, 50)),
                "h_p90": float(np.percentile(h, 90)),
            })

for label in stats:
    entries = stats[label]
    if not entries:
        print(f"\n{'='*60}")
        print(f"{label}: no points")
        continue
    total = sum(e["count"] for e in entries)
    h_all = []
    for e in entries:
        h_all.extend([e["h_p10"], e["h_p50"], e["h_p90"]])
    print(f"\n{'='*60}")
    print(f"{label}: {total:,} points across {len(entries)} scenes")
    print(f"  height (above ground) stats per scene:")
    print(f"  {'scene':<25s} {'count':>7s} {'h_mean':>7s} {'h_std':>6s} {'h_p10':>6s} {'h_p50':>6s} {'h_p90':>6s}")
    for e in entries:
        print(f"  {e['scene']:<25s} {e['count']:>7d} {e['h_mean']:>7.2f} {e['h_std']:>6.2f} {e['h_p10']:>6.2f} {e['h_p50']:>6.2f} {e['h_p90']:>6.2f}")
