import os
import numpy as np
import laspy

data_root = "/home/caozhou/workshop/deep_learning/LitePT/data/forest/val"
result_root = "/home/caozhou/workshop/deep_learning/LitePT/exp/forest/semseg-litept-small-v1m1/result"

# Get all .npy files in the result directory
npy_files = [f for f in os.listdir(result_root) if f.endswith('_pred.npy')]

# Create list of (las_name, pred_name) tuples
files = []
for pred_name in npy_files:
    # Extract base name without '_pred.npy' to get the original las filename
    base_name = pred_name.replace('_pred.npy', '')
    files.append((base_name, pred_name))

for las_name, pred_name in files:
    las_path = os.path.join(data_root, las_name)
    pred_path = os.path.join(result_root, pred_name)
    out_path = os.path.join(result_root, las_name.replace(".las", "_pred.las"))

    las = laspy.read(las_path)
    pred = np.load(pred_path)

    print(f"{las_name}: {len(las.points)} points, pred shape {pred.shape}")
    print(f"  Original label unique: {np.unique(las['label'])}")
    print(f"  Pred unique: {np.unique(pred)}")

    las["label"] = pred.astype(np.float64)
    las.write(out_path)
    print(f"  -> {out_path}")