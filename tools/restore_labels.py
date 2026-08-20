"""
Restore labels from backup: backup 4→5, 5→6, 6→4 → current scheme
DOES NOT modify label_backup/ directories.
"""
import os, sys, shutil
import numpy as np, laspy

DATA_ROOT = "data/forest"

def remap_label(old):
    """Backup scheme → Current scheme: BRC(4)→SNG(5), SNG(5)→NTC(6), NTC(6)→BRC(4)"""
    if old == 4: return 5
    elif old == 5: return 6
    elif old == 6: return 4
    else: return old  # 0,1,2,3,7 unchanged

def read_labels_from_file(filepath):
    las = laspy.read(filepath)
    if "label" in list(las.point_format.dimension_names):
        seg = np.array(las.label).astype(np.int32)
    elif "classification" in list(las.point_format.dimension_names):
        seg = np.array(las.classification).astype(np.int32)
    else:
        return None, None
    return las, seg

def write_labels_to_las(filepath, new_labels):
    las = laspy.read(filepath)
    if "label" in list(las.point_format.dimension_names):
        dim_name = "label"
    elif "classification" in list(las.point_format.dimension_names):
        dim_name = "classification"
    else:
        return False
    
    dim_idx = list(las.point_format.dimension_names).index(dim_name)
    old_val = getattr(las, dim_name)
    
    # Make a copy of the original file first
    backup_path = filepath + ".before_restore"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
    
    # Write new labels - laspy doesn't support direct in-place modification of
    # scaled array views, so we reconstruct the LAS
    header = laspy.LasHeader(version="1.4", point_format=las.header.point_format.id)
    header.offsets = las.header.offsets
    header.scales = las.header.scales
    
    out_las = laspy.LasData(header)
    out_las.x = las.x
    out_las.y = las.y
    out_las.z = las.z
    
    # Copy color if present
    for color in ["red", "green", "blue"]:
        if color in list(las.point_format.dimension_names):
            setattr(out_las, color, getattr(las, color))
    
    # Add label dimension
    from laspy import ExtraBytesParams
    out_las.add_extra_dim(ExtraBytesParams(name="label", type=np.int32))
    out_las.label = new_labels
    
    out_las.write(filepath)
    return True

# Define all files to restore: (current_file, backup_file)
restorations = []

# ---- TRAIN pole files ----
train_pole_files = [
    "pole_05_10", "pole_05_11", "pole_05_12", "pole_05_16",
    "pole_05_1_2", "pole_05_3", "pole_05_6", "pole_05_7", "pole_05_8_9",
    "pole3", "pole_05_5",
    # Also restore the "OK" ones since they lost SNG labels
    "pole_05_13", "pole_05_4", "pole_05_14_15",
]
for f in train_pole_files:
    restorations.append((
        os.path.join(DATA_ROOT, "train", f + ".las"),
        os.path.join(DATA_ROOT, "train", "label_backup", f + ".las"),
        "train/" + f + ".las"
    ))

# ---- TRAIN nontree files ----
restorations.append((
    os.path.join(DATA_ROOT, "train", "nontree_other1.las"),
    os.path.join(DATA_ROOT, "train", "label_backup", "nontree.las"),
    "train/nontree_other1.las"
))
restorations.append((
    os.path.join(DATA_ROOT, "train", "nontree_other2.las"),
    os.path.join(DATA_ROOT, "train", "label_backup", "nontree1.las"),
    "train/nontree_other2.las"
))

# ---- TRAIN snag files ----
for f in ["snag1", "snag2", "snag3"]:
    restorations.append((
        os.path.join(DATA_ROOT, "train", f + ".las"),
        os.path.join(DATA_ROOT, "train", "label_backup", f + ".las"),
        "train/" + f + ".las"
    ))

# ---- VAL pole files ----
restorations.append((
    os.path.join(DATA_ROOT, "val", "pole.las"),
    os.path.join(DATA_ROOT, "val", "label_backup", "pole.las"),
    "val/pole.las"
))
restorations.append((
    os.path.join(DATA_ROOT, "val", "pole1.las"),
    os.path.join(DATA_ROOT, "train", "label_backup", "pole1.las"),
    "val/pole1.las"
))
restorations.append((
    os.path.join(DATA_ROOT, "val", "pole2.las"),
    os.path.join(DATA_ROOT, "train", "label_backup", "pole2.las"),
    "val/pole2.las"
))

print("=" * 70)
print("Label Restoration Script")
print("Remapping: backup BRC(4)→SNG(5), SNG(5)→NTC(6), NTC(6)→BRC(4)")
print("Never modifies label_backup/ directories")
print("=" * 70)

for cur_path, bak_path, name in restorations:
    print("\n--- %s ---" % name)
    
    if not os.path.exists(bak_path):
        print("  SKIP: backup not found: %s" % bak_path)
        continue
    if not os.path.exists(cur_path):
        print("  SKIP: current file not found: %s" % cur_path)
        continue
    
    # Read backup labels
    _, bak_seg = read_labels_from_file(bak_path)
    if bak_seg is None:
        print("  SKIP: cannot read labels from backup")
        continue
    
    # Read current labels for comparison
    _, cur_seg = read_labels_from_file(cur_path)
    
    # Count before
    old_counts = {}
    if cur_seg is not None:
        u, c = np.unique(cur_seg, return_counts=True)
        old_counts = {int(k): int(v) for k, v in zip(u, c)}
    
    # Apply remapping
    new_seg = np.vectorize(remap_label)(bak_seg)
    
    # Count after
    u, c = np.unique(new_seg, return_counts=True)
    new_counts = {int(k): int(v) for k, v in zip(u, c)}
    
    names_map = {0:"GND",1:"FOL",2:"CWD",3:"TRK",4:"BRC",5:"SNG",6:"NTC",7:"IGN"}
    
    print("  BEFORE: " + "  ".join("%s=%s" % (names_map.get(k,str(k)), f"{v:,}") for k,v in sorted(old_counts.items())))
    print("  AFTER:  " + "  ".join("%s=%s" % (names_map.get(k,str(k)), f"{v:,}") for k,v in sorted(new_counts.items())))
    
    # Write
    if write_labels_to_las(cur_path, new_seg):
        print("  DONE: labels restored (original saved as .before_restore)")
    else:
        print("  FAILED")
