import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from engines.defaults import (
    default_argument_parser,
    default_config_parser,
)
from utils.logger import get_root_logger
from models.builder import MODELS


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    if not hasattr(cfg, "input_file") or cfg.input_file is None:
        print("Error: --options input_file=<path> is required")
        sys.exit(1)

    os.makedirs(cfg.save_path, exist_ok=True)
    logger = get_root_logger(
        log_file=os.path.join(cfg.save_path, "batched_inference.log")
    )
    logger.info(f"Save path: {cfg.save_path}")
    logger.info(f"Loading weight: {cfg.weight}")
    logger.info(f"Input file: {cfg.input_file}")

    model = MODELS.build(cfg.model).cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {n_parameters}")

    state = torch.load(cfg.weight, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    logger.info(f"Loaded weight (epoch {state.get('epoch', '?')})")
    model.eval()

    grid_size = cfg.data.get("grid_size", 0.02)
    block_xy = cfg.data.get("block_xy", 40.0)
    overlap = cfg.data.get("overlap", 10.0)
    num_classes = cfg.data.get("num_classes", 7)

    logger.info(
        f"grid_size={grid_size}, block_xy={block_xy}m, overlap={overlap}m, "
        f"num_classes={num_classes}"
    )

    input_file = cfg.input_file
    name = os.path.splitext(os.path.basename(input_file))[0]

    if input_file.endswith(".pcd"):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(input_file)
        coord = np.asarray(pcd.points).astype(np.float32)
        segment = np.zeros(len(coord), dtype=np.int32)
    elif input_file.endswith(".las"):
        import laspy
        las = laspy.read(input_file)
        coord = np.vstack([las.X, las.Y, las.Z]).T.astype(np.float32) * 0.001
        segment = np.array(las.label, dtype=np.int32).reshape([-1])
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    orig_coord = coord.copy()
    logger.info(f"Loaded {len(coord)} points, xy extent: "
                f"x=[{coord[:,0].min():.1f}, {coord[:,0].max():.1f}], "
                f"y=[{coord[:,1].min():.1f}, {coord[:,1].max():.1f}], "
                f"z=[{coord[:,2].min():.1f}, {coord[:,2].max():.1f}]")

    x_min, y_min, z_min = coord.min(axis=0)
    x_max, y_max, _ = coord.max(axis=0)
    shift = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min])
    coord = coord - shift

    scaled = coord / grid_size
    grid_coord = np.floor(scaled).astype(np.int64)
    gmin = grid_coord.min(axis=0)
    grid_coord = grid_coord - gmin

    hash_key = grid_coord[:, 0] + grid_coord[:, 1] * 1000000 + grid_coord[:, 2] * 1000000000000
    idx_sort = np.argsort(hash_key)
    hash_key_sort = hash_key[idx_sort]
    _, inv, count = np.unique(hash_key_sort, return_inverse=True, return_counts=True)

    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[0:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_unique = idx_sort[idx_select]

    vox_coord = coord[idx_unique]
    vox_grid_coord = grid_coord[idx_unique]

    logger.info(f"Voxelized: {len(vox_coord)} points at grid_size={grid_size}")

    stride = block_xy - overlap
    x_starts = np.arange(vox_coord[:, 0].min(), vox_coord[:, 0].max(), stride)
    y_starts = np.arange(vox_coord[:, 1].min(), vox_coord[:, 1].max(), stride)

    total_blocks = len(x_starts) * len(y_starts)
    logger.info(f"XY blocks: {len(x_starts)} x {len(y_starts)} = {total_blocks} "
                f"(block={block_xy}m, stride={stride:.1f}m, overlap={overlap}m)")

    full_probs = np.zeros((len(coord), num_classes), dtype=np.float32)
    full_counts = np.zeros(len(coord), dtype=np.float32)

    for xi, x_start in enumerate(tqdm(x_starts, desc="X blocks")):
        x_end = x_start + block_xy
        for yi, y_start in enumerate(y_starts):
            y_end = y_start + block_xy

            mask = (
                (vox_coord[:, 0] >= x_start) & (vox_coord[:, 0] < x_end) &
                (vox_coord[:, 1] >= y_start) & (vox_coord[:, 1] < y_end)
            )
            block_idx = np.where(mask)[0]
            if len(block_idx) == 0:
                continue

            block_vox_coord = vox_coord[block_idx]
            block_vox_grid = vox_grid_coord[block_idx]
            block_orig_idx = idx_unique[block_idx]

            block_grid_min = block_vox_grid.min(axis=0)
            block_vox_grid = block_vox_grid - block_grid_min

            block_coord_t = torch.from_numpy(block_vox_coord).float().cuda()
            block_grid_coord_t = torch.from_numpy(block_vox_grid).int().cuda()
            block_offset = torch.tensor([len(block_vox_coord)], dtype=torch.int32).cuda()
            block_feat = block_coord_t.clone()

            with torch.no_grad():
                input_dict = {
                    "coord": block_coord_t.contiguous(),
                    "grid_coord": block_grid_coord_t.contiguous(),
                    "offset": block_offset,
                    "feat": block_feat.contiguous(),
                }
                output = model(input_dict)
                logits = output["seg_logits"]
                probs = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

            np.add.at(full_probs, block_orig_idx, probs)
            full_counts[block_orig_idx] += 1

    full_counts = np.maximum(full_counts, 1.0)
    full_probs /= full_counts[:, None]
    pred = full_probs.argmax(axis=-1).astype(np.int32)

    save_path = os.path.join(cfg.save_path, "result")
    os.makedirs(save_path, exist_ok=True)
    pred_path = os.path.join(save_path, f"{name}_pred.npy")
    np.save(pred_path, pred)
    logger.info(f"Saved prediction: {pred_path} ({len(pred)} points)")

    for i in range(num_classes):
        count = (pred == i).sum()
        logger.info(f"  class {i}: {count}/{len(pred)} ({100*count/len(pred):.1f}%)")

    class_colors = np.array([
        [128, 64, 0],     # 0 terrain - brown
        [0, 255, 0],      # 1 foliage - green
        [255, 0, 0],      # 2 CWD - red
        [139, 69, 19],    # 3 trunk - saddle brown
        [0, 255, 255],    # 4 branch - cyan
        [255, 255, 0],    # 5 snag - yellow
        [128, 0, 128],    # 6 non-tree-cyl - purple
    ], dtype=np.uint8)
    colors = class_colors[pred]

    if input_file.endswith(".pcd"):
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(orig_coord)
        pcd_out.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        pcd_path = os.path.join(save_path, f"{name}_pred.pcd")
        o3d.io.write_point_cloud(pcd_path, pcd_out)
        logger.info(f"Saved PCD: {pcd_path}")

    if input_file.endswith(".las"):
        import laspy
        las_out = laspy.read(input_file)
        las_out.classification = pred
        las_path = os.path.join(save_path, f"{name}_pred.las")
        las_out.write(las_path)
        logger.info(f"Saved LAS: {las_path}")

    # Also save as LAS for pcd input (since user asked for las output)
    import laspy as _laspy
    header = _laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = orig_coord.min(axis=0)
    header.scales = [0.001, 0.001, 0.001]
    las_out = _laspy.LasData(header)
    las_out.x = orig_coord[:, 0]
    las_out.y = orig_coord[:, 1]
    las_out.z = orig_coord[:, 2]
    las_out.classification = pred.astype(np.uint8)
    las_path = os.path.join(save_path, f"{name}_pred.las")
    las_out.write(las_path)
    logger.info(f"Saved LAS: {las_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
