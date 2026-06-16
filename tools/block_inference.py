#!/usr/bin/env python
"""
点云语义分割推理 - 分块存盘方案（参考 FSCT）
步骤1: 加载点云 → CPU空间分桶(O(N)) → 保存coords.npy + 每块索引npy
步骤2: 逐块推理 → 存预测 npy（断点续跑）
步骤3: 按全局索引合并 → 写 LAS
"""

import argparse
import os
import sys
import time
import glob
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import laspy
import open3d as o3d
import gc
from copy import deepcopy
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models.builder import build_model
from datasets.transform import (
    CenterShift, GridSample, ToTensor, Collect, RandomRotateTargetAngle,
    RandomScale, RandomFlip, Compose,
)


class Inference:
    def __init__(self, weight_path, grid_size=0.02, num_classes=7,
                 block_size=20.0, overlap=2.0, use_tta=True, device="cuda"):
        self.weight_path = weight_path
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.block_size = block_size
        self.overlap = overlap
        self.use_tta = use_tta
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._build_model()
        self._build_pipeline()

    def _build_model(self):
        model_cfg = dict(
            type="DefaultSegmentorV2", num_classes=self.num_classes,
            backbone_out_channels=72,
            backbone=dict(
                type="LitePT", in_channels=3,
                order=("z", "z-trans", "hilbert", "hilbert-trans"),
                stride=(2, 2, 2, 2), enc_depths=(2, 2, 2, 6, 2),
                enc_channels=(36, 72, 144, 252, 504),
                enc_num_head=(2, 4, 8, 14, 28),
                enc_patch_size=(1024, 1024, 1024, 2048, 2048),
                enc_conv=(True, True, True, False, False),
                enc_attn=(False, False, False, True, True),
                enc_rope_freq=(100., 100., 100., 100., 100.),
                dec_depths=(0, 0, 0, 0), dec_channels=(72, 72, 144, 252),
                dec_num_head=(4, 4, 8, 14),
                dec_patch_size=(1024, 1024, 1024, 1024),
                dec_conv=(False, False, False, False),
                dec_attn=(False, False, False, False),
                dec_rope_freq=(100., 100., 100., 100., 100.),
                mlp_ratio=4, qkv_bias=True, qk_scale=None,
                attn_drop=0., proj_drop=0., drop_path=0.3,
                shuffle_orders=True, pre_norm=True, enc_mode=False,
            ), criteria=[],
        )
        self.model = build_model(model_cfg).to(self.device)
        print(f"加载权重: {self.weight_path}", flush=True)
        ckpt = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        sd = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            sd[k[7:] if k.startswith("module.") else k] = v
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        print(f"模型完成 epoch {ckpt['epoch']} | {self.device}", flush=True)

    def _build_pipeline(self):
        self.test_voxelize = GridSample(
            grid_size=self.grid_size, hash_type="fnv", mode="train",
            return_grid_coord=True, return_inverse=True,
        )
        self.post_transform = Compose([
            dict(type="CenterShift", apply_z=False), dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=("coord",)),
        ])
        if self.use_tta:
            aug_list = [
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0,0,0], p=1), dict(type="RandomScale", scale=[0.95,0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1), dict(type="RandomScale", scale=[0.95,0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0,0,0], p=1), dict(type="RandomScale", scale=[0.95,0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1), dict(type="RandomScale", scale=[0.95,0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0,0,0], p=1), dict(type="RandomScale", scale=[1.05,1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1), dict(type="RandomScale", scale=[1.05,1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0,0,0], p=1), dict(type="RandomScale", scale=[1.05,1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1), dict(type="RandomScale", scale=[1.05,1.05])],
                [dict(type="RandomFlip", p=1)],
            ]
        else:
            aug_list = [[dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0,0,0], p=1)]]
        self.aug_transform = [Compose(aug) for aug in aug_list]

    def infer_block(self, block_coords):
        block_coords = CenterShift(apply_z=True)({"coord": block_coords.copy()})["coord"]
        data_dict = self.test_voxelize({"coord": block_coords})
        n_ds = len(data_dict["coord"])
        inverse = data_dict["inverse"]
        pred = torch.zeros((n_ds, self.num_classes)).cuda()
        for ai, aug in enumerate(self.aug_transform):
            fd = aug(deepcopy(data_dict))
            fd["index"] = np.arange(n_ds)
            frag = self.post_transform(fd)
            for k in frag:
                if isinstance(frag[k], torch.Tensor): frag[k] = frag[k].cuda(non_blocking=True)
            with torch.no_grad():
                pred += F.softmax(self.model(frag)["seg_logits"], -1)
            del fd, frag
            if (ai+1) % 4 == 0 or ai == len(self.aug_transform)-1:
                print(f"        aug {ai+1}/{len(self.aug_transform)}", flush=True)
        preds = pred.max(1)[1].cpu().numpy()[inverse]
        del data_dict, pred; torch.cuda.empty_cache(); gc.collect()
        return preds

    def infer_block_safe(self, block_coords, max_pts=800000):
        if len(block_coords) <= max_pts:
            return self.infer_block(block_coords)
        print(f"      大块 {len(block_coords):,}点, 空间二次分块 (max={max_pts:,})...", flush=True)
        result = np.zeros(len(block_coords), dtype=np.uint8)

        xy = block_coords[:, :2]
        x_range = xy[:, 0].max() - xy[:, 0].min()
        y_range = xy[:, 1].max() - xy[:, 1].min()

        if x_range >= y_range:
            order = np.argsort(xy[:, 0])
        else:
            order = np.argsort(xy[:, 1])
        del xy; gc.collect()

        n_sub = (len(block_coords) - 1) // max_pts + 1
        sub_size = (len(block_coords) + n_sub - 1) // n_sub
        for i in range(0, len(block_coords), sub_size):
            sub_order = order[i:i+sub_size]
            sub = block_coords[sub_order]
            print(f"      子块 {i//sub_size+1}/{n_sub}: {len(sub):,}点", flush=True)
            result[sub_order] = self.infer_block(sub)
            del sub; gc.collect()
        del order; gc.collect()
        return result

    def _load_coords(self, input_path):
        file_mb = os.path.getsize(input_path) / 1024 / 1024
        print(f"  加载点云: {input_path} ({file_mb:.0f}MB)", flush=True)
        print(f"  读取中，请等待...", flush=True)
        t0 = time.time()
        if input_path.endswith(".pcd"):
            pcd = o3d.io.read_point_cloud(input_path)
            coords = np.asarray(pcd.points, dtype=np.float32)
            del pcd; gc.collect()
        elif input_path.endswith(".las"):
            las = laspy.read(input_path)
            coords = np.vstack([las.X, las.Y, las.Z]).T.astype(np.float32) * 0.001
            del las; gc.collect()
        print(f"  加载完成: {len(coords):,}点 ({time.time()-t0:.1f}s, "
              f"{len(coords)/1024/1024/time.time()-t0:.0f}M点/s)", flush=True)
        return coords

    def step1_split(self, input_path, working_dir):
        print(f"\n{'='*60}", flush=True)
        print(f"步骤1: 空间分块 (CPU分桶)", flush=True)
        print(f"{'='*60}", flush=True)
        os.makedirs(working_dir, exist_ok=True)

        meta_path = os.path.join(working_dir, "meta.npy")
        if os.path.exists(meta_path):
            m = np.load(meta_path, allow_pickle=True).item()
            print(f"  已存在: {m['block_count']}块, 跳过", flush=True)
            return m

        t0 = time.time()
        coords = self._load_coords(input_path)
        n = len(coords)

        coords_path = os.path.join(working_dir, "_coords.npy")
        print(f"  保存坐标到 {coords_path}...", flush=True)
        np.save(coords_path, coords)
        del coords; gc.collect()

        coords = np.load(coords_path, mmap_mode="r")
        min_c = np.array([coords[:, 0].min(), coords[:, 1].min(), coords[:, 2].min()], dtype=np.float64)
        max_c = np.array([coords[:, 0].max(), coords[:, 1].max(), coords[:, 2].max()], dtype=np.float64)
        ranges = max_c - min_c

        bs = self.block_size
        ov = self.overlap
        stride = bs - ov
        gx = max(1, int(np.ceil(ranges[0] / stride)))
        gy = max(1, int(np.ceil(ranges[1] / stride)))
        gz = 1
        total_boxes = gx * gy * gz
        print(f"  范围: {ranges[0]:.0f}x{ranges[1]:.0f}x{ranges[2]:.0f}m", flush=True)
        print(f"  网格: {gx}x{gy}x{gz}={total_boxes}块 | block={bs}m overlap={ov}m stride={stride}m", flush=True)

        half_bs = bs / 2.0
        half_ov = half_bs + ov

        print(f"  预排序X轴 (np.argsort)...", flush=True)
        t_sort = time.time()
        x_all = np.array(coords[:, 0], dtype=np.float32)
        order_x = np.argsort(x_all)
        x_sorted = x_all[order_x]
        del x_all; gc.collect()
        print(f"  X轴排序完成 {time.time()-t_sort:.1f}s", flush=True)

        print(f"  CPU分桶中 (searchsorted加速)...", flush=True)
        bc = 0
        t_bucket = time.time()
        for ix in range(gx):
            cx = min_c[0] + ix * stride + half_bs
            x_lo, x_hi = cx - half_ov, cx + half_ov
            il = np.searchsorted(x_sorted, x_lo, side="left")
            ir = np.searchsorted(x_sorted, x_hi, side="left")
            if ir <= il:
                continue
            cand_idx = order_x[il:ir]
            cand_x = x_sorted[il:ir]
            cand_y = np.array(coords[cand_idx, 1], dtype=np.float32)
            cand_z = np.array(coords[cand_idx, 2], dtype=np.float32)

            for iy in range(gy):
                cy = min_c[1] + iy * stride + half_bs
                y_lo, y_hi = cy - half_ov, cy + half_ov
                ym = (cand_y >= y_lo) & (cand_y < y_hi)
                if not ym.any():
                    continue
                sub_idx = cand_idx[ym]

                if len(sub_idx) < 10:
                    continue
                np.save(os.path.join(working_dir, f"block_{bc:06d}.npy"), sub_idx)
                bc += 1

            elapsed = time.time() - t_bucket
            eta = (elapsed / (ix + 1)) * (gx - ix - 1) if ix < gx - 1 else 0
            mem = os.popen("free -h").read().split("\n")[1].split()[2]
            print(f"    X行 {ix+1}/{gx} | 有效块 {bc} | "
                  f"{elapsed:.0f}s ETA {eta:.0f}s | RAM {mem}", flush=True)

        del coords, order_x, x_sorted, cand_y, cand_z; gc.collect()

        m = {"n_total": n, "min_c": min_c.tolist(), "ranges": ranges.tolist(), "block_count": bc}
        np.save(meta_path, m)
        print(f"  分块完成: {bc}块 (跳过空块 {total_boxes - bc}) | 总计 {time.time()-t0:.1f}s", flush=True)
        return m

    def step2_infer(self, working_dir):
        print(f"\n{'='*60}", flush=True)
        print(f"步骤2: 逐块推理 (断点续跑)", flush=True)
        print(f"{'='*60}", flush=True)

        blocks = sorted(glob.glob(os.path.join(working_dir, "block_*.npy")))
        coords_path = os.path.join(working_dir, "_coords.npy")
        if not os.path.exists(coords_path):
            print("  错误: 需要先运行步骤1", flush=True)
            return

        coords = np.load(coords_path, mmap_mode="r")
        done = set(f.replace(".npy", "_pred.npy") for f in blocks
                    if os.path.exists(f.replace(".npy", "_pred.npy")))
        todo = [f for f in blocks if f not in done]

        if not todo:
            print(f"  全部 {len(blocks)} 块已完成", flush=True)
            return

        print(f"  总计 {len(blocks)} 块, 已完成 {len(done)}, 待推理 {len(todo)}", flush=True)
        t0 = time.time()
        for bi, bf in enumerate(todo):
            pf = bf.replace(".npy", "_pred.npy")
            idx = np.load(bf)
            block_coords = np.array(coords[idx], dtype=np.float32)
            del idx
            n_pts = len(block_coords)
            try:
                preds = self.infer_block_safe(block_coords)
            except torch.cuda.OutOfMemoryError:
                print(f"  [{bi+1}/{len(todo)}] OOM! 清理重试 (max_pts=200000)...", flush=True)
                torch.cuda.empty_cache()
                gc.collect()
                preds = self.infer_block_safe(block_coords, max_pts=200000)
            del block_coords; gc.collect()
            np.save(pf, preds.astype(np.uint8))
            del preds; gc.collect()
            elapsed = time.time() - t0
            eta = (elapsed / (bi+1)) * (len(todo) - bi - 1)
            mem = os.popen("free -h").read().split("\n")[1].split()[2]
            gpu_mem = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader").read().strip()
            print(f"  [{bi+1}/{len(todo)}] {n_pts:,}点 | "
                  f"{elapsed:.0f}s ETA {eta:.0f}s({time.strftime('%H:%M:%S', time.gmtime(eta))}) | "
                  f"RAM {mem} GPU {gpu_mem}", flush=True)
        print(f"  推理完成: {time.time()-t0:.1f}s", flush=True)

    def step3_merge(self, working_dir, output_path):
        print(f"\n{'='*60}", flush=True)
        print(f"步骤3: 合并写 LAS", flush=True)
        print(f"{'='*60}", flush=True)

        meta = np.load(os.path.join(working_dir, "meta.npy"), allow_pickle=True).item()
        n = meta["n_total"]
        min_c = np.array(meta["min_c"], dtype=np.float64)
        bc = meta["block_count"]

        coords = np.load(os.path.join(working_dir, "_coords.npy"), mmap_mode="r")

        preds = np.zeros(n, dtype=np.uint8)
        count = np.zeros(n, dtype=np.uint8)
        valid_mask = np.zeros(n, dtype=bool)

        t0 = time.time()
        for bi in range(bc):
            bp = os.path.join(working_dir, f"block_{bi:06d}.npy")
            pp = os.path.join(working_dir, f"block_{bi:06d}_pred.npy")
            if not os.path.exists(pp):
                continue
            idx = np.load(bp)
            p = np.load(pp).astype(np.uint8)
            for c in range(self.num_classes):
                mask_c = (p == c)
                count[idx[mask_c]] += 1
            preds[idx] = p
            valid_mask[idx] = True
            del idx, p; gc.collect()
            if (bi+1) % 50 == 0 or bi == bc - 1:
                overlapped = (count > 1).sum()
                print(f"  合并: {bi+1}/{bc} | 覆盖 {valid_mask.sum():,}/{n:,} | 重叠点 {overlapped:,} | {time.time()-t0:.0f}s", flush=True)

        print(f"  重叠区域投票中...", flush=True)
        overlap_mask = count > 1
        n_overlap = overlap_mask.sum()
        print(f"    重叠点: {n_overlap:,}", flush=True)
        if n_overlap > 0:
            overlap_pred = np.zeros(n_overlap, dtype=np.int32)
            overlap_idx = np.where(overlap_mask)[0]
            overlap_idx.sort()
            overlap_vote = np.zeros((n_overlap, self.num_classes), dtype=np.int32)
            for bi in range(bc):
                bp = os.path.join(working_dir, f"block_{bi:06d}.npy")
                pp = os.path.join(working_dir, f"block_{bi:06d}_pred.npy")
                if not os.path.exists(pp):
                    continue
                idx = np.load(bp)
                p = np.load(pp).astype(np.uint8)
                local_overlap = overlap_mask[idx]
                if not local_overlap.any():
                    del idx, p; gc.collect()
                    continue
                global_pos = np.searchsorted(overlap_idx, idx[local_overlap])
                overlap_vote[global_pos, p[idx[local_overlap]]] += 1
                del idx, p; gc.collect()
                if (bi+1) % 25 == 0 or bi == bc - 1:
                    print(f"    投票进度: {bi+1}/{bc}", flush=True)
            overlap_pred = np.argmax(overlap_vote, axis=1)
            preds[overlap_idx] = overlap_pred.astype(np.uint8)
            del overlap_pred, overlap_vote, overlap_idx
        vmask = valid_mask
        final = preds[vmask]
        n_final = len(final)
        del preds, count, valid_mask; gc.collect()

        print(f"  写 LAS ({n_final:,}点)...", flush=True)
        header = laspy.LasHeader(version="1.4", point_format=6)
        header.offsets = min_c
        header.scales = [0.001, 0.001, 0.001]
        out = laspy.LasData(header)
        out.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.uint8))
        out.x = np.array(coords[vmask, 0], dtype=np.float64)
        out.y = np.array(coords[vmask, 1], dtype=np.float64)
        out.z = np.array(coords[vmask, 2], dtype=np.float64)
        out["label"] = final
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out.write(output_path)
        del out, coords; gc.collect()

        print(f"\n  完成: {time.time()-t0:.1f}s", flush=True)
        print(f"  保存: {output_path}", flush=True)
        for c in range(self.num_classes):
            cnt = (final == c).sum()
            if cnt > 0:
                print(f"    Class {c}: {cnt:,} ({cnt/n_final*100:.1f}%)")
        del final

    def run_single(self, input_path, output_path):
        name = os.path.splitext(os.path.basename(input_path))[0]
        wd = os.path.join(os.path.dirname(output_path) or ".", f"{name}_working")

        meta = self.step1_split(input_path, wd)
        self.step2_infer(wd)
        self.step3_merge(wd, output_path)

        print(f"\n  清理临时文件: {wd}", flush=True)
        shutil.rmtree(wd, ignore_errors=True)

    def run(self, input_path, output_path):
        files = []
        if os.path.isdir(input_path):
            for ext in ("*.pcd", "*.las"):
                files.extend(glob.glob(os.path.join(input_path, ext)))
        else:
            files = [input_path]
        files.sort(key=lambda f: os.path.getsize(f))
        print(f"找到 {len(files)} 个文件:", flush=True)
        for f in files:
            print(f"  {os.path.getsize(f)/1024/1024:.1f}MB - {os.path.basename(f)}", flush=True)
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            out = output_path if os.path.isfile(input_path) else os.path.join(output_path, f"{name}_pred.las")
            if os.path.exists(out):
                print(f"跳过已完成: {out}", flush=True)
                continue
            print(f"\n{'='*60}\n处理: {f}\n{'='*60}", flush=True)
            self.run_single(f, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weight", required=True)
    p.add_argument("--grid-size", type=float, default=0.02)
    p.add_argument("--block-size", type=float, default=20.0)
    p.add_argument("--overlap", type=float, default=2.0)
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    Inference(
        weight_path=args.weight, grid_size=args.grid_size,
        num_classes=args.num_classes, block_size=args.block_size,
        overlap=args.overlap, use_tta=not args.no_tta, device=args.device,
    ).run(args.input, args.output)
