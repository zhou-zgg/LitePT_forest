#!/usr/bin/env python
"""wj old model batch inference - Small model + grid_size=0.02"""

import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from collections import OrderedDict
from models.builder import build_model
from tools.block_inference import Inference as _Inference

WEIGHT = "/root/autodl-tmp/workshop/wj/Forest_survey/model/model_best0618.pth"
GRID_SIZE = 0.02
OUT_ROOT = "/root/autodl-tmp/workshop/wj/Forest_survey/inference"

INPUT_FILES = [
    "/root/autodl-tmp/dataset/tree/snag/quickmap1_20260110_152658.pcd",
    "/root/autodl-tmp/dataset/tree/client/0616/a/geomap_20260616_093006.pcd",
    "/root/autodl-tmp/dataset/tree/client/0616/b/geomap_20260616_101448.pcd",
    "/root/autodl-tmp/dataset/tree/client/0618/a/geomap_20260618_150830.pcd",
    "/root/autodl-tmp/dataset/tree/client/0618/b/geomap_20260618_153707.pcd",
    "/root/autodl-tmp/dataset/tree/client/1+2+3/1/1.pcd",
    "/root/autodl-tmp/dataset/tree/client/1+2+3/2/2.pcd",
    "/root/autodl-tmp/dataset/tree/client/1+2+3/3/3.pcd",
]


class WJInference(_Inference):
    def _build_model(self):
        model_cfg = dict(
            type="DefaultSegmentorV2",
            num_classes=self.num_classes,
            backbone_out_channels=72,
            backbone=dict(
                type="LitePT", in_channels=3,
                order=("z", "z-trans", "hilbert", "hilbert-trans"),
                stride=(2, 2, 2, 2),
                enc_depths=(2, 2, 2, 6, 2),
                enc_channels=(36, 72, 144, 252, 504),
                enc_num_head=(2, 4, 8, 14, 28),
                enc_patch_size=(1024, 1024, 1024, 2048, 2048),
                enc_conv=(True, True, True, False, False),
                enc_attn=(False, False, False, True, True),
                enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
                dec_depths=(0, 0, 0, 0),
                dec_channels=(72, 72, 144, 252),
                dec_num_head=(4, 4, 8, 14),
                dec_patch_size=(1024, 1024, 1024, 1024),
                dec_conv=(False, False, False, False),
                dec_attn=(False, False, False, False),
                dec_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
                mlp_ratio=4, qkv_bias=True, qk_scale=None,
                attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
                shuffle_orders=True, pre_norm=True, enc_mode=False,
            ),
            criteria=[],
        )
        self.model = build_model(model_cfg).to(self.device)
        print("Loading wj weights: " + self.weight_path, flush=True)
        ckpt = torch.load(self.weight_path, map_location=self.device, weights_only=False)
        sd = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            sd[k[7:] if k.startswith("module.") else k] = v
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        nt = sum(p.numel() for p in self.model.parameters())
        ep = ckpt["epoch"]
        print("wj model epoch " + str(ep) + " | params: " + format(nt, ",") + " | " + str(self.device), flush=True)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    sep = "=" * 60
    print("wj Batch Inference", flush=True)
    print("Model: " + WEIGHT, flush=True)
    print("grid_size: " + str(GRID_SIZE), flush=True)
    print("Output: " + OUT_ROOT, flush=True)
    print("Files: " + str(len(INPUT_FILES)), flush=True)

    for i, f in enumerate(INPUT_FILES):
        if not os.path.exists(f):
            print("\nFile not found, skip: " + f, flush=True)
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(OUT_ROOT, name + "_pred.las")
        if os.path.exists(out):
            print("[" + str(i+1) + "/" + str(len(INPUT_FILES)) + "] Skip (exists): " + out, flush=True)
            continue
        print("\n" + sep + "\n[" + str(i+1) + "/" + str(len(INPUT_FILES)) + "] Inference: " + name + "\n" + sep, flush=True)
        inf = WJInference(
            weight_path=WEIGHT, grid_size=GRID_SIZE,
            num_classes=7, block_size=20.0, overlap=2.0,
            use_tta=True, device="cuda",
        )
        inf.run_single(f, out)
        del inf
        torch.cuda.empty_cache()
        gc.collect()

    print("\nDone! Results: " + OUT_ROOT, flush=True)


if __name__ == "__main__":
    main()
