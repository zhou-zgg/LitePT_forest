import json
import os
import laspy
import numpy as np
import open3d as o3d
import torch

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ForestDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "segment",
    ]

    def __init__(self, class_mapping=None, pred_check_dir=None,
                 oversample=False, oversample_exclude_classes=None,
                 ignore_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.class_mapping = class_mapping
        self.pred_check_dir = pred_check_dir
        # 不参与学习的类别（如 CWD）：标签映射为 ignore_index(-1)，
        # 损失与指标均跳过，但模型输出仍保留该类别通道（推理兼容）。
        self.ignore_classes = list(ignore_classes) if ignore_classes else None

        if oversample and not self.test_mode:
            self.sample_weights = self._compute_sample_weights(
                oversample_exclude_classes or []
            )
        else:
            self.sample_weights = None

    @staticmethod
    def compute_loss_weights(data_root, split, num_classes=7,
                             exclude_classes=None, method="inverse_sqrt"):
        cache_file = os.path.join(data_root, f"{os.path.basename(split)}_class_hist.json")
        if not os.path.exists(cache_file):
            return [1.0] * num_classes
        with open(cache_file, "r") as f:
            cached = json.load(f)
        global_counts = np.zeros(num_classes, dtype=np.float64)
        for fname, fhist in cached.get("hist", {}).items():
            for k, v in fhist.items():
                c = int(k)
                if 0 <= c < num_classes:
                    global_counts[c] += v
        if exclude_classes:
            for c in exclude_classes:
                global_counts[c] = 0
        total = global_counts.sum()
        if total == 0:
            return [1.0] * num_classes
        freqs = global_counts / total
        raw = np.ones(num_classes)
        if method == "inverse_sqrt":
            mask = freqs > 0
            raw[mask] = 1.0 / np.sqrt(freqs[mask])
        elif method == "inverse":
            mask = freqs > 0
            raw[mask] = 1.0 / freqs[mask]
        if exclude_classes:
            for c in exclude_classes:
                raw[c] = 1.0
        raw /= raw[0]
        return raw.tolist()

    def _scan_class_hist(self):
        hist = {}
        for fname in self.data_list:
            if fname.endswith(".las"):
                las = laspy.read(fname)
                if "label" in [d for d in las.point_format.dimension_names]:
                    labels = np.array(las.label, dtype=np.int32)
                else:
                    labels = np.array(las.classification, dtype=np.int32)
                if self.class_mapping is not None:
                    mapping = lambda x: self.class_mapping.get(x, x)
                    labels = np.vectorize(mapping, otypes=[np.int32])(labels)
                cls, cnt = np.unique(labels, return_counts=True)
                hist[os.path.basename(fname)] = {int(c): int(n) for c, n in zip(cls, cnt)}
        return hist

    def _load_or_scan_class_hist(self):
        cache_dir = self.data_root
        cache_file = os.path.join(cache_dir, f"{os.path.basename(self.split)}_class_hist.json")
        current_files = sorted([os.path.basename(p) for p in self.data_list])

        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if (sorted(cached.get("files", [])) == current_files
                    and "hist" in cached):
                hist = {}
                for fname, fhist in cached["hist"].items():
                    hist[fname] = {int(k): v for k, v in fhist.items()}
                return hist

        hist = self._scan_class_hist()
        cache_data = {"files": current_files, "hist": hist}
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
        return hist

    def _compute_sample_weights(self, exclude_classes):
        file_hist = self._load_or_scan_class_hist()
        num_files = len(self.data_list)

        class_files = {}
        for fname, hist in file_hist.items():
            for c in hist:
                if c >= 0 and c not in exclude_classes:
                    class_files[c] = class_files.get(c, 0) + 1

        max_cls = max(class_files.keys(), default=0)
        file_rarity = np.ones(max_cls + 1)
        for c in range(max_cls + 1):
            if c in class_files and class_files[c] > 0:
                file_rarity[c] = num_files / class_files[c]

        weights = np.ones(num_files)
        for i, fname in enumerate(self.data_list):
            key = os.path.basename(fname)
            hist = file_hist.get(key, {})
            max_rarity = 1.0
            for c, _ in hist.items():
                if c >= 0 and c < len(file_rarity) and c not in exclude_classes:
                    max_rarity = max(max_rarity, file_rarity[c])
            weights[i] = max_rarity

        weights /= weights.mean()
        sample_weights = np.tile(weights, self.loop)
        return torch.as_tensor(sample_weights, dtype=torch.double)

    def get_data_list(self):
        data_list = super().get_data_list()
        return [p for p in data_list if p.endswith(".las") or p.endswith(".pcd")]

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)

        if self.pred_check_dir is not None:
            pred_file = os.path.join(self.pred_check_dir, f"{name}_pred.npy")
            if os.path.exists(pred_file):
                las = laspy.read(data_path)
                if "label" in [d for d in las.point_format.dimension_names]:
                    segment = np.array(las.label, dtype=np.int32).reshape([-1])
                else:
                    segment = np.array(las.classification, dtype=np.int32).reshape([-1])
                if self.class_mapping is not None:
                    mapping = lambda x: self.class_mapping.get(x, x)
                    segment = np.vectorize(mapping, otypes=[np.int32])(segment)
                return dict(
                    name=name,
                    segment=segment,
                    fast_skip=True,
                )

        if data_path.endswith(".pcd"):
            pcd = o3d.io.read_point_cloud(data_path)
            points = np.asarray(pcd.points).astype(np.float32)
            data_dict = {
                "coord": points,
                "segment": np.zeros(len(points), dtype=np.int32),
                "name": name,
                "split": split,
            }
        else:
            las = laspy.read(data_path)
            # 语义：LAS 真实坐标 = 整数 * scale + offset。这里取整数 * scale，
            # 即以该文件 LAS 原点为基准的相对坐标（真实坐标减去 offset 后的结果），
            # 从而把经纬度等绝对基准抵消掉，模型拿到的是规整的局部坐标。
            # 使用 las.header.scales 而非写死的 0.001，兼容不同 scale 的 las 文件。
            scale = np.asarray(las.header.scales, dtype=np.float32)
            coord = (
                np.vstack([las.X, las.Y, las.Z]).T.astype(np.float32) * scale
            )
            if "label" in [d for d in las.point_format.dimension_names]:
                segment = np.array(las.label, dtype=np.int32).reshape([-1])
            else:
                segment = np.array(las.classification, dtype=np.int32).reshape([-1])

            if self.class_mapping is not None:
                mapping = lambda x: self.class_mapping.get(x, x)
                segment = np.vectorize(mapping, otypes=[np.int32])(segment)

            if self.ignore_classes:
                segment[np.isin(segment, self.ignore_classes)] = -1

            data_dict = {
                "coord": coord,
                "segment": segment,
                "name": name,
                "split": split,
            }

        return data_dict