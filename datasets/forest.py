import os
import laspy
import numpy as np
import open3d as o3d

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ForestDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "segment",
    ]

    def __init__(self, class_mapping=None, pred_check_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.class_mapping = class_mapping
        self.pred_check_dir = pred_check_dir

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
            coord = np.vstack([las.X, las.Y, las.Z]).T.astype(np.float32) * 0.001
            if "label" in [d for d in las.point_format.dimension_names]:
                segment = np.array(las.label, dtype=np.int32).reshape([-1])
            else:
                segment = np.array(las.classification, dtype=np.int32).reshape([-1])

            if self.class_mapping is not None:
                mapping = lambda x: self.class_mapping.get(x, x)
                segment = np.vectorize(mapping, otypes=[np.int32])(segment)

            data_dict = {
                "coord": coord,
                "segment": segment,
                "name": name,
                "split": split,
            }

        return data_dict