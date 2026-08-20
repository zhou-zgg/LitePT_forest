import torch
import numpy as np
import random
import copy
from collections.abc import Sequence, Mapping

from datasets.transform import TRANSFORMS, index_operator


def _arr2tensor(arr, device="cuda"):
    if isinstance(arr, torch.Tensor):
        return arr.to(device)
    if isinstance(arr, np.ndarray):
        if np.issubdtype(arr.dtype, np.bool_):
            return torch.from_numpy(arr).to(device)
        elif np.issubdtype(arr.dtype, np.integer):
            return torch.from_numpy(arr).long().to(device)
        elif np.issubdtype(arr.dtype, np.floating):
            return torch.from_numpy(arr).float().to(device)
    return arr


def _tensor2arr(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype in (torch.bool,):
            return tensor.cpu().numpy()
        elif tensor.dtype in (torch.int32, torch.int64, torch.long):
            return tensor.cpu().to(torch.int32).numpy()
        else:
            return tensor.cpu().numpy()
    return tensor


@TRANSFORMS.register_module()
class CenterShiftCUDA:
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord = _arr2tensor(data_dict["coord"])
            x_min, y_min, z_min = coord.min(dim=0)[0]
            x_max, y_max, _ = coord.max(dim=0)[0]
            if self.apply_z:
                shift = torch.tensor(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min],
                    device=coord.device,
                )
            else:
                shift = torch.tensor(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, 0.0],
                    device=coord.device,
                )
            coord = coord - shift
            data_dict["coord"] = _tensor2arr(coord)
        return data_dict


@TRANSFORMS.register_module()
class PositiveShiftCUDA:
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord = _arr2tensor(data_dict["coord"])
            coord = coord - coord.min(dim=0)[0]
            data_dict["coord"] = _tensor2arr(coord)
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoordCUDA:
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord = _arr2tensor(data_dict["coord"])
            centroid = coord.mean(dim=0)
            coord = coord - centroid
            m = torch.sqrt((coord ** 2).sum(dim=1)).max()
            coord = coord / m
            data_dict["coord"] = _tensor2arr(coord)
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColorCUDA:
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = _arr2tensor(data_dict["color"]).cpu().numpy() / 255
            data_dict["color"] = data_dict["color"].astype(np.float32)
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateCUDA:
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = torch.tensor(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=torch.float32,
            )
        elif self.axis == "y":
            rot_t = torch.tensor(
                [[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]],
                dtype=torch.float32,
            )
        elif self.axis == "z":
            rot_t = torch.tensor(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rot_t = rot_t.to(device)
        if "coord" in data_dict.keys():
            coord = _arr2tensor(data_dict["coord"], device)
            if self.center is None:
                x_min, y_min, z_min = coord.min(dim=0)[0]
                x_max, y_max, z_max = coord.max(dim=0)[0]
                center = torch.tensor(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2],
                    device=device,
                )
            else:
                center = torch.tensor(self.center, device=device, dtype=torch.float32)
            coord = coord - center
            coord = coord @ rot_t.T
            coord = coord + center
            data_dict["coord"] = _tensor2arr(coord)
        if "normal" in data_dict.keys():
            normal = _arr2tensor(data_dict["normal"], device)
            normal = normal @ rot_t.T
            data_dict["normal"] = _tensor2arr(normal)
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngleCUDA:
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = torch.tensor(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=torch.float32,
            )
        elif self.axis == "y":
            rot_t = torch.tensor(
                [[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]],
                dtype=torch.float32,
            )
        elif self.axis == "z":
            rot_t = torch.tensor(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            raise NotImplementedError
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rot_t = rot_t.to(device)
        if "coord" in data_dict.keys():
            coord = _arr2tensor(data_dict["coord"], device)
            if self.center is None:
                x_min, y_min, z_min = coord.min(dim=0)[0]
                x_max, y_max, z_max = coord.max(dim=0)[0]
                center = torch.tensor(
                    [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2],
                    device=device,
                )
            else:
                center = torch.tensor(self.center, device=device, dtype=torch.float32)
            coord = coord - center
            coord = coord @ rot_t.T
            coord = coord + center
            data_dict["coord"] = _tensor2arr(coord)
        if "normal" in data_dict.keys():
            normal = _arr2tensor(data_dict["normal"], device)
            normal = normal @ rot_t.T
            data_dict["normal"] = _tensor2arr(normal)
        return data_dict


@TRANSFORMS.register_module()
class RandomScaleCUDA:
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            scale = torch.tensor(
                np.random.uniform(
                    self.scale[0], self.scale[1], 3 if self.anisotropic else 1
                ),
                device=device,
                dtype=torch.float32,
            )
            coord = _arr2tensor(data_dict["coord"], device)
            coord = coord * scale
            data_dict["coord"] = _tensor2arr(coord)
        return data_dict


@TRANSFORMS.register_module()
class RandomFlipCUDA:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                coord = _arr2tensor(data_dict["coord"], device)
                coord[:, 0] = -coord[:, 0]
                data_dict["coord"] = _tensor2arr(coord)
            if "normal" in data_dict.keys():
                normal = _arr2tensor(data_dict["normal"], device)
                normal[:, 0] = -normal[:, 0]
                data_dict["normal"] = _tensor2arr(normal)
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                coord = _arr2tensor(data_dict["coord"], device)
                coord[:, 1] = -coord[:, 1]
                data_dict["coord"] = _tensor2arr(coord)
            if "normal" in data_dict.keys():
                normal = _arr2tensor(data_dict["normal"], device)
                normal[:, 1] = -normal[:, 1]
                data_dict["normal"] = _tensor2arr(normal)
        return data_dict


@TRANSFORMS.register_module()
class RandomJitterCUDA:
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            coord = _arr2tensor(data_dict["coord"], device)
            jitter = torch.randn(coord.shape[0], 3, device=device) * self.sigma
            jitter = jitter.clamp(-self.clip, self.clip)
            coord = coord + jitter
            data_dict["coord"] = _tensor2arr(coord)
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitterCUDA:
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            coord = _arr2tensor(data_dict["coord"], device)
            jitter = torch.randn(coord.shape[0], 3, device=device)
            jitter = self.scalar * jitter.clamp(-1, 1)
            coord = coord + jitter
            data_dict["coord"] = _tensor2arr(coord)
            if self.store_jitter:
                data_dict["jitter"] = _tensor2arr(jitter)
        return data_dict


@TRANSFORMS.register_module()
class RandomDropoutCUDA:
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            n = len(data_dict["coord"])
            keep_count = int(n * (1 - self.dropout_ratio))
            idx = torch.randperm(n, device=device)[:keep_count].sort()[0].cpu().numpy()
            if "sampled_index" in data_dict:
                sampled_idx = torch.from_numpy(data_dict["sampled_index"]).to(device)
                idx_tensor = torch.from_numpy(idx).to(device)
                idx_tensor = torch.unique(torch.cat([idx_tensor, sampled_idx])).cpu().numpy()
                mask = np.zeros(data_dict["segment"].shape[0], dtype=bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_tensor])[0]
                idx = idx_tensor
            data_dict = index_operator(data_dict, idx)
        return data_dict


@TRANSFORMS.register_module()
class PointClipCUDA:
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            coord = _arr2tensor(data_dict["coord"], device)
            rmin = torch.tensor(self.point_cloud_range[:3], device=device, dtype=torch.float32)
            rmax = torch.tensor(self.point_cloud_range[3:], device=device, dtype=torch.float32)
            coord = coord.clamp(min=rmin, max=rmax)
            data_dict["coord"] = _tensor2arr(coord)
        return data_dict


@TRANSFORMS.register_module()
class GridSampleCUDA:
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash_type = hash_type
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def _hash_cuda(self, grid_coord):
        return self._ravel_hash_cuda(grid_coord)

    @staticmethod
    def _ravel_hash_cuda(grid_coord):
        coord = grid_coord.clone()
        min_vals = coord.min(dim=0)[0]
        coord = coord - min_vals
        max_vals = coord.max(dim=0)[0] + 1
        keys = coord[:, 0].clone()
        for j in range(1, coord.shape[1]):
            keys = keys * max_vals[j] + coord[:, j]
        return keys

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        grid_size = torch.tensor(self.grid_size, device=device, dtype=torch.float32)

        coord = _arr2tensor(data_dict["coord"], device)
        scaled_coord = coord / grid_size
        grid_coord = scaled_coord.floor().long()
        min_coord = grid_coord.min(dim=0)[0]
        grid_coord = grid_coord - min_coord
        scaled_coord = scaled_coord - min_coord.float()
        min_coord_float = (min_coord.float() * grid_size).cpu().numpy()

        key = self._hash_cuda(grid_coord)

        idx_sort = torch.argsort(key)

        key_sort = key[idx_sort]
        unique_key, inverse, count = torch.unique(
            key_sort, return_inverse=True, return_counts=True
        )

        if self.mode == "train":
            cumsum = torch.cumsum(
                torch.cat([torch.tensor([0], device=device), count[:-1]]), dim=0
            )
            idx_select = cumsum + torch.rand(count.shape[0], device=device).mul(count.float()).long()
            idx_unique = idx_sort[idx_select]

            if "sampled_index" in data_dict:
                sampled_index = torch.from_numpy(data_dict["sampled_index"]).long().to(device)
                idx_unique = torch.unique(torch.cat([idx_unique, sampled_index]))
                mask = torch.zeros(data_dict["segment"].shape[0], dtype=torch.bool, device=device)
                mask[torch.from_numpy(data_dict["sampled_index"]).long().to(device)] = True
                data_dict["sampled_index"] = torch.where(mask[idx_unique])[0].cpu().numpy()

            idx_unique_np = idx_unique.cpu().numpy()
            data_dict = index_operator(data_dict, idx_unique_np)
            data_dict["index"] = idx_unique_np

            if self.return_inverse:
                data_dict["inverse"] = np.zeros(inverse.shape[0], dtype=inverse.cpu().numpy().dtype)
                data_dict["inverse"][idx_sort.cpu().numpy()] = inverse.cpu().numpy()
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique].cpu().numpy()
                data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord_float.reshape([1, 3])
            if self.return_displacement:
                displacement = scaled_coord - grid_coord.float() - 0.5
                if self.project_displacement and "normal" in data_dict:
                    normal = _arr2tensor(data_dict["normal"], device)
                    displacement = (displacement * normal).sum(dim=-1, keepdim=True)
                data_dict["displacement"] = displacement[idx_unique].cpu().numpy()
                data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":
            data_part_list = []
            count_max = count.max().item()
            cumsum = torch.cumsum(
                torch.cat([torch.tensor([0], device=device), count[:-1]]), dim=0
            )
            for i in range(count_max):
                idx_select = cumsum + i % count
                idx_part = idx_sort[idx_select]
                idx_part_np = idx_part.cpu().numpy()
                data_part = index_operator(data_dict, idx_part_np, duplicate=True)
                data_part["index"] = idx_part_np
                if self.return_inverse:
                    data_part["inverse"] = np.zeros(inverse.shape[0], dtype=inverse.cpu().numpy().dtype)
                    data_part["inverse"][idx_sort.cpu().numpy()] = inverse.cpu().numpy()
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part].cpu().numpy()
                    data_dict["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord_float.reshape([1, 3])
                if self.return_displacement:
                    displacement = scaled_coord - grid_coord.float() - 0.5
                    if self.project_displacement and "normal" in data_dict:
                        normal = _arr2tensor(data_dict["normal"], device)
                        displacement = (displacement * normal).sum(dim=-1, keepdim=True)
                    data_dict["displacement"] = displacement[idx_part].cpu().numpy()
                    data_dict["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError


@TRANSFORMS.register_module()
class ElasticDistortionCUDA:
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def _elastic_distortion_gpu(coords, granularity, magnitude, device):
        coords_min = coords.min(dim=0)[0]
        coords_range = coords.max(dim=0)[0] - coords_min
        noise_dim = (coords_range / granularity).long() + 3
        noise_dim_list = noise_dim.tolist()

        noise = torch.randn(*noise_dim_list, 3, device=device, dtype=torch.float32)

        noise_5d = noise.permute(3, 0, 1, 2).unsqueeze(0)

        kd = torch.ones(3, 1, 3, 1, 1, device=device, dtype=torch.float32) / 3.0
        kh = torch.ones(3, 1, 1, 3, 1, device=device, dtype=torch.float32) / 3.0
        kw = torch.ones(3, 1, 1, 1, 3, device=device, dtype=torch.float32) / 3.0

        for _ in range(2):
            noise_5d = torch.nn.functional.conv3d(noise_5d, kd, groups=3, padding=(1, 0, 0))
            noise_5d = torch.nn.functional.conv3d(noise_5d, kh, groups=3, padding=(0, 1, 0))
            noise_5d = torch.nn.functional.conv3d(noise_5d, kw, groups=3, padding=(0, 0, 1))

        noise_smooth = noise_5d.squeeze(0).permute(1, 2, 3, 0)

        D, H, W = noise_dim_list
        grid_origin = coords_min - granularity
        coords_normalized = (coords - grid_origin) / granularity

        x0 = coords_normalized[:, 0].clamp(0, D - 1 - 1e-6)
        y0 = coords_normalized[:, 1].clamp(0, H - 1 - 1e-6)
        z0 = coords_normalized[:, 2].clamp(0, W - 1 - 1e-6)

        ix = x0.floor().long().clamp(0, D - 2)
        iy = y0.floor().long().clamp(0, H - 2)
        iz = z0.floor().long().clamp(0, W - 2)

        fx = x0 - ix.float()
        fy = y0 - iy.float()
        fz = z0 - iz.float()

        ix1 = ix + 1
        iy1 = iy + 1
        iz1 = iz + 1

        c000 = noise_smooth[ix, iy, iz]
        c001 = noise_smooth[ix, iy, iz1]
        c010 = noise_smooth[ix, iy1, iz]
        c011 = noise_smooth[ix, iy1, iz1]
        c100 = noise_smooth[ix1, iy, iz]
        c101 = noise_smooth[ix1, iy, iz1]
        c110 = noise_smooth[ix1, iy1, iz]
        c111 = noise_smooth[ix1, iy1, iz1]

        c00 = c000 * (1 - fz).unsqueeze(1) + c001 * fz.unsqueeze(1)
        c01 = c010 * (1 - fz).unsqueeze(1) + c011 * fz.unsqueeze(1)
        c10 = c100 * (1 - fz).unsqueeze(1) + c101 * fz.unsqueeze(1)
        c11 = c110 * (1 - fz).unsqueeze(1) + c111 * fz.unsqueeze(1)

        c0 = c00 * (1 - fy).unsqueeze(1) + c01 * fy.unsqueeze(1)
        c1 = c10 * (1 - fy).unsqueeze(1) + c11 * fy.unsqueeze(1)

        interp = c0 * (1 - fx).unsqueeze(1) + c1 * fx.unsqueeze(1)

        out_of_bounds = (
            (coords_normalized[:, 0] < 0)
            | (coords_normalized[:, 0] > D - 1)
            | (coords_normalized[:, 1] < 0)
            | (coords_normalized[:, 1] > H - 1)
            | (coords_normalized[:, 2] < 0)
            | (coords_normalized[:, 2] > W - 1)
        )
        interp[out_of_bounds] = 0

        return coords + interp * magnitude

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                coord = _arr2tensor(data_dict["coord"], device)
                for granularity, magnitude in self.distortion_params:
                    gran = torch.tensor(granularity, device=device, dtype=torch.float32)
                    mag = torch.tensor(magnitude, device=device, dtype=torch.float32)
                    coord = self._elastic_distortion_gpu(coord, gran, mag, device)
                data_dict["coord"] = _tensor2arr(coord)
        return data_dict


@TRANSFORMS.register_module()
class SphereCropCUDA:
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            coord = _arr2tensor(data_dict["coord"], device)
            if self.mode == "random":
                center_idx = torch.randint(0, coord.shape[0], (1,), device=device).item()
                center = coord[center_idx]
            elif self.mode == "center":
                center = coord[coord.shape[0] // 2]
            else:
                raise NotImplementedError
            dist = (coord - center.unsqueeze(0)).pow(2).sum(dim=1)
            idx_crop = dist.argsort()[:point_max].cpu().numpy()
            data_dict = index_operator(data_dict, idx_crop)
        return data_dict


@TRANSFORMS.register_module()
class CylinderCropCUDA:
    def __init__(self, point_max=500000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            coord = _arr2tensor(data_dict["coord"], device)
            if self.mode == "random":
                center_idx = torch.randint(0, coord.shape[0], (1,), device=device).item()
                center = coord[center_idx]
            elif self.mode == "center":
                center = coord[coord.shape[0] // 2]
            else:
                raise NotImplementedError

            xy_dist_sq = (coord[:, :2] - center[:2].unsqueeze(0)).pow(2).sum(dim=1)
            idx_crop = xy_dist_sq.argsort()[:point_max].cpu().numpy()

            data_dict = index_operator(data_dict, idx_crop)
        return data_dict


@TRANSFORMS.register_module()
class ShufflePointCUDA:
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        n = data_dict["coord"].shape[0]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        shuffle_index = torch.randperm(n, device=device).cpu().numpy()
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict


@TRANSFORMS.register_module()
class CropBoundaryCUDA:
    def __call__(self, data_dict):
        assert "segment" in data_dict
        device = "cuda" if torch.cuda.is_available() else "cpu"
        segment = _arr2tensor(data_dict["segment"], device)
        mask = (segment != 0) & (segment != 1)
        mask_np = mask.cpu().numpy().astype(bool)
        data_dict = index_operator(data_dict, mask_np)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrastCUDA:
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            color = _arr2tensor(data_dict["color"], device)
            lo = color.min(dim=0, keepdim=True)[0]
            hi = color.max(dim=0, keepdim=True)[0]
            scale = 255.0 / (hi - lo + 1e-6)
            contrast_feat = (color[:, :3] - lo[:, :3]) * scale[:, :3]
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            color[:, :3] = (1 - blend_factor) * color[:, :3] + blend_factor * contrast_feat
            data_dict["color"] = _tensor2arr(color)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrastv2CUDA:
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            color = _arr2tensor(data_dict["color"], device)
            lo = color.min(dim=0, keepdim=True)[0]
            hi = color.max(dim=0, keepdim=True)[0]
            scale = 255.0 / (hi - lo + 1e-6)
            contrast_feat = (color[:, :3] - lo[:, :3]) * scale[:, :3]
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            color[:, :3] = (1 - blend_factor) * color[:, :3] + blend_factor * contrast_feat
            data_dict["color"] = _tensor2arr(color)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslationCUDA:
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            color = _arr2tensor(data_dict["color"], device)
            tr = (torch.rand(1, 3, device=device) - 0.5) * 255 * 2 * self.ratio
            color[:, :3] = (tr + color[:, :3]).clamp(0, 255)
            data_dict["color"] = _tensor2arr(color)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitterCUDA:
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            color = _arr2tensor(data_dict["color"], device)
            noise = torch.randn(color.shape[0], 3, device=device) * (self.std * 255)
            color[:, :3] = (noise + color[:, :3]).clamp(0, 255)
            data_dict["color"] = _tensor2arr(color)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDropCUDA:
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            color = _arr2tensor(data_dict["color"], device)
            color = color * self.color_augment
            data_dict["color"] = _tensor2arr(color)
        return data_dict


@TRANSFORMS.register_module()
class CopyCUDA:
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if key in data_dict:
                if isinstance(data_dict[key], np.ndarray):
                    data_dict[value] = data_dict[key].copy()
                elif isinstance(data_dict[key], torch.Tensor):
                    data_dict[value] = data_dict[key].clone().detach()
                else:
                    data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class UpdateCUDA:
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for key, value in self.keys_dict.items():
            if isinstance(value, (int, float)):
                data_dict[key] = torch.tensor([value], device=device, dtype=torch.float32)
            elif isinstance(value, np.ndarray):
                data_dict[key] = torch.from_numpy(value).to(device)
            elif isinstance(value, torch.Tensor):
                data_dict[key] = value.to(device)
            else:
                data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class RandomShiftCUDA:
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            coord = _arr2tensor(data_dict["coord"], device)
            shift = torch.tensor(
                [
                    np.random.uniform(self.shift[0][0], self.shift[0][1]),
                    np.random.uniform(self.shift[1][0], self.shift[1][1]),
                    np.random.uniform(self.shift[2][0], self.shift[2][1]),
                ],
                device=device,
                dtype=torch.float32,
            )
            coord = coord + shift
            data_dict["coord"] = _tensor2arr(coord)
        return data_dict


@TRANSFORMS.register_module()
class ToTensorCUDA:
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data.cuda() if data.is_cuda else data.to("cuda")
        elif isinstance(data, str):
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data]).cuda()
        elif isinstance(data, float):
            return torch.FloatTensor([data]).cuda()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data).cuda()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long().cuda()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float().cuda()
        elif isinstance(data, Mapping):
            return {sub_key: self(item) for sub_key, item in data.items()}
        elif isinstance(data, Sequence):
            return [self(item) for item in data]
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class CollectCUDA:
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            val = data_dict[key]
            if isinstance(val, torch.Tensor):
                data[key] = val.cuda() if not val.is_cuda else val
            else:
                data[key] = val
        for key, value in self.offset_keys.items():
            val = data_dict[value]
            if isinstance(val, torch.Tensor):
                data[key] = torch.tensor([val.shape[0]], device=val.device)
            else:
                data[key] = torch.tensor([val.shape[0]]).cuda()
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            tensors = []
            for key in keys:
                val = data_dict[key]
                if isinstance(val, torch.Tensor):
                    tensors.append(val.float() if not val.is_cuda else val.float())
                else:
                    tensors.append(torch.tensor(val, dtype=torch.float32).cuda())
            data[name] = torch.cat(tensors, dim=1)
        return data


@TRANSFORMS.register_module()
class InstanceParserCUDA:
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        coord = _arr2tensor(data_dict["coord"], device)
        segment = _arr2tensor(data_dict["segment"], device)
        instance = _arr2tensor(data_dict["instance"], device)
        mask = ~torch.isin(segment, torch.tensor(self.segment_ignore_index, device=device))
        instance[~mask] = self.instance_ignore_index
        unique, inverse = torch.unique(instance[mask], return_inverse=True)
        instance_num = unique.shape[0]
        instance[mask] = inverse
        centroid = torch.ones(coord.shape[0], 3, device=device, dtype=torch.float32) * self.instance_ignore_index
        bbox = torch.ones(instance_num, 8, device=device, dtype=torch.float32) * self.instance_ignore_index
        vacancy = [index for index in self.segment_ignore_index if index >= 0]
        for inst_id in range(instance_num):
            inst_mask = instance == inst_id
            coord_ = coord[inst_mask]
            bbox_min = coord_.min(dim=0)[0]
            bbox_max = coord_.max(dim=0)[0]
            bbox_centroid = coord_.mean(dim=0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = torch.zeros(1, device=device, dtype=coord.dtype)
            seg_val = segment[inst_mask][0]
            bbox_class = seg_val.unsqueeze(0).float()
            for v in vacancy:
                if seg_val > v:
                    bbox_class = bbox_class - 1
            centroid[inst_mask] = bbox_centroid
            bbox[inst_id] = torch.cat(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )
        data_dict["instance"] = _tensor2arr(instance)
        data_dict["instance_centroid"] = _tensor2arr(centroid)
        data_dict["bbox"] = _tensor2arr(bbox)
        return data_dict