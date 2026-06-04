import os
import sys
import argparse
import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from time import gmtime, strftime

import utils.comm as comm
from utils.env import get_random_seed, set_seed
from utils.config import Config, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = None if seed is None else num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config-file", default="configs/nuscenes/semseg-litept-small-v1m1.py", metavar="FILE", help="path to config file"
    )

    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        # default="tcp://172.28.30.96:27465",
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser


def _fix_transform_cuda_suffix(cfg_node, use_gpu):
    if not use_gpu and isinstance(cfg_node, dict):
        for key in ('post_transform', 'aug_transform'):
            if key not in cfg_node:
                continue
            if key == 'post_transform':
                for t in cfg_node[key]:
                    if hasattr(t, 'type') and t.type.endswith('CUDA'):
                        t['type'] = t.type[:-4]
            elif key == 'aug_transform':
                for aug_list in cfg_node[key]:
                    for t in aug_list:
                        if hasattr(t, 'type') and t.type.endswith('CUDA'):
                            t['type'] = t.type[:-4]
        if 'voxelize' in cfg_node and hasattr(cfg_node.voxelize, 'type'):
            if cfg_node.voxelize.type.endswith('CUDA'):
                cfg_node.voxelize['type'] = cfg_node.voxelize.type[:-4]


def default_config_parser(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    # Apply top-level variable overrides to nested dicts
    # (mmcv Config inheritance doesn't update variable references in nested dicts)

    # If use_gpu_transform is False, strip "CUDA" suffix from all transform types
    use_gpu_transform = getattr(cfg, 'use_gpu_transform', True)
    if not use_gpu_transform:
        for split in ('train', 'val', 'test'):
            if not hasattr(cfg.data, split):
                continue
            ds = getattr(cfg.data, split)
            if hasattr(ds, 'transform'):
                for t in ds.transform:
                    if hasattr(t, 'type') and t.type.endswith('CUDA'):
                        t['type'] = t.type[:-4]
            if hasattr(ds, 'test_cfg'):
                _fix_transform_cuda_suffix(ds.test_cfg, use_gpu_transform)

    for split in ('train', 'val', 'test'):
        if not hasattr(cfg.data, split):
            continue
        ds = getattr(cfg.data, split)

        # data_root, dataset_type, class_mapping
        if hasattr(cfg, 'data_root') and hasattr(ds, 'data_root'):
            ds['data_root'] = cfg.data_root
        if hasattr(cfg, 'dataset_type') and hasattr(ds, 'type'):
            ds['type'] = cfg.dataset_type
        if split == 'train' and hasattr(cfg, 'class_mapping') and hasattr(ds, 'class_mapping'):
            ds['class_mapping'] = cfg.class_mapping

        # crop_type, crop_point_max, grid_size in transform list
        if hasattr(ds, 'transform'):
            for t in ds.transform:
                # Handle both crop types (CPU and CUDA variants)
                if hasattr(t, 'type') and t.type.rstrip('CUDA') in ('CylinderCrop', 'SphereCrop', 'CylinderCropCUDA', 'SphereCropCUDA'):
                    if hasattr(cfg, 'crop_type'):
                        crop_type = cfg.crop_type
                        if not use_gpu_transform and crop_type.endswith('CUDA'):
                            crop_type = crop_type[:-4]
                        t['type'] = crop_type
                    if hasattr(cfg, 'crop_point_max'):
                        t['point_max'] = cfg.crop_point_max
                # grid_size in GridSample and Update transforms (CPU and CUDA variants)
                if hasattr(cfg, 'grid_size'):
                    if hasattr(t, 'type') and t.type in ('GridSample', 'GridSampleCUDA') and 'grid_size' in t:
                        t['grid_size'] = cfg.grid_size
                    if hasattr(t, 'type') and t.type in ('Update', 'UpdateCUDA') and 'keys_dict' in t:
                        if 'grid_size' in t.keys_dict:
                            t.keys_dict['grid_size'] = cfg.grid_size

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    # cfg.save_path = cfg.save_path + '_' + cfg.current_time


    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    print("world_size: ")
    print(world_size)
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    # update data loop
    assert cfg.epoch % cfg.eval_epoch == 0
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed + rank * cfg.num_worker_per_gpu
    set_seed(seed)
    return cfg
