# LitePT

## Environment Setup

```bash
conda activate litept
export PYTHONPATH=./
```

## Key Commands

Train (single machine, multi-GPU):
```bash
PYTHONPATH=./ python tools/train.py --config-file configs/<dataset>/<config>.py --num-gpus <N> \
  --options epoch=<N> eval_epoch=<N> weight=<PRETRAINED_WEIGHT> save_path=<SAVE_PATH>
```

Example - train forest with pretrained weight:
```bash
PYTHONPATH=./ python tools/train.py \
  --config-file configs/forest/semseg-litept-small-v1m1.py \
  --num-gpus 1 \
  --options epoch=100 eval_epoch=100 \
    weight=exp/forest/semseg-litept-small-v1m1-nuscenes/model/model_best.pth \
    save_path=exp/forest/semseg-litept-small-v1m1
```

Resume training from checkpoint:
```bash
PYTHONPATH=./ python tools/train.py \
  --config-file configs/forest/semseg-litept-small-v1m1.py \
  --num-gpus 1 \
  --options resume=True weight=<CHECKPOINT_PATH> save_path=<SAVE_PATH>
```

Test with pretrained weights:
```bash
PYTHONPATH=./ python tools/test.py \
  --config-file <CONFIG_PATH> --num-gpus <N> \
  --options save_path=<SAVE_PATH> weight=<CHECKPOINT_PATH>
```

## Architecture

- **Entry points**: `tools/train.py`, `tools/test.py` — both use `engines/launch.py` for DDP spawn
- **Config system**: Python-based configs in `configs/` using `_base_` inheritance (`configs/_base_/default_runtime.py`). Resolved via `utils/config.py` (adapted from mmcv).
- **Config override**: `--options key=value key2=value2` on CLI, e.g. `--options save_path=exp/foo weight=path/to/model.pth`
- **Model registry**: `models/builder.py` uses `MODELS = Registry("models")`. Register with `@MODELS.register_module("Name")`.
- **Task registry**: Trainer types in `engines/train.py`, Tester types in `engines/test.py`
- **Standalone usage**: `demo_use.py` + `litept/` + `libs/pointrope/` can be copied into other projects

## CUDA Extensions (libs/)

All three must be compiled after installing PyTorch:

1. **`libs/pointrope/`** — PointROPE (required). Edit `setup.py` `all_cuda_archs` to match your GPU compute capability before building:
   - RTX 3090: `sm_86`, RTX 4080: `sm_89`, H100: `sm_90`
   ```bash
   cd libs/pointrope && python setup.py install && cd ../..
   ```
   Falls back to pure PyTorch if CUDA build unavailable (slightly slower).

2. **`libs/pointops/`** — Required for evaluator hooks (SemSegEvaluator, PreciseEvaluator)
   ```bash
   cd libs/pointops && python setup.py install && cd ../..
   ```

3. **`libs/pointgroup_ops/`** — Required for instance segmentation (PointGroup). Needs `google-sparsehash`:
   ```bash
   conda install -c bioconda google-sparsehash
   cd libs/pointgroup_ops && python setup.py install && cd ../..
   ```

## Dataset Convention

Data should be placed under `data/<dataset>/` (e.g. `data/scannet/`, `data/nuscenes/`), following [Pointcept data preparation](https://github.com/Pointcept/Pointcept#data-preparation).

## Config Notes

- `batch_size` in config = **total across all GPUs** (divided by world_size at runtime)
- `epoch` / `eval_epoch`: must be divisible (actual eval freq = `eval_epoch`); checkpoints saved every `save_freq` epochs
- `hooks` list controls training pipeline: checkpointing, evaluation, wandb logging
- AMP enabled per-config via `enable_amp = True`
- `weight`: path to pretrained checkpoint (optional, start from scratch if not set)
- `resume`: if True, resume from `model_last.pth` in `save_path`
