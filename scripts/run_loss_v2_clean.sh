#!/bin/bash
cd "$(dirname "$0")/.." || exit

source ~/anaconda3/etc/profile.d/conda.sh
conda activate litept
export PYTHONPATH=./

echo "========== Loss-v2 Clean Data Training =========="
echo "Start: $(date)"

python tools/train.py \
    --config-file configs/forest/semseg-litept-small-v1m1-loss-v2-clean-local.py \
    --num-gpus 1

echo "End: $(date)"
