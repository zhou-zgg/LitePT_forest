#!/bin/bash
# 1M crop + patch2048 + Loss-v2 + clean data on RTX 4080 32GB
# 预期 mIoU 0.65+, 60 epoch 约 5-6 小时

set -e
cd /root/workshop/LitePT_forest
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0

SAVE_DIR="/root/autodl-tmp/exp/forest/semseg-litept-small-v1m1-loss-v2-1m"
mkdir -p "$SAVE_DIR"

nohup setsid python tools/train.py \
  --config-file configs/forest/semseg-litept-small-v1m1-loss-v2-1m-server.py \
  --num-gpus 1 \
  --options resume=True \
  > "$SAVE_DIR/train.log" 2>&1 < /dev/null &

PID=$!
echo "Training started, PID=$PID"
echo "Log: $SAVE_DIR/train.log"
echo "Monitor: tail -f $SAVE_DIR/train.log"
