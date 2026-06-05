#!/bin/bash
# Phase 1: Loss 增强实验 (CE + label_smoothing + Lovasz + Dice)
# 从 best 权重 fine-tune 30 epoch

cd "$(dirname "$0")/.." || exit

EXP_NAME="semseg-litept-small-v1m1-loss-v2"
LOG_FILE="exp/forest/${EXP_NAME}/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "exp/forest/${EXP_NAME}"

echo "========== Phase 1: Loss-v2 实验 ==========" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "配置: CE(0.1 smooth) + Lovasz + Dice, lr=0.001, 30 epoch" | tee -a "$LOG_FILE"
echo "基线 mIoU: 0.5529" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

export PYTHONPATH=./

nohup python tools/train.py \
    --config-file configs/forest/semseg-litept-small-v1m1-loss-v2-local.py \
    --num-gpus 1 \
    --options save_path="exp/forest/${EXP_NAME}" \
    weight="exp/forest/semseg-litept-small-v1m1/model/model_best.pth" \
    >> "$LOG_FILE" 2>&1 &

echo "训练已在后台启动, PID: $!"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
