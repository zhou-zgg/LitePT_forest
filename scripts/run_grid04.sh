#!/bin/bash
cd "$(dirname "$0")/.." || exit

CONDA_ENV="litept"
PYTHON=$(conda run -n "$CONDA_ENV" which python 2>/dev/null || echo "python")

EXP_NAME="semseg-litept-small-v1m1-grid04"
LOG_FILE="exp/forest/${EXP_NAME}/train_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "exp/forest/${EXP_NAME}"

echo "========== Phase 2: grid_size=0.04 实验 ==========" | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"
echo "配置: grid_size 0.02→0.04, 其余同 Phase 1" | tee -a "$LOG_FILE"
echo "Phase 1 best mIoU: (待填入)" | tee -a "$LOG_FILE"
echo "Python: $PYTHON" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

export PYTHONPATH=./

nohup "$PYTHON" tools/train.py \
    --config-file configs/forest/semseg-litept-small-v1m1-grid04-local.py \
    --num-gpus 1 \
    --options save_path="exp/forest/${EXP_NAME}" \
    weight="exp/forest/semseg-litept-small-v1m1-loss-v2/model/model_best.pth" \
    >> "$LOG_FILE" 2>&1 &

echo "训练已在后台启动, PID: $!"
echo "日志文件: $LOG_FILE"
echo "查看日志: tail -f $LOG_FILE"
