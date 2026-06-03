_base_ = ["semseg-litept-small-v1m1.py"]

# === 服务器配置 ===

# --- 资源相关（与本地差异大）---
batch_size = 4                 # 服务器显存大，可开大 batch
crop_point_max = 500000         # 500000 点 ≈ 10m 柱半径
num_worker = 64
data_root = "/data/dataset/forest"  # 改成服务器实际路径
save_path = "exp/forest/semseg-litept-small-v1m1"

# 推理参数
data = dict(
    block_xy=40,
    overlap=10,
)