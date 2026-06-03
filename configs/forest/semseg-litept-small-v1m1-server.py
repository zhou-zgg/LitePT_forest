_base_ = ["semseg-litept-small-v1m1.py"]

# === 服务器配置（vGPU-32GB, 128核, 1TB内存）===

# --- 资源相关 ---
batch_size = 8                 # 32GB 显存，可以开大 batch
crop_point_max = 500000        # 500000 点 ≈ 10m 柱半径
num_worker = 32                # 128核 CPU，32 足够
data_root = "data/forest"
save_path = "exp/forest/semseg-litept-small-v1m1"

# 推理参数
data = dict(
    block_xy=40,
    overlap=10,
)