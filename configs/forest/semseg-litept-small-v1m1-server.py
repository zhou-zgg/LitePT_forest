_base_ = ["semseg-litept-small-v1m1.py"]

# === 服务器配置（vGPU-32GB, 128核, 1TB内存）===

# --- 路径 ---
data_root = "/root/autodl-tmp/forest"
save_path = "exp/forest/semseg-litept-small-v1m1"
weight = "exp/pretrain/model_best.pth"

# --- 资源相关 ---
batch_size = 8                 # 32GB 显存，可以开大 batch
crop_point_max = 600000        # 600000 点 ≈ 11m 柱半径
num_worker = 2                 # GPU 模式需降到 1~2，多 worker 占显存 OOM

# 推理参数
data = dict(
    block_xy=40,
    overlap=10,
)
