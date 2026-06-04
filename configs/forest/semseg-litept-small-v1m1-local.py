_base_ = ["semseg-litept-small-v1m1.py"]

# === 本地机器配置（RTX 4080 16GB）===

# --- 路径 ---
data_root = "data/forest"
save_path = "exp/forest/semseg-litept-small-v1m1"
weight = None

# --- 资源相关 ---
batch_size = 1                 # 16GB 显存限制
crop_point_max = 150000        # 150000 点，16GB 极限
num_worker = 1                 # GPU 模式需降到 1，多 worker 占显存 OOM

# 推理参数
data = dict(
    block_xy=20,
    overlap=5,
)
