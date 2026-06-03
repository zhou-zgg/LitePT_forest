_base_ = ["semseg-litept-small-v1m1.py"]

# === 本地机器配置（RTX 4080 16GB）===

# --- 资源相关（与服务器差异大）---
batch_size = 2                 # 16GB 显存，CylinderCrop 只能开 2
crop_point_max = 200000         # 200000 点 ≈ 6m 柱半径，16GB 极限
num_worker = 20
data_root = "data/forest"
save_path = "exp/forest/semseg-litept-small-v1m1"

# 推理参数
data = dict(
    block_xy=20,
    overlap=5,
)