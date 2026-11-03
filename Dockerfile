# Forest_net 训练/推理镜像
# 基础镜像复用已有 pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel：
#   驱动 535 -> CUDA 最高支持 12.2，cu118 最稳；devel 版自带 nvcc 用于编译扩展
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

# RTX A6000 = Ampere = 8.6；4090/3090 换成 8.9/8.6，H100 用 9.0
ARG GPU_ARCH_LIST=8.6
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git vim tmux htop ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) Python 依赖
#    PyG 系列 (torch-scatter/sparse/cluster) 走 cu118 预编译 wheel，避免本地编译
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        -f https://data.pyg.org/whl/torch-2.6.0+cu118.html \
    && pip install --no-cache-dir spconv-cu118 laspy

# 2) 编译 CUDA 扩展（烧进镜像，之后换机器/重建容器都不用再编）
COPY libs/pointrope /opt/pointrope
COPY libs/pointops  /opt/pointops
RUN GPU_ARCH_NUM="$(echo "${GPU_ARCH_LIST}" | tr -d '.')" \
    # pointrope 的 setup.py 硬编码 sm_89/sm_90，patch 成目标 GPU 的 arch
    && sed -i "s/arch=compute_89,code=sm_89/arch=compute_${GPU_ARCH_NUM},code=sm_${GPU_ARCH_NUM}/; /compute_90/d" \
        /opt/pointrope/setup.py \
    && cd /opt/pointrope && TORCH_CUDA_ARCH_LIST="${GPU_ARCH_LIST}" python setup.py install \
    && cd /opt/pointops && TORCH_CUDA_ARCH_LIST="${GPU_ARCH_LIST}" python setup.py install

# 3) flash-attn（models/litept 的硬依赖；源码编译是最耗时步骤，约 30-60 分钟）
RUN MAX_JOBS=8 pip install --no-build-isolation flash-attn

ENV PYTHONPATH=/workspace \
    TORCH_CUDA_ARCH_LIST=${GPU_ARCH_LIST}

CMD ["sleep", "infinity"]
