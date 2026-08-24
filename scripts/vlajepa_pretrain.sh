#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/dataset-assist-0/ginwind/VLA-JEPA
ENV_ROOT=/home/dataset-assist-0/ginwind/VLA-JEPA/env
CONFIG_PATH=${PROJECT_ROOT}/scripts/configs/vlajepa_co_pretrain.yaml
RUN_DIR=/home/dataset-local/raw_train_runs/vlajepa_cotrain

mkdir -p "${RUN_DIR}" /home/dataset-local/tmp

exec >"${RUN_DIR}/train.log" 2>&1

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export CUDA_HOME="${ENV_ROOT}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_NVIDIA_SITE_PACKAGES="${CUDA_HOME}/lib/python3.10/site-packages/nvidia"
export CUDA_INCLUDE_PATHS="${CUDA_HOME}/targets/x86_64-linux/include:${CUDA_HOME}/targets/x86_64-linux/include/cccl:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_runtime/include:${CUDA_NVIDIA_SITE_PACKAGES}/cusparse/include:${CUDA_NVIDIA_SITE_PACKAGES}/cublas/include:${CUDA_NVIDIA_SITE_PACKAGES}/cusolver/include:${CUDA_NVIDIA_SITE_PACKAGES}/cufft/include:${CUDA_NVIDIA_SITE_PACKAGES}/cudnn/include:${CUDA_NVIDIA_SITE_PACKAGES}/nccl/include:${CUDA_NVIDIA_SITE_PACKAGES}/nvtx/include:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_cupti/include:${CUDA_NVIDIA_SITE_PACKAGES}/curand/include:${CUDA_NVIDIA_SITE_PACKAGES}/nvjitlink/include:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_nvrtc/include"
export CUDA_LIBRARY_PATHS="${CUDA_NVIDIA_SITE_PACKAGES}/cuda_runtime/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cusparse/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cublas/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cusolver/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cufft/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cudnn/lib:${CUDA_NVIDIA_SITE_PACKAGES}/nccl/lib:${CUDA_NVIDIA_SITE_PACKAGES}/nvtx/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_cupti/lib:${CUDA_NVIDIA_SITE_PACKAGES}/curand/lib:${CUDA_NVIDIA_SITE_PACKAGES}/nvjitlink/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_nvrtc/lib:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATHS}:${CUDA_HOME}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"

exec "${ENV_ROOT}/bin/accelerate" launch \
  --config_file "${PROJECT_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml" \
  --num_processes 8 \
  "${PROJECT_ROOT}/starVLA/training/train_jevla_cotrain.py" \
  --config_yaml "${CONFIG_PATH}"
