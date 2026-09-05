#!/bin/bash
#SBATCH --job-name=finetune_vlajepa_kobo
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:8                # full node -- accelerate/deepspeed data-parallel across all 8
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=112
#SBATCH --mem=1800G
#SBATCH --time=48:00:00                  # 30k steps ~30h per Fabian's plan (2026-09-03), generous margin
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail

# VLA-JEPA fine-tuning on kobo real-robot data (30k steps, ~30h, per
# Fabian's plan 2026-09-03) -- the one step in the whole QC investigation
# that strictly needs the cluster (everything else -- labeling, critic
# training, LIBERO/LIBERO-Plus eval -- runs fine locally). MUST start
# before the cluster's Monday maintenance outage.
#
# PREREQUISITE (not yet done as of writing this script): the kobo LeRobot
# v3 dataset(s) referenced by starVLA/dataloader/gr00t_lerobot/mixtures.py's
# "kobo_all" entry don't exist yet -- raw .bin recordings (see
# ~/zed_memory_node/start_recording.sh) get compressed to .h5
# (~/zed_memory_node/convert_data.py, already working) but the FINAL
# .h5 -> LeRobot v3 conversion step (convert_to_task.py) is written for
# the OLD dual-arm setup and a single hardcoded task -- needs adapting for
# single-arm kobo + the actual task set before this job can run. Update
# mixtures.py's "kobo_all" entry with the real dataset directory name(s)
# once built.
#
# Reuses the SAME shared workspace/checkpoint cache as
# label_train_libero_qc_critic_8gpu.slurm (Qwen3-VL-2B-Instruct,
# vjepa2-vitl-fpc64-256 -- skips re-downloading if that job already ran)
# and the SAME REPO_DIR .venv (has accelerate==1.14.0 + deepspeed==0.16.9
# already installed, confirmed).
#
# Submit with:
#   KOBO_DATASET_ROOT=/path/to/kobo/lerobot_v3/datasets sbatch ~/vla-jepa-qc/slurm/finetune_vlajepa_kobo.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc KOBO_DATASET_ROOT=... sbatch /path/to/vla-jepa-qc/slurm/finetune_vlajepa_kobo.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_libero_qc"   # same shared workspace the labeling/eval jobs use
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
KOBO_DATASET_ROOT="${KOBO_DATASET_ROOT:-}"

CONFIG_FILE="$REPO_DIR/scripts/configs/vlajepa_kobo_ft.yaml"
DS_CONFIG="$REPO_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml"
NUM_PROCESSES="${NUM_PROCESSES:-8}"      # match --gres=gpu:h100:N above if changed

if [ -z "$KOBO_DATASET_ROOT" ]; then
    echo "ERROR: KOBO_DATASET_ROOT must be set (parent dir containing the per-task" >&2
    echo "  LeRobot v3 dataset dirs referenced by mixtures.py's 'kobo_all' entry)." >&2
    echo "  Example: KOBO_DATASET_ROOT=\$SHARED_WS/fabian/kobo_lerobot_v3 sbatch ..." >&2
    exit 1
fi
if [ ! -d "$KOBO_DATASET_ROOT" ]; then
    echo "ERROR: KOBO_DATASET_ROOT=$KOBO_DATASET_ROOT does not exist." >&2
    exit 1
fi

# WORKSPACE is used by vlajepa_kobo_ft.yaml's ${oc.env:WORKSPACE} interpolations
# (base_vlm, base_encoder, pretrained_checkpoint, run_root_dir) -- unlike the
# LIBERO/SimplerEnv checkpoints' stray ${oc.env:WORKSPACE} fields (which get
# patched to literal paths post-download since we don't control those config
# files), THIS config was written by us specifically to use this env var, so
# no post-download patching is needed here.
export WORKSPACE="$WORK_DIR"
export KOBO_DATASET_ROOT
export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0   # matches the reference launch script's default -- adjust if this cluster's node interconnect uses a different interface name
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export OMP_NUM_THREADS=1
export FFMPEG_THREADS=1

mkdir -p "$WORK_DIR" "$CKPT_DIR" "$LOG_DIR" "$WORK_DIR/slurm_logs" "$WORKSPACE/runs"

echo "=== GPU diagnostics ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv
echo "========================"

cd "$REPO_DIR"
if [ ! -f ".venv/.setup_complete" ]; then
    echo "ERROR: $REPO_DIR/.venv is missing or incomplete. Build it from a login node first -- see label_train_simplerenv_qc_critic.slurm's header comment for the exact steps." >&2
    exit 1
fi
ACCELERATE_BIN="$REPO_DIR/.venv/bin/accelerate"
PY="$REPO_DIR/.venv/bin/python"

# --- Base backbone checkpoints (Qwen3-VL-2B-Instruct, V-JEPA2-ViT-L) --
# same cache location/download as label_train_libero_qc_critic_8gpu.slurm,
# skipped here if that job already populated it. ---
if [ ! -d "$CKPT_DIR/Qwen3-VL-2B-Instruct" ]; then
    echo "=== Downloading Qwen3-VL-2B-Instruct ==="
    "$PY" -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-VL-2B-Instruct \
        --local-dir "$CKPT_DIR/Qwen3-VL-2B-Instruct"
fi
if [ ! -d "$CKPT_DIR/vjepa2-vitl-fpc64-256" ]; then
    echo "=== Downloading vjepa2-vitl-fpc64-256 ==="
    "$PY" -m huggingface_hub.commands.huggingface_cli download facebook/vjepa2-vitl-fpc64-256 \
        --local-dir "$CKPT_DIR/vjepa2-vitl-fpc64-256"
fi

# --- VLA-JEPA pretrain checkpoint (the actual fine-tuning starting point) ---
PRETRAIN_CKPT="$CKPT_DIR/VLA-JEPA-pretrain-repo/Pretrain/checkpoints/VLA-JEPA-pretrain.pt"
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "=== Downloading VLA-JEPA-pretrain checkpoint ==="
    "$PY" -m huggingface_hub.commands.huggingface_cli download ginwind/VLA-JEPA \
        --include "Pretrain/*" --local-dir "$CKPT_DIR/VLA-JEPA-pretrain-repo"
fi
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: expected pretrain checkpoint not found at $PRETRAIN_CKPT after download -- check ginwind/VLA-JEPA's actual repo layout hasn't changed." >&2
    exit 1
fi

echo "=== [$(date)] Starting VLA-JEPA kobo fine-tune (config: $CONFIG_FILE) ==="
"$ACCELERATE_BIN" launch \
    --config_file "$DS_CONFIG" \
    --num_processes "$NUM_PROCESSES" \
    "$REPO_DIR/starVLA/training/train_starvla.py" \
    --config_yaml "$CONFIG_FILE" \
    2>&1 | tee -a "$LOG_DIR/finetune_vlajepa_kobo.log"

echo "=== [$(date)] DONE. Checkpoints under \$WORKSPACE/runs/vlajepa_kobo_ft ==="
