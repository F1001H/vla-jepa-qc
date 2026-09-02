#!/bin/bash
#SBATCH --job-name=label_train_libero_qc_critic_8gpu
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:8                # full node -- sharding the labeling pass across all 8 GPUs
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=112              # full node's CPU count (14/GPU, matching the 1-GPU jobs' ratio)
#SBATCH --mem=1800G                      # generous headroom under the node's ~2TB -- 8 concurrent
                                          # VLA-JEPA model copies (Qwen3-VL-2B + V-JEPA2-ViT-L each)
#SBATCH --time=04:00:00                  # sharded subgoal ~1693/8 eps * ~19s/ep ~= 1.1h, sharded
                                          # intrinsic well under that, generous margin for cold downloads
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail   # NOT -e: need to check each of the 8 parallel shards' own exit status

# LIBERO QC subgoal+intrinsic labeling, sharded across all 8 GPUs on one
# node (per-GPU wall-clock: sharded subgoal ~1.1h vs ~9h single-GPU;
# sharded intrinsic proportionally faster too), then critic training.
# Fixed reward detector (qc/label_rewards.py, 2026-09-02) throughout --
# see that file's module comment for why the old fixed-threshold
# hysteresis detector badly under-detected real grasp/release events.
#
# First run on this cluster: LIBERO checkpoint + dataset aren't there yet
# (only SimplerEnv's checkpoint was downloaded previously) -- this job
# downloads both if missing (~39GB total, one-time).
#
# Runs IN PLACE against your existing `git clone` of
# https://github.com/F1001H/vla-jepa-qc (default: $HOME/vla-jepa-qc) --
# reuses the SAME venv built for the SimplerEnv jobs (same repo, same
# dependencies), see label_train_simplerenv_qc_critic.slurm's header
# comment for the one-time login-node venv setup steps if not done yet.
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/label_train_libero_qc_critic_8gpu.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/label_train_libero_qc_critic_8gpu.slurm
#
# Each stage's shards resume independently via their own
# --checkpoint-every support, so if the whole job dies (node failure,
# preemption) resubmitting is cheap -- already-done shards' progress
# isn't lost, they just pick up where they left off.

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_libero_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
CKPT_DIR="$WORK_DIR/checkpoints"
DATASET_DIR="$WORK_DIR/datasets/physical-intelligence-libero"
OUT_DIR="$WORK_DIR/output"
LOG_DIR="$WORK_DIR/logs"
NUM_SHARDS=8

LIBERO_CKPT="$CKPT_DIR/VLA-JEPA-LIBERO-repo/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1

mkdir -p "$WORK_DIR" "$CKPT_DIR" "$DATASET_DIR" "$OUT_DIR" "$LOG_DIR" "$WORK_DIR/slurm_logs"

echo "=== GPU diagnostics ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv
echo "========================"

cd "$REPO_DIR"

if [ ! -f ".venv/.setup_complete" ]; then
    echo "ERROR: $REPO_DIR/.venv is missing or incomplete. Build it from a login node first -- see label_train_simplerenv_qc_critic.slurm's header comment for the exact steps." >&2
    exit 1
fi
PY="$REPO_DIR/.venv/bin/python"

# --- 0. Download checkpoint + dataset if missing (idempotent) ---
if [ ! -f "$LIBERO_CKPT" ]; then
    echo "=== Downloading VLA-JEPA-LIBERO checkpoint ==="
    "$PY" -m huggingface_hub.commands.huggingface_cli download ginwind/VLA-JEPA \
        --include "LIBERO/*" --local-dir "$CKPT_DIR/VLA-JEPA-LIBERO-repo"
fi
if [ ! -d "$DATASET_DIR/data" ]; then
    echo "=== Downloading physical-intelligence/libero dataset ==="
    "$PY" -m huggingface_hub.commands.huggingface_cli download physical-intelligence/libero \
        --repo-type dataset --local-dir "$DATASET_DIR"
fi

# --- Stage 1: sharded subgoal labeling (fixed detector), 8 parallel GPUs ---
echo "=== [$(date)] Stage 1: sharded subgoal labeling ==="
subgoal_shards=()
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_path="$OUT_DIR/qc_cache_libero_subgoal_shard${i}of${NUM_SHARDS}.npz"
    subgoal_shards+=("$shard_path")
    CUDA_VISIBLE_DEVICES="$i" PYTHONPATH="$REPO_DIR" "$PY" qc/label_rewards.py \
        --dataset-root "$DATASET_DIR" --ckpt-path "$LIBERO_CKPT" \
        --output-path "$shard_path" \
        --shard-id "$i" --num-shards "$NUM_SHARDS" \
        --checkpoint-every 25 --cuda 0 \
        > "$LOG_DIR/label_subgoal_shard${i}.log" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "ERROR: one or more subgoal-labeling shards failed -- check $LOG_DIR/label_subgoal_shard*.log" >&2
    exit 1
fi
echo "=== [$(date)] Stage 1 done, merging shards ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/merge_shards.py --mode subgoal \
    --shard-paths "${subgoal_shards[@]}" \
    --output-path "$OUT_DIR/qc_cache_libero_full.npz"

# --- Stage 2: sharded intrinsic labeling, 8 parallel GPUs ---
echo "=== [$(date)] Stage 2: sharded intrinsic labeling ==="
intrinsic_shards=()
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_path="$OUT_DIR/qc_cache_libero_intrinsic_shard${i}of${NUM_SHARDS}.npz"
    intrinsic_shards+=("$shard_path")
    CUDA_VISIBLE_DEVICES="$i" PYTHONPATH="$REPO_DIR" "$PY" qc/label_intrinsic_reward.py \
        --dataset-root "$DATASET_DIR" --ckpt-path "$LIBERO_CKPT" \
        --input-cache "$OUT_DIR/qc_cache_libero_full.npz" \
        --output-path "$shard_path" \
        --shard-id "$i" --num-shards "$NUM_SHARDS" \
        --checkpoint-every 25 --cuda 0 \
        > "$LOG_DIR/label_intrinsic_shard${i}.log" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "ERROR: one or more intrinsic-labeling shards failed -- check $LOG_DIR/label_intrinsic_shard*.log" >&2
    exit 1
fi
echo "=== [$(date)] Stage 2 done, merging shards ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/merge_shards.py --mode intrinsic \
    --shard-paths "${intrinsic_shards[@]}" \
    --output-path "$OUT_DIR/qc_cache_libero_intrinsic_full.npz"

# --- Stage 3: train critic (single GPU) ---
echo "=== [$(date)] Stage 3: training critic ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_DIR" "$PY" qc/train_critic.py \
    --cache-path "$OUT_DIR/qc_cache_libero_intrinsic_full.npz" \
    --output-path "$OUT_DIR/vlajepa_critic_libero_intrinsic.pt" \
    --epochs 20 --batch-size 256 --num-qs 5 --cuda 0

echo "=== [$(date)] ALL DONE. Critic: $OUT_DIR/vlajepa_critic_libero_intrinsic.pt ==="
echo "rsync it back locally with:"
echo "  rsync -avz \$(hostname):$OUT_DIR/vlajepa_critic_libero_intrinsic.pt ~/starvla_jepa/"
