#!/bin/bash
#SBATCH --job-name=train_simplerenv_qc_critic
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1                # lightweight: critic-over-cached-embeddings training, no VLM/JEPA
                                          # model loaded, just qc/train_critic.py -- see label_train_simplerenv_qc_critic.slurm
                                          # for the (separate, expensive) labeling pass that produces the cache this reads.
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out

set -euo pipefail

# Standalone critic-training job, decoupled from labeling
# (label_train_simplerenv_qc_critic.slurm) so it's cheap to retry or sweep
# hyperparameters on without repeating the expensive GPU labeling pass
# (loads the VLM+JEPA world model over ~1000 episodes) -- mirrors the same
# split already used for LIBERO, see
# slurm/train_libero_qc_critic_full_finetune_goal.slurm in openpi's repo.
#
# PREREQUISITE: label_train_simplerenv_qc_critic.slurm must have already
# produced $CACHE_PATH (defaults to the same path that job writes to).
#
# Runs IN PLACE against your existing `git clone` of
# https://github.com/F1001H/vla-jepa-qc (default: $HOME/vla-jepa-qc),
# reusing the same venv built for the labeling job -- see that job's header
# comment for the one-time login-node venv setup steps if you haven't done
# that yet.
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/train_simplerenv_qc_critic.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/train_simplerenv_qc_critic.slurm
#
# Override any hyperparameter via env var, e.g. for a quick sweep:
#   EPOCHS=40 LR=1e-4 OUTPUT_NAME=vlajepa_critic_simplerenv_lr1e-4.pt sbatch ...

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_simplerenv_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
OUT_DIR="$WORK_DIR/output"

CACHE_PATH="${CACHE_PATH:-$OUT_DIR/qc_cache_simplerenv.npz}"
OUTPUT_NAME="${OUTPUT_NAME:-vlajepa_critic_simplerenv.pt}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
TAU="${TAU:-0.005}"
DISCOUNT="${DISCOUNT:-0.99}"
Q_AGG="${Q_AGG:-mean}"
NUM_QS="${NUM_QS:-5}"

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"

mkdir -p "$WORK_DIR" "$OUT_DIR" "$WORK_DIR/slurm_logs"

echo "=== GPU diagnostics ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv
echo "========================"

cd "$REPO_DIR"

if [ ! -f ".venv/.setup_complete" ]; then
    echo "ERROR: $REPO_DIR/.venv is missing or incomplete. Build it from a login node first -- see label_train_simplerenv_qc_critic.slurm's header comment for the exact steps." >&2
    exit 1
fi
PY="$REPO_DIR/.venv/bin/python"

if [ ! -f "$CACHE_PATH" ]; then
    echo "ERROR: reward cache not found at $CACHE_PATH -- run label_train_simplerenv_qc_critic.slurm's labeling stage first (or point CACHE_PATH at wherever it landed)." >&2
    exit 1
fi

echo "=== Training critic on $CACHE_PATH (epochs=$EPOCHS, lr=$LR, num_qs=$NUM_QS) ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/train_critic.py \
    --cache-path "$CACHE_PATH" \
    --output-path "$OUT_DIR/$OUTPUT_NAME" \
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --lr "$LR" \
    --tau "$TAU" --discount "$DISCOUNT" --q-agg "$Q_AGG" --num-qs "$NUM_QS" \
    --cuda 0

echo "Done. Critic checkpoint: $OUT_DIR/$OUTPUT_NAME"
echo "rsync it back locally with:"
echo "  rsync -avz \$(hostname):$OUT_DIR/$OUTPUT_NAME ~/starvla_jepa/"
