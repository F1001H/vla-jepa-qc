#!/bin/bash
#SBATCH --job-name=train_simplerenv_qc_critic_intrinsic_only
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1                # lightweight: critic-over-cached-embeddings training, no VLM/JEPA
                                          # model loaded, just qc/train_critic.py.
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=14
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out

set -euo pipefail

# intrinsic_only-cache variant of train_simplerenv_qc_critic.slurm, for the
# reward-composition ablation (subgoal-only vs intrinsic-only vs combined).
#
# PREREQUISITE: label_intrinsic_only_reward_simplerenv.slurm must have
# already produced $CACHE_PATH (defaults to the path that job writes to).
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/train_simplerenv_qc_critic_intrinsic_only.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/train_simplerenv_qc_critic_intrinsic_only.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_simplerenv_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
OUT_DIR="$WORK_DIR/output"

CACHE_PATH="${CACHE_PATH:-$OUT_DIR/qc_cache_simplerenv_intrinsic_only.npz}"
OUTPUT_NAME="${OUTPUT_NAME:-vlajepa_critic_simplerenv_intrinsic_only.pt}"

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
    echo "ERROR: intrinsic_only reward cache not found at $CACHE_PATH -- run label_intrinsic_only_reward_simplerenv.slurm first (or point CACHE_PATH at wherever it landed)." >&2
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
