#!/bin/bash
#SBATCH --job-name=label_intrinsic_reward_simplerenv
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1                # JEPA world-model forward pass per frame -- more expensive than the
                                          # subgoal-only pass (label_train_simplerenv_qc_critic.slurm's labeling
                                          # stage), similar in kind to LIBERO's label_intrinsic_reward.py.
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=06:00:00                  # intrinsic reward needs a full JEPA forward per frame (not just per
                                          # episode like the subgoal pass), so budget generously vs the ~30min
                                          # subgoal-only run observed for the same episode count.
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out

set -euo pipefail

# Adds VLA-JEPA intrinsic (world-model prediction-error) reward on top of
# the subgoal-only cache produced by label_train_simplerenv_qc_critic.slurm's
# labeling stage -- see qc/label_intrinsic_reward_simplerenv.py's module
# docstring for the full mechanics and the one flagged unverified assumption
# (single-camera V=1 world-model conditioning).
#
# PREREQUISITE: label_train_simplerenv_qc_critic.slurm must have already
# produced $INPUT_CACHE (defaults to that job's cache path).
#
# Runs IN PLACE against your existing `git clone` of
# https://github.com/F1001H/vla-jepa-qc (default: $HOME/vla-jepa-qc),
# reusing the same venv built for the labeling job -- see
# label_train_simplerenv_qc_critic.slurm's header comment for the one-time
# login-node venv setup steps if you haven't done that yet.
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/label_intrinsic_reward_simplerenv.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/label_intrinsic_reward_simplerenv.slurm
#
# After this finishes, train a critic on the combined (subgoal+intrinsic)
# cache with:
#   CACHE_PATH=$OUT_DIR/qc_cache_simplerenv_intrinsic.npz \
#   OUTPUT_NAME=vlajepa_critic_simplerenv_intrinsic.pt \
#   sbatch ~/vla-jepa-qc/slurm/train_simplerenv_qc_critic.slurm
#
# For the reward-composition ablation (subgoal-only vs intrinsic-only vs
# combined), use label_intrinsic_only_reward_simplerenv.slurm instead of
# overriding REWARD_MODE here directly -- it has the right defaults baked
# in so there's nothing to remember.

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_simplerenv_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
CKPT_DIR="$WORK_DIR/checkpoints"
OUT_DIR="$WORK_DIR/output"

SIMPLERENV_CKPT="${SIMPLERENV_CKPT:-$CKPT_DIR/VLA-JEPA-SimplerEnv-repo/SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt}"
INPUT_CACHE="${INPUT_CACHE:-$OUT_DIR/qc_cache_simplerenv.npz}"
OUTPUT_CACHE="${OUTPUT_CACHE:-$OUT_DIR/qc_cache_simplerenv_intrinsic.npz}"
REWARD_MODE="${REWARD_MODE:-combined}"   # or "intrinsic_only" -- see qc/label_intrinsic_reward_simplerenv.py

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1

mkdir -p "$WORK_DIR" "$CKPT_DIR" "$OUT_DIR" "$WORK_DIR/slurm_logs"

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

if [ ! -f "$INPUT_CACHE" ]; then
    echo "ERROR: subgoal-only cache not found at $INPUT_CACHE -- run label_train_simplerenv_qc_critic.slurm's labeling stage first." >&2
    exit 1
fi
if [ ! -f "$SIMPLERENV_CKPT" ]; then
    echo "ERROR: checkpoint not found at $SIMPLERENV_CKPT -- run label_train_simplerenv_qc_critic.slurm first (it downloads this)." >&2
    exit 1
fi

echo "=== Labeling intrinsic reward (mode=$REWARD_MODE) on top of $INPUT_CACHE ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/label_intrinsic_reward_simplerenv.py \
    --ckpt-path "$SIMPLERENV_CKPT" \
    --input-cache "$INPUT_CACHE" \
    --output-path "$OUTPUT_CACHE" \
    --reward-mode "$REWARD_MODE" \
    --checkpoint-every 25 --cuda 0

echo "Done. Cache ($REWARD_MODE): $OUTPUT_CACHE"
