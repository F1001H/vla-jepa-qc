#!/bin/bash
#SBATCH --job-name=label_intrinsic_only_reward_simplerenv
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out

set -euo pipefail

# intrinsic_only variant of label_intrinsic_reward_simplerenv.slurm, for
# the reward-composition ablation: subgoal-only vs intrinsic-only vs
# combined. Reward = JEPA world-model prediction error ALONE, with the
# subgoal reward (including its unconditional terminal +1 bonus) fully
# zeroed out -- isolates what intrinsic reward contributes on its own,
# with zero notion of task success in the signal at all. See
# qc/label_intrinsic_reward_simplerenv.py's --reward-mode flag.
#
# The other two ablation arms need no new labeling at all:
#   - subgoal-only: qc_cache_simplerenv.npz (already produced by
#     label_train_simplerenv_qc_critic.slurm's labeling stage) -- train a
#     critic on it directly.
#   - combined: qc_cache_simplerenv_intrinsic.npz (the "real" pipeline,
#     label_intrinsic_reward_simplerenv.slurm + train_simplerenv_qc_critic_intrinsic.slurm).
#
# PREREQUISITE: label_train_simplerenv_qc_critic.slurm must have already
# produced $INPUT_CACHE (defaults to that job's cache path).
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/label_intrinsic_only_reward_simplerenv.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/label_intrinsic_only_reward_simplerenv.slurm
#
# After this finishes, train the ablation-arm critic with:
#   sbatch ~/vla-jepa-qc/slurm/train_simplerenv_qc_critic_intrinsic_only.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_simplerenv_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
CKPT_DIR="$WORK_DIR/checkpoints"
OUT_DIR="$WORK_DIR/output"

SIMPLERENV_CKPT="${SIMPLERENV_CKPT:-$CKPT_DIR/VLA-JEPA-SimplerEnv-repo/SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt}"
INPUT_CACHE="${INPUT_CACHE:-$OUT_DIR/qc_cache_simplerenv.npz}"
OUTPUT_CACHE="${OUTPUT_CACHE:-$OUT_DIR/qc_cache_simplerenv_intrinsic_only.npz}"

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

echo "=== Labeling intrinsic-only reward (subgoal zeroed out) on top of $INPUT_CACHE ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/label_intrinsic_reward_simplerenv.py \
    --ckpt-path "$SIMPLERENV_CKPT" \
    --input-cache "$INPUT_CACHE" \
    --output-path "$OUTPUT_CACHE" \
    --reward-mode intrinsic_only \
    --checkpoint-every 25 --cuda 0

echo "Done. intrinsic_only cache: $OUTPUT_CACHE"
