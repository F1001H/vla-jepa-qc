#!/bin/bash
#SBATCH --job-name=smoketest_libero_qc_pipeline
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=00:30:00                  # tiny workload (3 episodes through each stage) -- generous
                                          # margin is for a cold checkpoint/dataset download, not compute
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail   # NOT -e: want every stage's own pass/fail reported, not an early bail

# 1-GPU smoke test for the LIBERO QC pipeline, meant to run BEFORE
# submitting slurm/label_train_libero_qc_critic_8gpu.slurm -- that job
# requests a full 8-GPU node (potentially a real queue wait) and takes
# ~1.5-2h; better to catch a broken checkpoint config, a bad path, or a
# missing dependency here first, on 1 GPU, in minutes.
#
# Also pre-warms the checkpoint/dataset download + config patch (shares
# the exact same $CKPT_DIR/$DATASET_DIR as the 8-GPU job) -- so once this
# passes, the big job's own download step is a no-op.
#
# Three stages, --max-episodes 3 throughout (just enough to exercise the
# real code paths, not a real training signal):
#   1. qc/label_rewards.py       (subgoal labeling, fixed detector)
#   2. qc/label_intrinsic_reward.py (intrinsic reward on top)
#   3. qc/train_critic.py --epochs 1 (critic training)
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/smoketest_libero_qc_pipeline.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/smoketest_libero_qc_pipeline.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_libero_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
CKPT_DIR="$WORK_DIR/checkpoints"
DATASET_DIR="$WORK_DIR/datasets/physical-intelligence-libero"
OUT_DIR="$WORK_DIR/output"
LOG_DIR="$WORK_DIR/logs"

LIBERO_CKPT="$CKPT_DIR/VLA-JEPA-LIBERO-repo/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
SMOKE_SUBGOAL_CACHE="$OUT_DIR/qc_cache_libero_smoketest.npz"
SMOKE_INTRINSIC_CACHE="$OUT_DIR/qc_cache_libero_smoketest_intrinsic.npz"
SMOKE_CRITIC_OUT="$OUT_DIR/vlajepa_critic_libero_smoketest.pt"

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

# --- Download + patch (identical to label_train_libero_qc_critic_8gpu.slurm --
# shares the same $CKPT_DIR, so this either does the real work once or is a
# fast no-op if the 8-GPU job (or a previous smoke test) already did it) ---
CKPT_PATCHED_MARKER="$CKPT_DIR/.libero_ckpt_patched"
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
# WORKSPACE is a belt-and-suspenders catch-all for any ${oc.env:WORKSPACE}
# interpolation the patch below doesn't cover -- see
# label_train_libero_qc_critic_8gpu.slurm's matching comment for why.
export WORKSPACE="$WORK_DIR"
if [ ! -f "$CKPT_PATCHED_MARKER" ]; then
    echo "=== Downloading base Qwen3-VL-2B-Instruct + V-JEPA2 checkpoints ==="
    "$PY" -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-VL-2B-Instruct \
        --local-dir "$CKPT_DIR/Qwen3-VL-2B-Instruct"
    "$PY" -m huggingface_hub.commands.huggingface_cli download facebook/vjepa2-vitl-fpc64-256 \
        --local-dir "$CKPT_DIR/vjepa2-vitl-fpc64-256"

    echo "=== Patching LIBERO checkpoint config ==="
    python3 - "$CKPT_DIR" "$DATASET_DIR" <<'PYEOF'
import json
import sys
ckpt_dir, dataset_dir = sys.argv[1], sys.argv[2]
cfg_path = f"{ckpt_dir}/VLA-JEPA-LIBERO-repo/LIBERO/config.json"
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["framework"]["qwenvl"]["base_vlm"] = f"{ckpt_dir}/Qwen3-VL-2B-Instruct"
cfg["framework"]["qwenvl"]["attn_implementation"] = "sdpa"
cfg["framework"]["vj2_model"]["base_encoder"] = f"{ckpt_dir}/vjepa2-vitl-fpc64-256"
cfg["datasets"]["vla_data"]["data_root_dir"] = dataset_dir
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=4)
print("Patched LIBERO config.json")
PYEOF
    python3 - "$CKPT_DIR" "$DATASET_DIR" <<'PYEOF'
import re
import sys
ckpt_dir, dataset_dir = sys.argv[1], sys.argv[2]
cfg_path = f"{ckpt_dir}/VLA-JEPA-LIBERO-repo/LIBERO/config.yaml"
text = open(cfg_path).read()
text = re.sub(r"base_vlm: .*", f"base_vlm: {ckpt_dir}/Qwen3-VL-2B-Instruct", text, count=1)
text = re.sub(r"attn_implementation: .*", "attn_implementation: sdpa", text, count=1)
text = re.sub(r"base_encoder: .*", f"base_encoder: {ckpt_dir}/vjepa2-vitl-fpc64-256", text, count=1)
text = re.sub(r"data_root_dir: .*", f"data_root_dir: {dataset_dir}", text, count=1)
open(cfg_path, "w").write(text)
print("Patched LIBERO config.yaml")
PYEOF
    remaining=$(grep -o '\${[^}]*}' "$CKPT_DIR/VLA-JEPA-LIBERO-repo/LIBERO/config.yaml" || true)
    if [ -n "$remaining" ]; then
        echo "WARNING: config.yaml still has unresolved interpolations after patching: $remaining" >&2
        echo "(WORKSPACE env var is set as a catch-all, so this may still work -- but check if it doesn't.)" >&2
    fi
    touch "$CKPT_PATCHED_MARKER"
fi

# --- Stage 1: tiny subgoal labeling ---
echo "=== [$(date)] Stage 1: subgoal labeling (3 episodes) ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/label_rewards.py \
    --dataset-root "$DATASET_DIR" --ckpt-path "$LIBERO_CKPT" \
    --output-path "$SMOKE_SUBGOAL_CACHE" \
    --max-episodes 3 --checkpoint-every 1 --cuda 0
stage1_status=$?
if [ "$stage1_status" -ne 0 ]; then
    echo "=== STAGE 1 (subgoal labeling) FAILED -- see above ===" >&2
    exit 1
fi
echo "=== Stage 1 PASSED ==="

# --- Stage 2: tiny intrinsic labeling ---
echo "=== [$(date)] Stage 2: intrinsic labeling (3 episodes) ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/label_intrinsic_reward.py \
    --dataset-root "$DATASET_DIR" --ckpt-path "$LIBERO_CKPT" \
    --input-cache "$SMOKE_SUBGOAL_CACHE" \
    --output-path "$SMOKE_INTRINSIC_CACHE" \
    --max-episodes 3 --checkpoint-every 1 --cuda 0
stage2_status=$?
if [ "$stage2_status" -ne 0 ]; then
    echo "=== STAGE 2 (intrinsic labeling) FAILED -- see above ===" >&2
    exit 1
fi
echo "=== Stage 2 PASSED ==="

# --- Stage 3: tiny critic training ---
echo "=== [$(date)] Stage 3: critic training (1 epoch) ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/train_critic.py \
    --cache-path "$SMOKE_INTRINSIC_CACHE" \
    --output-path "$SMOKE_CRITIC_OUT" \
    --epochs 1 --batch-size 8 --num-qs 5 --cuda 0
stage3_status=$?
if [ "$stage3_status" -ne 0 ]; then
    echo "=== STAGE 3 (critic training) FAILED -- see above ===" >&2
    exit 1
fi
echo "=== Stage 3 PASSED ==="

echo "=== [$(date)] ALL SMOKE TEST STAGES PASSED -- safe to submit the full 8-GPU job: ==="
echo "  sbatch ~/vla-jepa-qc/slurm/label_train_libero_qc_critic_8gpu.slurm"
