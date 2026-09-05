#!/bin/bash
#SBATCH --job-name=relabel_temp_boosted_train_libero_qc_critic_8gpu
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:8                # full node -- sharding the labeling pass across all 8 GPUs
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=112              # full node's CPU count (14/GPU, matching the 1-GPU jobs' ratio)
#SBATCH --mem=1800G                      # generous headroom under the node's ~2TB -- 8 concurrent
                                          # VLA-JEPA model copies (Qwen3-VL-2B + V-JEPA2-ViT-L each)
#SBATCH --time=04:00:00                  # same pace as label_train_libero_qc_critic_8gpu.slurm --
                                          # candidate_temperature only changes the flow-matching sampler's
                                          # initial noise draw, not the per-frame Qwen forward pass cost
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail   # NOT -e: need to check each of the 8 parallel shards' own exit status

# Relabels the LIBERO QC cache with candidates sampled at
# CANDIDATE_TEMPERATURE (default 1.3, the best fixed-temperature setting
# found in the isolation-ablation investigation, see
# ~/openpi/qc_ablation_log.md's "Candidate-diversity ablations" table) --
# NOT 1.0 like label_train_libero_qc_critic_8gpu.slurm's existing cache.
#
# Motivation: the deployed critic (vlajepa_critic_intrinsic.pt) was trained
# on candidates sampled at the default temperature=1.0, where candidates
# are nearly identical in raw action space (median std 0.0031) -- but
# best_of_n_action's own candidate_temperature/candidate_temperature_spread
# knobs sample MORE diverse candidates at eval time. That's a train/eval
# distribution mismatch: the critic never saw candidates as spread out as
# the ones it's asked to score during a temperature-boosted eval. This job
# retrains a critic on candidates matching that eval-time distribution
# instead, to test whether closing that gap helps (a different critic
# architecture, the dueling critic, was already tried as of 2026-09-05 and
# gave a clean NEGATIVE result -- see section 12 of
# ~/openpi/qc_and_vlajepa_test_writeup.md -- so this targets the sampling
# side of the problem instead of the architecture side).
#
# Structurally identical to label_train_libero_qc_critic_8gpu.slurm
# (3-stage subgoal -> intrinsic -> train_critic pipeline, same sharding/
# threading setup) with two differences: (1) stage 1 passes
# --candidate-temperature to qc/label_rewards.py (added 2026-09-05 --
# previously hardcoded to the function default of 1.0), and (2) every
# output path gets a `_temp_boosted` suffix so this run's cache/critic
# don't clobber the existing temperature=1.0 ones -- both remain available
# for comparison.
#
# Reuses the SAME downloaded checkpoint/dataset/venv as the existing
# label_train_libero_qc_critic_8gpu.slurm run (same WORK_DIR) -- the
# download/patch block below is a no-op if that job already ran on this
# cluster.
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/relabel_temp_boosted_train_libero_qc_critic_8gpu.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/relabel_temp_boosted_train_libero_qc_critic_8gpu.slurm
# Override the temperature with:
#   CANDIDATE_TEMPERATURE=1.6 sbatch ...
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
CANDIDATE_TEMPERATURE="${CANDIDATE_TEMPERATURE:-1.3}"
# CRITICAL: without this, each shard process defaults to using ALL visible
# cores for its own internal thread pools (PyTorch/OpenMP, NumPy's
# OpenBLAS/MKL, PyArrow's multi-threaded parquet reading) -- with 8 shards
# doing that simultaneously on one node, they thrash each other for the
# same 112 cores instead of getting ~14 each. Observed in practice: 0%
# GPU utilization, 98% CPU in userspace, each process at 1200-1600% CPU,
# load average 338 (vs 112 real cores) -- a ~30x slowdown (246s/episode
# instead of the expected ~10-20s) that looked like an I/O bottleneck but
# was pure CPU oversubscription.
THREADS_PER_SHARD=$((112 / NUM_SHARDS))

LIBERO_CKPT="$CKPT_DIR/VLA-JEPA-LIBERO-repo/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1

mkdir -p "$WORK_DIR" "$CKPT_DIR" "$DATASET_DIR" "$OUT_DIR" "$LOG_DIR" "$WORK_DIR/slurm_logs"

echo "=== GPU diagnostics ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv
echo "candidate_temperature=$CANDIDATE_TEMPERATURE"
echo "========================"

cd "$REPO_DIR"

if [ ! -f ".venv/.setup_complete" ]; then
    echo "ERROR: $REPO_DIR/.venv is missing or incomplete. Build it from a login node first -- see label_train_simplerenv_qc_critic.slurm's header comment for the exact steps." >&2
    exit 1
fi
PY="$REPO_DIR/.venv/bin/python"

# --- 0. Download checkpoint + dataset if missing (idempotent, shared with
# label_train_libero_qc_critic_8gpu.slurm's WORK_DIR -- no-op if already done) ---
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

# --- Base backbone checkpoints + config patch ---
# See label_train_libero_qc_critic_8gpu.slurm's identical block for the
# full rationale (${oc.env:WORKSPACE}-style interpolations in the LIBERO
# checkpoint's config need rewriting to plain absolute paths).
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

# --- Stage 1: sharded subgoal + temp-boosted-candidate labeling, 8 parallel GPUs ---
echo "=== [$(date)] Stage 1: sharded subgoal labeling (candidate_temperature=$CANDIDATE_TEMPERATURE) ==="
subgoal_shards=()
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_path="$OUT_DIR/qc_cache_libero_temp_boosted_subgoal_shard${i}of${NUM_SHARDS}.npz"
    subgoal_shards+=("$shard_path")
    CUDA_VISIBLE_DEVICES="$i" \
    OMP_NUM_THREADS="$THREADS_PER_SHARD" MKL_NUM_THREADS="$THREADS_PER_SHARD" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_SHARD" NUMEXPR_NUM_THREADS="$THREADS_PER_SHARD" \
    PYTHONPATH="$REPO_DIR" "$PY" qc/label_rewards.py \
        --dataset-root "$DATASET_DIR" --ckpt-path "$LIBERO_CKPT" \
        --output-path "$shard_path" \
        --candidate-temperature "$CANDIDATE_TEMPERATURE" \
        --shard-id "$i" --num-shards "$NUM_SHARDS" \
        --checkpoint-every 25 --cuda 0 \
        > "$LOG_DIR/label_temp_boosted_subgoal_shard${i}.log" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "ERROR: one or more subgoal-labeling shards failed -- check $LOG_DIR/label_temp_boosted_subgoal_shard*.log" >&2
    exit 1
fi
echo "=== [$(date)] Stage 1 done, merging shards ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/merge_shards.py --mode subgoal \
    --shard-paths "${subgoal_shards[@]}" \
    --output-path "$OUT_DIR/qc_cache_libero_temp_boosted_full.npz"

# --- Stage 2: sharded intrinsic labeling, 8 parallel GPUs ---
# label_intrinsic_reward.py only replaces the reward array -- it reads
# candidates from --input-cache (the temp-boosted cache from Stage 1) and
# leaves them untouched, so no --candidate-temperature flag needed here.
echo "=== [$(date)] Stage 2: sharded intrinsic labeling ==="
intrinsic_shards=()
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_path="$OUT_DIR/qc_cache_libero_temp_boosted_intrinsic_shard${i}of${NUM_SHARDS}.npz"
    intrinsic_shards+=("$shard_path")
    CUDA_VISIBLE_DEVICES="$i" \
    OMP_NUM_THREADS="$THREADS_PER_SHARD" MKL_NUM_THREADS="$THREADS_PER_SHARD" \
    OPENBLAS_NUM_THREADS="$THREADS_PER_SHARD" NUMEXPR_NUM_THREADS="$THREADS_PER_SHARD" \
    PYTHONPATH="$REPO_DIR" "$PY" qc/label_intrinsic_reward.py \
        --dataset-root "$DATASET_DIR" --ckpt-path "$LIBERO_CKPT" \
        --input-cache "$OUT_DIR/qc_cache_libero_temp_boosted_full.npz" \
        --output-path "$shard_path" \
        --shard-id "$i" --num-shards "$NUM_SHARDS" \
        --checkpoint-every 25 --cuda 0 \
        > "$LOG_DIR/label_temp_boosted_intrinsic_shard${i}.log" 2>&1 &
    pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "ERROR: one or more intrinsic-labeling shards failed -- check $LOG_DIR/label_temp_boosted_intrinsic_shard*.log" >&2
    exit 1
fi
echo "=== [$(date)] Stage 2 done, merging shards ==="
PYTHONPATH="$REPO_DIR" "$PY" qc/merge_shards.py --mode intrinsic \
    --shard-paths "${intrinsic_shards[@]}" \
    --output-path "$OUT_DIR/qc_cache_libero_temp_boosted_intrinsic_full.npz"

# --- Stage 3: train critic (single GPU) ---
# Same hyperparameters as the existing vlajepa_critic_intrinsic.pt
# (discount=0.99 default, q_agg=mean default, epochs=20, batch-size=256,
# num-qs=5) -- only the candidate distribution in the cache differs, so
# this is a clean apples-to-apples comparison isolating that one variable.
echo "=== [$(date)] Stage 3: training critic ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_DIR" "$PY" qc/train_critic.py \
    --cache-path "$OUT_DIR/qc_cache_libero_temp_boosted_intrinsic_full.npz" \
    --output-path "$OUT_DIR/vlajepa_critic_libero_temp_boosted_intrinsic.pt" \
    --epochs 20 --batch-size 256 --num-qs 5 --cuda 0

echo "=== [$(date)] ALL DONE. Critic: $OUT_DIR/vlajepa_critic_libero_temp_boosted_intrinsic.pt ==="
echo "rsync it back locally with:"
echo "  rsync -avz \$(hostname):$OUT_DIR/vlajepa_critic_libero_temp_boosted_intrinsic.pt ~/starvla_jepa/"
