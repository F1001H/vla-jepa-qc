#!/bin/bash
#SBATCH --job-name=eval_libero_plus_qc_sequential
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1                # single GPU shared by the policy server + MuJoCo/EGL rendering
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=48:00:00                  # full LIBERO-Plus is 10,026 tasks across 7 categories --
                                          # observed locally: ~1-2.5h/category except Sensor Noise
                                          # (CPU-bound motion-blur, up to ~13h) -- budget very generously
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail

# LIBERO-Plus (7-category robustness benchmark) on the cluster -- the
# actual target benchmark for this investigation (per Fabian: "beat
# baselines on libero plus"). Mirrors eval_vlajepa_libero_plus_sequential.sh
# (already proven locally over a 20+ hour run, including the
# timeout/result-extraction fixes baked in below) and reuses the same
# LIBERO checkpoint/sim-venv setup as eval_libero_qc_sequential_cluster.slurm
# (standard LIBERO) and label_train_libero_qc_critic_8gpu.slurm.
#
# ADDITIONAL PREREQUISITE beyond standard LIBERO: the Sensor Noise
# category's motion-blur corruption needs ImageMagick's libmagickwand
# SYSTEM library (not just the `Wand` python wheel, which this job installs
# itself) -- this needs `apt-get install libmagickwand-dev` (sudo) on the
# node's image, or an equivalent already baked into the cluster's compute
# node image. I don't have cluster admin access to confirm this is present.
# This job checks for it and FAILS CLEARLY with next steps if missing,
# attempting a no-sudo conda-forge fallback first (see the check below) --
# it does NOT silently hang or produce a cryptic traceback deep in a run.
#
# PREREQUISITE: the LIBERO checkpoint must already exist (from
# label_train_libero_qc_critic_8gpu.slurm, which downloads + patches it).
#
# Set CRITIC_PATH to run the QC variant instead of plain BC.
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/eval_libero_plus_qc_sequential_cluster.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/eval_libero_plus_qc_sequential_cluster.slurm
#
# For a QC run, e.g. (matching the best ablation setting from the standard-
# LIBERO sweep):
#   CRITIC_PATH=$OUT_DIR/vlajepa_critic_libero_intrinsic.pt RUN_TAG=intrinsic \
#   MAXIMIZE_SCORE=true UNCERTAINTY_PENALTY=1.0 ACTOR_DISAGREEMENT_PENALTY=1.0 \
#   sbatch ~/vla-jepa-qc/slurm/eval_libero_plus_qc_sequential_cluster.slurm
#
# For a quick subsampled sanity check before committing to the full
# 10,026-task run:
#   MAX_TASKS_PER_SUITE=100 sbatch ~/vla-jepa-qc/slurm/eval_libero_plus_qc_sequential_cluster.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_libero_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
CKPT_DIR="$WORK_DIR/checkpoints"
OUT_DIR="$WORK_DIR/output"
LOG_DIR="$WORK_DIR/logs"

LIBERO_CKPT="$CKPT_DIR/VLA-JEPA-LIBERO-repo/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
SIM_VENV_DIR="$WORK_DIR/libero_sim_venv"
LIBERO_PLUS_BENCH_DIR="$WORK_DIR/libero_sim/LIBERO-plus"
LIBERO_PLUS_CONFIG_DIR="$WORK_DIR/libero_sim/.libero_plus_config"

CRITIC_PATH="${CRITIC_PATH:-}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
NUM_QS="${NUM_QS:-5}"
Q_AGG="${Q_AGG:-mean}"
UNCERTAINTY_PENALTY="${UNCERTAINTY_PENALTY:-0.0}"
ACTOR_DISAGREEMENT_PENALTY="${ACTOR_DISAGREEMENT_PENALTY:-0.0}"
CRITIC_WEIGHT="${CRITIC_WEIGHT:-1.0}"
MAXIMIZE_SCORE="${MAXIMIZE_SCORE:-}"
SELECTION_MODE="${SELECTION_MODE:-score}"
MAX_TASKS_PER_SUITE="${MAX_TASKS_PER_SUITE:-0}"   # 0 = full (10,026 tasks); set e.g. 100 for a quick check
CATEGORY_TIMEOUT_S="${CATEGORY_TIMEOUT_S:-86400}" # Sensor Noise needs ~13h+ at worst-case pace, see header
RUN_TAG="${RUN_TAG:-plain}"

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p "$WORK_DIR" "$CKPT_DIR" "$OUT_DIR" "$LOG_DIR" "$WORK_DIR/libero_sim" "$LIBERO_PLUS_CONFIG_DIR" "$WORK_DIR/slurm_logs"

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
SERVER_PY="$REPO_DIR/.venv/bin/python"

if [ ! -f "$LIBERO_CKPT" ]; then
    echo "ERROR: LIBERO checkpoint not found under $WORK_DIR -- run label_train_libero_qc_critic_8gpu.slurm first (it downloads + patches it)." >&2
    exit 1
fi
if [ ! -f "$SIM_VENV_DIR/.setup_complete" ]; then
    echo "ERROR: $SIM_VENV_DIR is missing. Run eval_libero_qc_sequential_cluster.slurm (standard LIBERO) or smoketest_libero_qc_pipeline.slurm first -- either builds this same shared sim venv." >&2
    exit 1
fi
SIM_PY="$SIM_VENV_DIR/bin/python"

# --- libmagickwand SYSTEM library check (Sensor Noise category needs it) ---
echo "=== Checking for libmagickwand ==="
if ! ldconfig -p 2>/dev/null | grep -qi "libMagickWand"; then
    echo "libMagickWand not found via ldconfig -- trying conda-forge as a no-sudo fallback..."
    if command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; then
        CONDA_ENV_DIR="$WORK_DIR/libmagickwand_conda_env"
        if [ ! -d "$CONDA_ENV_DIR" ]; then
            (command -v mamba >/dev/null 2>&1 && mamba create -y -p "$CONDA_ENV_DIR" -c conda-forge imagemagick) \
                || conda create -y -p "$CONDA_ENV_DIR" -c conda-forge imagemagick
        fi
        export LD_LIBRARY_PATH="$CONDA_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
        if ! (ldconfig -p 2>/dev/null | grep -qi "libMagickWand"; find "$CONDA_ENV_DIR/lib" -iname "libMagickWand*" 2>/dev/null | grep -q .); then
            echo "ERROR: conda-forge fallback didn't produce a usable libMagickWand either. Sensor Noise category needs libmagickwand-dev installed system-wide (sudo apt-get install libmagickwand-dev) -- ask cluster admins, or skip Sensor Noise by removing it from CATEGORIES below." >&2
            exit 1
        fi
        echo "Using conda-forge imagemagick at $CONDA_ENV_DIR"
    else
        echo "ERROR: libMagickWand missing and no conda/mamba available for a no-sudo fallback. Sensor Noise category needs libmagickwand-dev installed system-wide -- ask cluster admins, or skip Sensor Noise by removing it from CATEGORIES below." >&2
        exit 1
    fi
fi
uv pip install --python "$SIM_PY" Wand==0.7.2

# --- LIBERO-Plus benchmark clone + assets (separate from base LIBERO's clone) ---
if [ ! -d "$LIBERO_PLUS_BENCH_DIR" ]; then
    echo "=== Cloning LIBERO-Plus benchmark ==="
    git clone https://github.com/sylvestf/LIBERO-plus.git "$LIBERO_PLUS_BENCH_DIR"
fi
if [ ! -d "$LIBERO_PLUS_BENCH_DIR/libero/libero/assets" ]; then
    echo "=== Downloading + unzipping LIBERO-Plus assets ==="
    "$REPO_DIR/.venv/bin/python" -m huggingface_hub.commands.huggingface_cli download Sylvest/LIBERO-plus \
        --repo-type dataset --include "assets.zip" --local-dir "$WORK_DIR/libero_sim"
    unzip -q -o "$WORK_DIR/libero_sim/assets.zip" -d "$LIBERO_PLUS_BENCH_DIR/libero/libero"
fi

# --- LIBERO-Plus config.yaml (separate from base LIBERO's, points at the
# LIBERO-Plus clone's own assets/bddl_files) ---
if [ ! -f "$LIBERO_PLUS_CONFIG_DIR/config.yaml" ]; then
    echo "=== Generating LIBERO-Plus config.yaml ==="
    "$SIM_PY" -c "
import sys, yaml
sys.path.insert(0, '$LIBERO_PLUS_BENCH_DIR')
from libero.libero import get_default_path_dict
paths = get_default_path_dict('$LIBERO_PLUS_BENCH_DIR/libero/libero')
with open('$LIBERO_PLUS_CONFIG_DIR/config.yaml', 'w') as f:
    yaml.dump(paths, f)
print(paths)
"
fi

# --- Sequential 7-category eval ---
RESULTS_FILE="$WORK_DIR/vlajepa_libero_plus_${RUN_TAG}_eval_results.txt"
touch "$RESULTS_FILE"

CATEGORIES=("Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise")
base_port=15283

wait_for_port () {
    local port="$1"
    local timeout_s=600
    local waited=0
    while ! "$SERVER_PY" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('127.0.0.1', $port))
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "=== [$(date)] Server process died during startup on port $port ===" | tee -a "$RESULTS_FILE"
            return 1
        fi
        sleep 3
        waited=$((waited+3))
        if [ "$waited" -ge "$timeout_s" ]; then
            echo "=== [$(date)] Timed out waiting for server on port $port ===" | tee -a "$RESULTS_FILE"
            return 1
        fi
    done
    return 0
}

index=0
for category in "${CATEGORIES[@]}"; do
    index=$((index+1))
    port=$((base_port+index))
    cat_tag="${category// /_}"

    if grep -q "^${category}:.*# successes" "$RESULTS_FILE" 2>/dev/null; then
        echo "=== [$(date)] Skipping already-done category: $category ===" | tee -a "$LOG_DIR/driver_libero_plus_${RUN_TAG}.log"
        continue
    fi

    echo "=== [$(date)] Starting category: $category (port=$port) ===" | tee -a "$LOG_DIR/driver_libero_plus_${RUN_TAG}.log"

    cd "$REPO_DIR"
    if [ -n "$CRITIC_PATH" ]; then
        PYTHONPATH="$REPO_DIR" "$SERVER_PY" deployment/model_server/server_qc_policy.py \
            --ckpt_path "$LIBERO_CKPT" --critic_checkpoint_path "$CRITIC_PATH" \
            --num_samples "$NUM_SAMPLES" --num_qs "$NUM_QS" \
            --q_agg "$Q_AGG" --uncertainty_penalty "$UNCERTAINTY_PENALTY" \
            --actor_disagreement_penalty "$ACTOR_DISAGREEMENT_PENALTY" \
            --critic_weight "$CRITIC_WEIGHT" --selection_mode "$SELECTION_MODE" \
            ${MAXIMIZE_SCORE:+--maximize_score} \
            --port "$port" --cuda 0 \
            > "$LOG_DIR/server_libero_plus_${RUN_TAG}_${cat_tag}.log" 2>&1 &
    else
        PYTHONPATH="$REPO_DIR" "$SERVER_PY" deployment/model_server/server_policy.py \
            --ckpt_path "$LIBERO_CKPT" --port "$port" --cuda 0 \
            > "$LOG_DIR/server_libero_plus_${RUN_TAG}_${cat_tag}.log" 2>&1 &
    fi
    SERVER_PID=$!

    if ! wait_for_port "$port"; then
        echo "$category: SERVER FAILED TO START (see $LOG_DIR/server_libero_plus_${RUN_TAG}_${cat_tag}.log)" | tee -a "$RESULTS_FILE"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        continue
    fi

    mkdir -p "$REPO_DIR/results/libero_plus_${RUN_TAG}_${cat_tag}"
    PYTHONPATH="$REPO_DIR:$LIBERO_PLUS_BENCH_DIR" \
    LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG_DIR" \
    timeout "$CATEGORY_TIMEOUT_S" "$SIM_PY" examples/LIBERO/eval_libero.py \
        --args.pretrained-path "$LIBERO_CKPT" \
        --args.host 127.0.0.1 \
        --args.port "$port" \
        --args.task-suite-name libero_mix \
        --args.category_value "$category" \
        --args.num-trials-per-task 1 \
        --args.max-tasks-per-suite "$MAX_TASKS_PER_SUITE" \
        --args.video-out-path "results/libero_plus_${RUN_TAG}_${cat_tag}" \
        --args.with-state true \
        > "$LOG_DIR/eval_libero_plus_${RUN_TAG}_${cat_tag}.log" 2>&1
    status=$?

    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    sleep 5

    if [ "$status" -ne 0 ]; then
        echo "$category: EVAL FAILED or TIMED OUT (see $LOG_DIR/eval_libero_plus_${RUN_TAG}_${cat_tag}.log)" | tee -a "$RESULTS_FILE"
        continue
    fi

    success_line="$(grep '# successes' "$LOG_DIR/eval_libero_plus_${RUN_TAG}_${cat_tag}.log" | tail -1)"
    echo "$category: $success_line" | tee -a "$RESULTS_FILE"
    echo "=== [$(date)] Finished category: $category ===" | tee -a "$LOG_DIR/driver_libero_plus_${RUN_TAG}.log"
done

echo "=== [$(date)] ALL LIBERO-PLUS CATEGORIES DONE ===" | tee -a "$LOG_DIR/driver_libero_plus_${RUN_TAG}.log"
echo "Results: $RESULTS_FILE"
