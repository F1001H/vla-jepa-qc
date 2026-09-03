#!/bin/bash
#SBATCH --job-name=eval_libero_plus_qc_parallel
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:4                # default 4 -- override at submit time if NUM_GPUS is overridden too,
                                          # SBATCH directives can't read shell vars, see submit examples below
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=56               # 14/GPU x 4, matching the 1-GPU jobs' ratio
#SBATCH --mem=256G
#SBATCH --time=24:00:00                  # 7 categories over ~4 GPUs, worst-case (Sensor Noise, ~13h)
                                          # landing on one worker -- generous margin
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail   # NOT -e: need to check each parallel worker's own exit status

# Parallel-category variant of eval_libero_plus_qc_sequential_cluster.slurm
# -- splits the 7 LIBERO-Plus categories across NUM_GPUS workers (static
# round-robin by category index, same convention as
# label_train_libero_qc_critic_8gpu.slurm's --shard-id/--num-shards) instead
# of running all 7 sequentially on 1 GPU. Meant for comparing multiple
# critics: submit this job once per critic (each with its own CRITIC_PATH/
# RUN_TAG), and SLURM will place them on the node(s) as GPUs free up --
# e.g. two 4-GPU submissions for two critics fit on one 8-GPU node together.
#
# NOTE on load balance: round-robin assignment is NOT aware that Sensor
# Noise (CPU-bound motion-blur) takes ~5-10x longer than the other
# categories -- whichever worker draws it will finish noticeably later
# than the other NUM_GPUS-1 workers. Good enough for getting a full
# 7-category result faster than sequential, not perfectly load-balanced.
#
# ADDITIONAL PREREQUISITE beyond standard LIBERO: Sensor Noise needs
# libmagickwand (system library, sudo-installed) -- same as the sequential
# version, checked once up front and if missing, Sensor Noise is dropped
# from every worker's category list (not just skipped mid-run).
#
# PREREQUISITE: the LIBERO checkpoint + shared sim venv must already exist
# (from label_train_libero_qc_critic_8gpu.slurm / eval_libero_qc_sequential_cluster.slurm).
#
# NUM_GPUS must match --gres -- SBATCH directives are static and can't read
# NUM_GPUS, so override BOTH when submitting with a different count:
#   sbatch --gres=gpu:h100:2 --cpus-per-task=28 --mem=128G \
#       --export=ALL,NUM_GPUS=2,CRITIC_PATH=...,RUN_TAG=... \
#       slurm/eval_libero_plus_qc_parallel_cluster.slurm
#
# Default (4 GPUs), e.g. for the best ablation setting:
#   CRITIC_PATH=$OUT_DIR/vlajepa_critic_libero_intrinsic.pt RUN_TAG=intrinsic \
#   MAXIMIZE_SCORE=true UNCERTAINTY_PENALTY=1.0 ACTOR_DISAGREEMENT_PENALTY=1.0 \
#   sbatch ~/vla-jepa-qc/slurm/eval_libero_plus_qc_parallel_cluster.slurm
#
# To run two critics side by side on one 8-GPU node, submit twice with
# NUM_GPUS=4 each and distinct RUN_TAGs/CRITIC_PATHs -- no special
# coordination needed, SLURM schedules both onto whatever GPUs are free.

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

NUM_GPUS="${NUM_GPUS:-4}"
CRITIC_PATH="${CRITIC_PATH:-}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
NUM_QS="${NUM_QS:-5}"
Q_AGG="${Q_AGG:-mean}"
UNCERTAINTY_PENALTY="${UNCERTAINTY_PENALTY:-0.0}"
ACTOR_DISAGREEMENT_PENALTY="${ACTOR_DISAGREEMENT_PENALTY:-0.0}"
CRITIC_WEIGHT="${CRITIC_WEIGHT:-1.0}"
MAXIMIZE_SCORE="${MAXIMIZE_SCORE:-}"
SELECTION_MODE="${SELECTION_MODE:-score}"
MAX_TASKS_PER_SUITE="${MAX_TASKS_PER_SUITE:-0}"
CATEGORY_TIMEOUT_S="${CATEGORY_TIMEOUT_S:-86400}"
RUN_TAG="${RUN_TAG:-plain}"

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p "$WORK_DIR" "$CKPT_DIR" "$OUT_DIR" "$LOG_DIR" "$WORK_DIR/libero_sim" "$LIBERO_PLUS_CONFIG_DIR" "$WORK_DIR/slurm_logs"

echo "=== GPU diagnostics ==="
echo "Node: $(hostname)  NUM_GPUS=$NUM_GPUS"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv
echo "========================"

cd "$REPO_DIR"
if [ ! -f ".venv/.setup_complete" ]; then
    echo "ERROR: $REPO_DIR/.venv is missing or incomplete. Build it from a login node first." >&2
    exit 1
fi
SERVER_PY="$REPO_DIR/.venv/bin/python"

if [ ! -f "$LIBERO_CKPT" ]; then
    echo "ERROR: LIBERO checkpoint not found under $WORK_DIR -- run label_train_libero_qc_critic_8gpu.slurm first." >&2
    exit 1
fi
if [ ! -f "$SIM_VENV_DIR/.setup_complete" ]; then
    echo "ERROR: $SIM_VENV_DIR is missing." >&2
    exit 1
fi
SIM_PY="$SIM_VENV_DIR/bin/python"
if [ -n "$CRITIC_PATH" ] && [ ! -f "$CRITIC_PATH" ]; then
    echo "ERROR: CRITIC_PATH=$CRITIC_PATH does not exist." >&2
    exit 1
fi

# --- libmagickwand check (same as sequential version -- see that file's
# header for why this can't just fail the whole job) ---
echo "=== Checking for libmagickwand ==="
HAVE_LIBMAGICKWAND=1
if ! ldconfig -p 2>/dev/null | grep -qi "libMagickWand"; then
    if command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; then
        CONDA_ENV_DIR="$WORK_DIR/libmagickwand_conda_env"
        if [ ! -d "$CONDA_ENV_DIR" ]; then
            (command -v mamba >/dev/null 2>&1 && mamba create -y -p "$CONDA_ENV_DIR" -c conda-forge imagemagick) \
                || conda create -y -p "$CONDA_ENV_DIR" -c conda-forge imagemagick
        fi
        export LD_LIBRARY_PATH="$CONDA_ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
        if ! (ldconfig -p 2>/dev/null | grep -qi "libMagickWand"; find "$CONDA_ENV_DIR/lib" -iname "libMagickWand*" 2>/dev/null | grep -q .); then
            HAVE_LIBMAGICKWAND=0
        fi
    else
        HAVE_LIBMAGICKWAND=0
    fi
fi
if [ "$HAVE_LIBMAGICKWAND" -eq 0 ]; then
    echo "WARNING: libMagickWand unavailable -- SKIPPING Sensor Noise on every worker." >&2
else
    uv pip install --python "$SIM_PY" Wand==0.7.2
fi

# --- LIBERO-Plus benchmark clone + assets (shared across workers) ---
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
if [ ! -f "$LIBERO_PLUS_CONFIG_DIR/config.yaml" ]; then
    echo "=== Generating LIBERO-Plus config.yaml ==="
    "$SIM_PY" -c "
import sys, yaml
sys.path.insert(0, '$LIBERO_PLUS_BENCH_DIR')
from libero.libero import get_default_path_dict
paths = get_default_path_dict('$LIBERO_PLUS_BENCH_DIR/libero/libero')
with open('$LIBERO_PLUS_CONFIG_DIR/config.yaml', 'w') as f:
    yaml.dump(paths, f)
"
fi

RESULTS_FILE="$WORK_DIR/vlajepa_libero_plus_${RUN_TAG}_eval_results.txt"
touch "$RESULTS_FILE"

ALL_CATEGORIES=("Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States")
if [ "$HAVE_LIBMAGICKWAND" -eq 1 ]; then
    ALL_CATEGORIES+=("Sensor Noise")
fi

run_worker () {
    local worker_id="$1"
    local base_port=$((15500 + worker_id * 20))  # wide spacing so workers' ports never collide
    local index=-1
    for category in "${ALL_CATEGORIES[@]}"; do
        index=$((index+1))
        if [ $((index % NUM_GPUS)) -ne "$worker_id" ]; then
            continue
        fi
        local port=$((base_port + index))
        local cat_tag="${category// /_}"

        if grep -q "^${category}:.*# successes" "$RESULTS_FILE" 2>/dev/null; then
            echo "=== [$(date)] worker $worker_id: skipping already-done category: $category ===" >> "$LOG_DIR/driver_libero_plus_${RUN_TAG}_w${worker_id}.log"
            continue
        fi

        echo "=== [$(date)] worker $worker_id (gpu $worker_id): starting category: $category (port=$port) ===" >> "$LOG_DIR/driver_libero_plus_${RUN_TAG}_w${worker_id}.log"

        cd "$REPO_DIR"
        if [ -n "$CRITIC_PATH" ]; then
            CUDA_VISIBLE_DEVICES="$worker_id" PYTHONPATH="$REPO_DIR" "$SERVER_PY" deployment/model_server/server_qc_policy.py \
                --ckpt_path "$LIBERO_CKPT" --critic_checkpoint_path "$CRITIC_PATH" \
                --num_samples "$NUM_SAMPLES" --num_qs "$NUM_QS" \
                --q_agg "$Q_AGG" --uncertainty_penalty "$UNCERTAINTY_PENALTY" \
                --actor_disagreement_penalty "$ACTOR_DISAGREEMENT_PENALTY" \
                --critic_weight "$CRITIC_WEIGHT" --selection_mode "$SELECTION_MODE" \
                ${MAXIMIZE_SCORE:+--maximize_score} \
                --port "$port" --cuda 0 \
                > "$LOG_DIR/server_libero_plus_${RUN_TAG}_${cat_tag}.log" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="$worker_id" PYTHONPATH="$REPO_DIR" "$SERVER_PY" deployment/model_server/server_policy.py \
                --ckpt_path "$LIBERO_CKPT" --port "$port" --cuda 0 \
                > "$LOG_DIR/server_libero_plus_${RUN_TAG}_${cat_tag}.log" 2>&1 &
        fi
        local server_pid=$!

        local waited=0
        local up=0
        while [ "$waited" -lt 600 ]; do
            if "$SERVER_PY" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('127.0.0.1', $port))
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
                up=1
                break
            fi
            if ! kill -0 "$server_pid" 2>/dev/null; then
                break
            fi
            sleep 3
            waited=$((waited+3))
        done
        if [ "$up" -ne 1 ]; then
            echo "$category: SERVER FAILED TO START (see $LOG_DIR/server_libero_plus_${RUN_TAG}_${cat_tag}.log)" >> "$RESULTS_FILE"
            kill "$server_pid" 2>/dev/null
            wait "$server_pid" 2>/dev/null
            continue
        fi

        mkdir -p "$REPO_DIR/results/libero_plus_${RUN_TAG}_${cat_tag}"
        PYTHONPATH="$REPO_DIR:$LIBERO_PLUS_BENCH_DIR" \
        LIBERO_CONFIG_PATH="$LIBERO_PLUS_CONFIG_DIR" \
        CUDA_VISIBLE_DEVICES="$worker_id" \
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
        local status=$?

        kill "$server_pid" 2>/dev/null
        wait "$server_pid" 2>/dev/null
        sleep 5

        if [ "$status" -ne 0 ]; then
            echo "$category: EVAL FAILED or TIMED OUT (see $LOG_DIR/eval_libero_plus_${RUN_TAG}_${cat_tag}.log)" >> "$RESULTS_FILE"
            continue
        fi

        local success_line
        success_line="$(grep '# successes' "$LOG_DIR/eval_libero_plus_${RUN_TAG}_${cat_tag}.log" | tail -1)"
        echo "$category: $success_line" >> "$RESULTS_FILE"
        echo "=== [$(date)] worker $worker_id: finished category: $category ===" >> "$LOG_DIR/driver_libero_plus_${RUN_TAG}_w${worker_id}.log"
    done
    echo "=== [$(date)] worker $worker_id DONE ===" >> "$LOG_DIR/driver_libero_plus_${RUN_TAG}_w${worker_id}.log"
}

pids=()
for w in $(seq 0 $((NUM_GPUS - 1))); do
    run_worker "$w" &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

echo "=== [$(date)] ALL WORKERS DONE (fail=$fail) ===" | tee -a "$LOG_DIR/driver_libero_plus_${RUN_TAG}.log"
cat "$RESULTS_FILE"
echo "Results: $RESULTS_FILE"
