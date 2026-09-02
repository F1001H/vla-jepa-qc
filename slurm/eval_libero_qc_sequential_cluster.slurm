#!/bin/bash
#SBATCH --job-name=eval_libero_qc_sequential
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1                # single GPU shared by the policy server + MuJoCo/EGL rendering,
                                          # matching the local eval scripts' convention
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=24:00:00                  # generous -- standard LIBERO's 4 suites x 50 trials/task is much
                                          # smaller than LIBERO-Plus's 10,026 tasks, but budget safely
                                          # (see eval_vlajepa_libero_plus_sequential.sh's timeout history)
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail

# Standard LIBERO (4 suites: libero_spatial, libero_object, libero_goal,
# libero_10) sequential eval on the cluster -- first time LIBERO's SIM
# (not just the VLA-JEPA policy server) has been ported here. Confirmed
# feasible via slurm/smoketest_mujoco_egl.slurm (MuJoCo/EGL headless
# rendering works on these H100 nodes, unlike SAPIEN/Vulkan for
# SimplerEnv -- see smoketest_sapien_vulkan.slurm's history).
#
# This job is self-contained: builds the LIBERO sim Python 3.8 venv
# (separate from the VLA-JEPA .venv, mirroring examples/libero/.venv in
# the openpi repo) and clones the LIBERO benchmark package if missing,
# entirely under the shared workspace (NOT $HOME -- unlike the VLA-JEPA
# .venv, which had to be built on a login node because compute nodes
# mount $HOME restrictively, $SHARED_WS doesn't have that problem since
# every other job here already writes outputs there fine).
#
# PREREQUISITE: the LIBERO checkpoint + dataset must already exist (from
# slurm/label_train_libero_qc_critic_8gpu.slurm, which downloads both).
#
# Set CRITIC_PATH to run the QC (critic-scored best-of-N) variant instead
# of plain BC -- same convention as eval_vlajepa_libero_plus_sequential.sh.
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/eval_libero_qc_sequential_cluster.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/eval_libero_qc_sequential_cluster.slurm
#
# Override for a QC run, e.g.:
#   CRITIC_PATH=$OUT_DIR/vlajepa_critic_libero_intrinic.pt RUN_TAG=intrinsic \
#   MAXIMIZE_SCORE=true UNCERTAINTY_PENALTY=1.0 ACTOR_DISAGREEMENT_PENALTY=1.0 \
#   sbatch ~/vla-jepa-qc/slurm/eval_libero_qc_sequential_cluster.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_libero_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
CKPT_DIR="$WORK_DIR/checkpoints"
DATASET_DIR="$WORK_DIR/datasets/physical-intelligence-libero"
OUT_DIR="$WORK_DIR/output"
LOG_DIR="$WORK_DIR/logs"

LIBERO_CKPT="$CKPT_DIR/VLA-JEPA-LIBERO-repo/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
SIM_VENV_DIR="$WORK_DIR/libero_sim_venv"
LIBERO_BENCH_DIR="$WORK_DIR/libero_sim/LIBERO"
LIBERO_CONFIG_DIR="$WORK_DIR/libero_sim/.libero_config"

CRITIC_PATH="${CRITIC_PATH:-}"   # empty = plain predict_action (server_policy.py), set to use QC (server_qc_policy.py)
NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
NUM_QS="${NUM_QS:-5}"
Q_AGG="${Q_AGG:-mean}"
UNCERTAINTY_PENALTY="${UNCERTAINTY_PENALTY:-0.0}"
ACTOR_DISAGREEMENT_PENALTY="${ACTOR_DISAGREEMENT_PENALTY:-0.0}"
CRITIC_WEIGHT="${CRITIC_WEIGHT:-1.0}"
MAXIMIZE_SCORE="${MAXIMIZE_SCORE:-}"
SELECTION_MODE="${SELECTION_MODE:-score}"
CATEGORY_TIMEOUT_S="${CATEGORY_TIMEOUT_S:-21600}"   # per-suite; see eval_vlajepa_libero_plus_sequential.sh's
                                                     # timeout history if a suite ever needs longer
RUN_TAG="${RUN_TAG:-}"
TAG_SUFFIX="${RUN_TAG:+_$RUN_TAG}"

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p "$WORK_DIR" "$CKPT_DIR" "$OUT_DIR" "$LOG_DIR" "$WORK_DIR/libero_sim" "$LIBERO_CONFIG_DIR" "$WORK_DIR/slurm_logs"

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

if [ ! -f "$LIBERO_CKPT" ] || [ ! -d "$DATASET_DIR/data" ]; then
    echo "ERROR: LIBERO checkpoint/dataset not found under $WORK_DIR -- run label_train_libero_qc_critic_8gpu.slurm first (it downloads both)." >&2
    exit 1
fi

# --- LIBERO benchmark clone (code + BDDL files + assets, not the demo
# datasets -- we use our own physical-intelligence/libero for that) ---
if [ ! -d "$LIBERO_BENCH_DIR" ]; then
    echo "=== Cloning LIBERO benchmark ==="
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_BENCH_DIR"
fi

# --- LIBERO sim venv (Python 3.8, separate from the VLA-JEPA .venv --
# mirrors examples/libero/.venv in the openpi repo). Lives under
# $SHARED_WS (not $HOME), so -- unlike the VLA-JEPA .venv -- building it
# from inside this compute-node job is fine. ---
if [ ! -f "$SIM_VENV_DIR/.setup_complete" ]; then
    echo "=== Building LIBERO sim venv ==="
    uv venv --python 3.8 "$SIM_VENV_DIR"
    uv pip install --python "$SIM_VENV_DIR/bin/python" -r "$REPO_DIR/libero_sim/requirements.txt"
    uv pip install --python "$SIM_VENV_DIR/bin/python" -e "$LIBERO_BENCH_DIR"
    touch "$SIM_VENV_DIR/.setup_complete"
fi
SIM_PY="$SIM_VENV_DIR/bin/python"

# --- LIBERO config.yaml -- auto-derived from get_default_path_dict(),
# same mechanism that must have produced the local one (paths relative to
# wherever the LIBERO package itself lives, see
# third_party/libero/libero/libero/__init__.py locally). ---
if [ ! -f "$LIBERO_CONFIG_DIR/config.yaml" ]; then
    echo "=== Generating LIBERO config.yaml ==="
    "$SIM_PY" -c "
import sys, yaml
sys.path.insert(0, '$LIBERO_BENCH_DIR')
from libero.libero import get_default_path_dict
paths = get_default_path_dict('$LIBERO_BENCH_DIR/libero/libero')
with open('$LIBERO_CONFIG_DIR/config.yaml', 'w') as f:
    yaml.dump(paths, f)
print(paths)
"
fi

# --- Sequential 4-suite eval ---
RESULTS_FILE="$WORK_DIR/vlajepa_libero_sequential${TAG_SUFFIX}_eval_results.txt"
touch "$RESULTS_FILE"   # NOT truncated -- resume support below skips suites already recorded here

SUITES=(libero_spatial libero_object libero_goal libero_10)
base_port=15183

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
for suite in "${SUITES[@]}"; do
    index=$((index+1))
    port=$((base_port+index))

    if grep -q "^${suite}:.*# successes" "$RESULTS_FILE" 2>/dev/null; then
        echo "=== [$(date)] Skipping already-done suite: $suite ===" | tee -a "$LOG_DIR/driver_libero${TAG_SUFFIX}.log"
        continue
    fi

    echo "=== [$(date)] Starting suite: $suite (port=$port) ===" | tee -a "$LOG_DIR/driver_libero${TAG_SUFFIX}.log"

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
            > "$LOG_DIR/server_libero${TAG_SUFFIX}_${suite}.log" 2>&1 &
    else
        PYTHONPATH="$REPO_DIR" "$SERVER_PY" deployment/model_server/server_policy.py \
            --ckpt_path "$LIBERO_CKPT" --port "$port" --cuda 0 \
            > "$LOG_DIR/server_libero${TAG_SUFFIX}_${suite}.log" 2>&1 &
    fi
    SERVER_PID=$!

    if ! wait_for_port "$port"; then
        echo "$suite: SERVER FAILED TO START (see $LOG_DIR/server_libero${TAG_SUFFIX}_${suite}.log)" | tee -a "$RESULTS_FILE"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        continue
    fi

    mkdir -p "$REPO_DIR/results/libero${TAG_SUFFIX}_${suite}"
    PYTHONPATH="$REPO_DIR:$LIBERO_BENCH_DIR" \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR" \
    timeout "$CATEGORY_TIMEOUT_S" "$SIM_PY" examples/LIBERO/eval_libero.py \
        --args.pretrained-path "$LIBERO_CKPT" \
        --args.host 127.0.0.1 \
        --args.port "$port" \
        --args.task-suite-name "$suite" \
        --args.num-trials-per-task "$NUM_TRIALS" \
        --args.video-out-path "results/libero${TAG_SUFFIX}_${suite}" \
        --args.with-state true \
        > "$LOG_DIR/eval_libero${TAG_SUFFIX}_${suite}.log" 2>&1
    status=$?

    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    sleep 5

    if [ "$status" -ne 0 ]; then
        echo "$suite: EVAL FAILED or TIMED OUT (see $LOG_DIR/eval_libero${TAG_SUFFIX}_${suite}.log)" | tee -a "$RESULTS_FILE"
        continue
    fi

    # NOT `grep 'Total success rate'` -- rich's console formatting wraps that
    # line's actual number onto the next terminal line (see
    # eval_vlajepa_libero_plus_sequential.sh's history) -- the last
    # "# successes: N (XX.X%)" line has the same final number, unwrapped.
    success_line="$(grep '# successes' "$LOG_DIR/eval_libero${TAG_SUFFIX}_${suite}.log" | tail -1)"
    echo "$suite: $success_line" | tee -a "$RESULTS_FILE"
    echo "=== [$(date)] Finished suite: $suite ===" | tee -a "$LOG_DIR/driver_libero${TAG_SUFFIX}.log"
done

echo "=== [$(date)] ALL LIBERO SUITES DONE ===" | tee -a "$LOG_DIR/driver_libero${TAG_SUFFIX}.log"
echo "Results: $RESULTS_FILE"
