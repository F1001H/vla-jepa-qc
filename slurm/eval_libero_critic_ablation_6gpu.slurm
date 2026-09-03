#!/bin/bash
#SBATCH --job-name=eval_libero_critic_ablation
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:6                # 6 critic variants, 1 GPU each, all in parallel
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=84               # 14/GPU x 6, matching the 1-GPU jobs' ratio
#SBATCH --mem=384G
#SBATCH --time=08:00:00                  # libero_10 alone: ~16min/variant locally (10 tasks x 20
                                          # trials) -- generous margin for a cold cluster environment
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail   # NOT -e: need to check each of the 6 parallel evals' own exit status

# Evaluates the 6 critic-training ablation variants (discount x q_agg,
# trained locally by run_critic_ablation_sweep.sh on the SAME fixed-reward
# cache as the main libero_10-comparison critic) against libero_10 --
# cheapest and most relevant suite for this specific ablation question,
# not the full 4-suite/LIBERO-Plus sweep. Runs all 6 in parallel, one GPU
# each, same pattern as label_train_libero_qc_critic_8gpu.slurm's sharding
# but each "shard" here is a full server+eval loop against a different
# critic checkpoint instead of a labeling shard.
#
# PREREQUISITE (these 6 checkpoints only exist locally, no cluster job
# produces them): rsync them over first --
#   rsync -avz ~/starvla_jepa/critic_ablation/ \
#       cm74paso@login-01.ai.tu-darmstadt.de:/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/output/critic_ablation/
#
# Also needs the LIBERO checkpoint + shared sim venv already set up (from
# label_train_libero_qc_critic_8gpu.slurm / eval_libero_qc_sequential_cluster.slurm
# / smoketest_libero_qc_pipeline.slurm -- any one of these builds it).
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/eval_libero_critic_ablation_6gpu.slurm
# or, if cloned elsewhere:
#   REPO_DIR=/path/to/vla-jepa-qc sbatch /path/to/vla-jepa-qc/slurm/eval_libero_critic_ablation_6gpu.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_libero_qc"
REPO_DIR="${REPO_DIR:-$HOME/vla-jepa-qc}"
CKPT_DIR="$WORK_DIR/checkpoints"
OUT_DIR="$WORK_DIR/output"
LOG_DIR="$WORK_DIR/logs"
CRITIC_DIR="$OUT_DIR/critic_ablation"

LIBERO_CKPT="$CKPT_DIR/VLA-JEPA-LIBERO-repo/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt"
SIM_VENV_DIR="$WORK_DIR/libero_sim_venv"
LIBERO_BENCH_DIR="$WORK_DIR/libero_sim/LIBERO"
LIBERO_CONFIG_DIR="$WORK_DIR/libero_sim/.libero_config"
NUM_TRIALS="${NUM_TRIALS:-20}"

# Best ablation setting (fixed for all 6 -- only the CRITIC differs between
# variants; these scoring-time knobs stay constant so the comparison is
# apples-to-apples).
NUM_SAMPLES=8
NUM_QS=5
Q_AGG=mean
UNCERTAINTY_PENALTY=1.0
ACTOR_DISAGREEMENT_PENALTY=1.0
CRITIC_WEIGHT=1.0

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
# Catch-all for stray ${oc.env:WORKSPACE} interpolations left in the
# checkpoint config.json/config.yaml (see label_train_libero_qc_critic_8gpu.slurm's
# header comment) -- these fields aren't read at runtime, they just need
# to resolve without a KeyError.
export WORKSPACE="$WORK_DIR"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

mkdir -p "$WORK_DIR" "$OUT_DIR" "$LOG_DIR" "$WORK_DIR/slurm_logs"

echo "=== GPU diagnostics ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv
echo "========================"

cd "$REPO_DIR"
if [ ! -f ".venv/.setup_complete" ]; then
    echo "ERROR: $REPO_DIR/.venv is missing or incomplete. Build it from a login node first -- see label_train_simplerenv_qc_critic.slurm's header comment." >&2
    exit 1
fi
SERVER_PY="$REPO_DIR/.venv/bin/python"

if [ ! -f "$LIBERO_CKPT" ]; then
    echo "ERROR: LIBERO checkpoint not found under $WORK_DIR -- run label_train_libero_qc_critic_8gpu.slurm first." >&2
    exit 1
fi
if [ ! -f "$SIM_VENV_DIR/.setup_complete" ]; then
    echo "ERROR: $SIM_VENV_DIR is missing -- run eval_libero_qc_sequential_cluster.slurm or smoketest_libero_qc_pipeline.slurm first." >&2
    exit 1
fi
SIM_PY="$SIM_VENV_DIR/bin/python"

VARIANTS=(disc0.99_qaggmean disc0.99_qaggmin disc0.95_qaggmean disc0.95_qaggmin disc0.9_qaggmean disc0.9_qaggmin)

missing=0
for variant in "${VARIANTS[@]}"; do
    if [ ! -f "$CRITIC_DIR/vlajepa_critic_${variant}.pt" ]; then
        echo "ERROR: missing $CRITIC_DIR/vlajepa_critic_${variant}.pt -- rsync the critic_ablation/ dir from local first, see this script's header comment." >&2
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    exit 1
fi

RESULTS_FILE="$WORK_DIR/vlajepa_libero10_critic_ablation_eval_results.txt"
: > "$RESULTS_FILE"

wait_for_port () {
    local port="$1" server_pid="$2"
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
        if ! kill -0 "$server_pid" 2>/dev/null; then
            return 1
        fi
        sleep 3
        waited=$((waited+3))
        if [ "$waited" -ge "$timeout_s" ]; then
            return 1
        fi
    done
    return 0
}

pids=()
for i in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$i]}"
    gpu="$i"
    port=$((15400 + i))
    (
        cd "$REPO_DIR"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_DIR" "$SERVER_PY" deployment/model_server/server_qc_policy.py \
            --ckpt_path "$LIBERO_CKPT" --critic_checkpoint_path "$CRITIC_DIR/vlajepa_critic_${variant}.pt" \
            --num_samples "$NUM_SAMPLES" --num_qs "$NUM_QS" --q_agg "$Q_AGG" \
            --uncertainty_penalty "$UNCERTAINTY_PENALTY" --actor_disagreement_penalty "$ACTOR_DISAGREEMENT_PENALTY" \
            --critic_weight "$CRITIC_WEIGHT" --selection_mode score --maximize_score \
            --port "$port" --cuda 0 \
            > "$LOG_DIR/server_critic_ablation_${variant}.log" 2>&1 &
        server_pid=$!

        if ! wait_for_port "$port" "$server_pid"; then
            echo "$variant: SERVER FAILED TO START (see $LOG_DIR/server_critic_ablation_${variant}.log)" >> "$RESULTS_FILE"
            kill "$server_pid" 2>/dev/null
            wait "$server_pid" 2>/dev/null
            exit 1
        fi

        mkdir -p "$REPO_DIR/results/critic_ablation_${variant}_libero_10"
        PYTHONPATH="$REPO_DIR:$LIBERO_BENCH_DIR" \
        LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR" \
        CUDA_VISIBLE_DEVICES="$gpu" \
        timeout 21600 "$SIM_PY" examples/LIBERO/eval_libero.py \
            --args.pretrained-path "$LIBERO_CKPT" \
            --args.host 127.0.0.1 \
            --args.port "$port" \
            --args.task-suite-name libero_10 \
            --args.num-trials-per-task "$NUM_TRIALS" \
            --args.video-out-path "results/critic_ablation_${variant}_libero_10" \
            --args.with-state true \
            > "$LOG_DIR/eval_critic_ablation_${variant}.log" 2>&1
        status=$?

        kill "$server_pid" 2>/dev/null
        wait "$server_pid" 2>/dev/null

        if [ "$status" -ne 0 ]; then
            echo "$variant: EVAL FAILED or TIMED OUT (see $LOG_DIR/eval_critic_ablation_${variant}.log)" >> "$RESULTS_FILE"
            exit 1
        fi

        # NOT `grep 'Total success rate'` -- rich's console formatting wraps that
        # line's actual number onto the next terminal line -- the last
        # "# successes: N (XX.X%)" line has the same final number, unwrapped.
        success_line="$(grep '# successes' "$LOG_DIR/eval_critic_ablation_${variant}.log" | tail -1)"
        echo "$variant: $success_line" >> "$RESULTS_FILE"
    ) &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

echo "=== Results ==="
cat "$RESULTS_FILE"
if [ "$fail" -ne 0 ]; then
    echo "WARNING: one or more variants failed -- see $RESULTS_FILE and per-variant logs in $LOG_DIR" >&2
fi
echo "Results: $RESULTS_FILE"
