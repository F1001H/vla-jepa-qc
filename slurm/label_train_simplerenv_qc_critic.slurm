#!/bin/bash
#SBATCH --job-name=label_train_simplerenv_qc
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1                # single-GPU inference-only labeling pass (no FSDP), matches
                                          # the local RTX 5090 run this mirrors -- see qc/label_rewards_simplerenv.py
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=03:00:00                  # local frame-count estimate: ~50k frames total (1000 episodes,
                                          # Bridge+Fractal) vs LIBERO's 273k frames/~4h -> well under 1h
                                          # expected; generous margin for a cold cluster environment
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out

set -euo pipefail

# Runs the SimplerEnv QC critic labeling + training pipeline on the cluster,
# entirely self-contained against the VLA-JEPA fork -- doesn't touch openpi
# at all (openpi's JAX/uv stack is unrelated; this is VLA-JEPA's own
# PyTorch/HuggingFace "starVLA" framework, with its own venv set up below).
#
# Clones directly from our own fork (upstream ginwind/VLA-JEPA + our
# validated qc/ additions + local env-compat patches, already committed and
# pushed there) -- no rsync prerequisite needed. All paths (log output,
# work dir, venv, checkpoints) are absolute, so this can be submitted from
# any directory on a login node -- it doesn't need to be run from inside
# any particular git clone:
#   sbatch /path/to/label_train_simplerenv_qc_critic.slurm
#
# Output: a trained critic checkpoint at $OUT_DIR/vlajepa_critic_simplerenv.pt
# -- rsync it back locally once done:
#   rsync -avz cm74paso@login-01.ai.tu-darmstadt.de:$OUT_DIR/vlajepa_critic_simplerenv.pt ~/starvla_jepa/

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_simplerenv_qc"
REPO_DIR="$WORK_DIR/VLA-JEPA"
CKPT_DIR="$WORK_DIR/checkpoints"
OUT_DIR="$WORK_DIR/output"

export HF_HOME="${HF_HOME:-$SHARED_WS/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$WORK_DIR" "$CKPT_DIR" "$OUT_DIR" "$WORK_DIR/slurm_logs"

# --- 1. Clone our fork (upstream + qc/ additions + env-compat patches already
#    committed there) + venv. Skip clone if already done by a prior attempt,
#    otherwise pull latest in case the fork was updated since. ---
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/F1001H/vla-jepa-qc.git "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"

if [ ! -d ".venv" ]; then
    uv venv --python 3.11 .venv
fi
PY="$REPO_DIR/.venv/bin/python"

uv pip install --python "$PY" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
grep -v -E "^(torchvision)" requirements.txt > /tmp/vlajepa_requirements_filtered_$$.txt
uv pip install --python "$PY" -r /tmp/vlajepa_requirements_filtered_$$.txt
uv pip install --python "$PY" -e .
uv pip install --python "$PY" huggingface_hub[hf_transfer] pyarrow av

# --- 2. Download checkpoints (shared HF_HOME cache -- may already be warm
#    from other jobs on this cluster) ---
"$PY" -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-VL-2B-Instruct \
    --local-dir "$CKPT_DIR/Qwen3-VL-2B-Instruct"
"$PY" -m huggingface_hub.commands.huggingface_cli download facebook/vjepa2-vitl-fpc64-256 \
    --local-dir "$CKPT_DIR/vjepa2-vitl-fpc64-256"
"$PY" -m huggingface_hub.commands.huggingface_cli download ginwind/VLA-JEPA \
    --include "SimplerEnv/*" --local-dir "$CKPT_DIR/VLA-JEPA-SimplerEnv-repo"

SIMPLERENV_CKPT="$CKPT_DIR/VLA-JEPA-SimplerEnv-repo/SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt"

python3 - "$CKPT_DIR" <<PYEOF
import json
ckpt_dir = "$CKPT_DIR"
cfg_path = f"{ckpt_dir}/VLA-JEPA-SimplerEnv-repo/SimplerEnv/config.json"
with open(cfg_path) as f:
    cfg = json.load(f)
cfg["framework"]["qwenvl"]["base_vlm"] = f"{ckpt_dir}/Qwen3-VL-2B-Instruct"
cfg["framework"]["qwenvl"]["attn_implementation"] = "sdpa"
cfg["framework"]["vj2_model"]["base_encoder"] = f"{ckpt_dir}/vjepa2-vitl-fpc64-256"
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=4)
print("Patched SimplerEnv config.json")
PYEOF

# Same patch for config.yaml (used by some code paths instead of config.json)
python3 - "$CKPT_DIR" <<PYEOF
import re
ckpt_dir = "$CKPT_DIR"
cfg_path = f"{ckpt_dir}/VLA-JEPA-SimplerEnv-repo/SimplerEnv/config.yaml"
text = open(cfg_path).read()
text = re.sub(r"base_vlm: .*", f"base_vlm: {ckpt_dir}/Qwen3-VL-2B-Instruct", text, count=1)
text = re.sub(r"attn_implementation: .*", "attn_implementation: sdpa", text, count=1)
text = re.sub(r"base_encoder: .*", f"base_encoder: {ckpt_dir}/vjepa2-vitl-fpc64-256", text, count=1)
open(cfg_path, "w").write(text)
print("Patched SimplerEnv config.yaml")
PYEOF

# --- 3. Label + train critic ---
cd "$REPO_DIR"
PYTHONPATH="$REPO_DIR" "$PY" qc/label_rewards_simplerenv.py \
    --ckpt-path "$SIMPLERENV_CKPT" \
    --output-path "$OUT_DIR/qc_cache_simplerenv.npz" \
    --bridge-episodes 500 --fractal-episodes 500 \
    --checkpoint-every 25 --cuda 0

PYTHONPATH="$REPO_DIR" "$PY" qc/train_critic.py \
    --cache-path "$OUT_DIR/qc_cache_simplerenv.npz" \
    --output-path "$OUT_DIR/vlajepa_critic_simplerenv.pt" \
    --epochs 20 --batch-size 256 --num-qs 5 --cuda 0

echo "Done. Critic checkpoint: $OUT_DIR/vlajepa_critic_simplerenv.pt"
echo "rsync it back locally with:"
echo "  rsync -avz \$(hostname):$OUT_DIR/vlajepa_critic_simplerenv.pt ~/starvla_jepa/"
