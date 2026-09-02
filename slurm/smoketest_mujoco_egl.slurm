#!/bin/bash
#SBATCH --job-name=smoketest_mujoco_egl
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_libero_qc/slurm_logs/%x-%j.out

set -uo pipefail   # NOT -e: want every check to run and report, not bail on the first failure

# Cheap feasibility check for whether LIBERO's renderer (MuJoCo via
# robosuite, headless EGL) can run on this cluster's compute nodes, before
# porting the full LIBERO sim eval stack (examples/libero/.venv in the
# OPENPI repo, separate from this VLA-JEPA fork -- robosuite 1.4.1 +
# mujoco 3.2.3, Python 3.8, not touched on this cluster yet) here.
#
# Better odds than SAPIEN/Vulkan (see smoketest_sapien_vulkan.slurm, which
# hung on 2 different nodes) -- EGL headless OpenGL is a much more
# standard, widely-supported capability on datacenter NVIDIA drivers than
# real-time ray-traced Vulkan rendering. But "usually works on clusters"
# isn't "confirmed on hessianAI 43", hence this check.
#
# Tests via plain `mujoco` (not robosuite) with MUJOCO_GL=egl -- this is
# faithful to the real thing: robosuite's own EGL context creation
# (renderers/context/egl_context.py) is built directly on
# mujoco.egl.egl_ext's device-query/display-init path, so if raw mujoco's
# EGL renderer works, robosuite's will too. Much lighter dependency (just
# `mujoco`, no robosuite/LIBERO/BDDL install needed for this check).
#
# Three tiers, cheapest first:
#   1. Does the NVIDIA EGL vendor config exist at all (no install needed)?
#   2. Does a minimal mujoco.Renderer() + render() call produce a real,
#      non-blank frame?
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/smoketest_mujoco_egl.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_libero_qc"
VENV_DIR="$WORK_DIR/mujoco_smoketest_venv"

mkdir -p "$WORK_DIR" "$WORK_DIR/slurm_logs"

echo "=== GPU diagnostics ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,driver_version --format=csv
echo "========================"

echo "=== Tier 1: NVIDIA EGL vendor config ==="
found=0
for f in /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json; do
    if [ -f "$f" ]; then
        echo "Found: $f"
        cat "$f"
        found=1
    fi
done
if [ "$found" -eq 0 ]; then
    echo "NOT FOUND at standard paths -- searching filesystem (best-effort)..."
    find /usr /etc -iname "*nvidia*.json" -path "*egl_vendor.d*" 2>/dev/null | head -5
fi

echo "=== Tier 2: minimal MuJoCo headless EGL render ==="
if [ ! -f "$VENV_DIR/.setup_complete" ]; then
    uv venv --python 3.11 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" mujoco==3.2.3 numpy
    touch "$VENV_DIR/.setup_complete"
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1   # SLURM stdout is a file, not a TTY -- avoid buffered-output artifacts (see smoketest_sapien_vulkan.slurm's history)

timeout 120 "$VENV_DIR/bin/python" -u - <<'PYEOF'
import numpy as np
import mujoco

print("Creating minimal scene...")
model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("Creating EGL renderer...")
renderer = mujoco.Renderer(model, height=480, width=640)

print("Rendering one frame...")
renderer.update_scene(data)
pixels = renderer.render()

assert pixels.shape == (480, 640, 3), f"unexpected shape {pixels.shape}"
assert np.isfinite(pixels).all(), "rendered image has non-finite values"
assert pixels.std() > 1.0, f"rendered frame has almost no variation (std={pixels.std():.4f}) -- likely a black/broken render"

print(f"SUCCESS: rendered {pixels.shape} frame, mean={pixels.mean():.2f}, std={pixels.std():.2f}")
PYEOF
mujoco_status=$?

echo "=== Result ==="
if [ "$mujoco_status" -eq 0 ]; then
    echo "MuJoCo/EGL rendering WORKS on this node ($(hostname)) -- reasonable to proceed with porting the full LIBERO sim eval stack."
elif [ "$mujoco_status" -eq 124 ]; then
    echo "MuJoCo/EGL rendering HUNG on this node ($(hostname)) -- killed after 120s with no completion. Same failure mode as SAPIEN/Vulkan on this cluster would be a bad sign for headless rendering generally."
else
    echo "MuJoCo/EGL rendering FAILED on this node ($(hostname)) -- see traceback above before investing in the full LIBERO eval port."
fi
exit "$mujoco_status"
