#!/bin/bash
#SBATCH --job-name=smoketest_sapien_vulkan
#SBATCH --partition=all
#SBATCH --gres=gpu:h100:1
#SBATCH --exclude=cn02,cn17              # known-bad/occupied GPUs on these nodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/workspaces/VLA_Reinforcement/fabian/vlajepa_simplerenv_qc/slurm_logs/%x-%j.out

set -uo pipefail   # NOT -e: we want every check below to run and report, not bail on the first failure

# Cheap feasibility check for whether SimplerEnv's renderer (SAPIEN 2.2.2,
# Vulkan-based offscreen ray-traced rendering) can even run on this
# cluster's compute nodes, BEFORE investing in porting the full SimplerEnv/
# ManiSkill2 eval pipeline there. Datacenter compute nodes often ship
# NVIDIA drivers without a working Vulkan ICD for headless rendering (only
# the compute/CUDA stack), unlike a workstation GPU -- this is the single
# biggest unverified risk for running SimplerEnv eval on this cluster.
#
# Three tiers, cheapest/fastest first -- each one only matters if the
# previous one passed:
#   1. Does an NVIDIA Vulkan ICD file exist at all (no install needed)?
#   2. Does `vulkaninfo` (if present) see a usable device?
#   3. Does an actual minimal SAPIEN offscreen render produce a real frame?
#      (same API calls examples/SimplerEnv relies on, verified against the
#      locally-working SAPIEN 2.2.2 install's API surface before writing
#      this -- add_camera/add_directional_light/add_ground/take_picture/
#      get_float_texture all confirmed present in that version.)
#
# Submit with:
#   sbatch ~/vla-jepa-qc/slurm/smoketest_sapien_vulkan.slurm

SHARED_WS="/mnt/vast/workspaces/VLA_Reinforcement"
WORK_DIR="$SHARED_WS/fabian/vlajepa_simplerenv_qc"
VENV_DIR="$WORK_DIR/sapien_smoketest_venv"

mkdir -p "$WORK_DIR" "$WORK_DIR/slurm_logs"

echo "=== GPU diagnostics ==="
echo "Node: $(hostname)"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}  SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,driver_version --format=csv
echo "========================"

echo "=== Tier 1: NVIDIA Vulkan ICD file ==="
found_icd=0
for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json \
         /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json /etc/vulkan/icd.d/nvidia_icd.x86_64.json; do
    if [ -f "$f" ]; then
        echo "Found: $f"
        cat "$f"
        found_icd=1
    fi
done
if [ "$found_icd" -eq 0 ]; then
    echo "NOT FOUND at either standard path -- searching filesystem (best-effort, may be slow/incomplete on a shared FS)..."
    find /usr /etc -iname "nvidia_icd*.json" 2>/dev/null | head -5
fi

echo "=== Tier 2: vulkaninfo ==="
if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>&1 | head -50
else
    echo "vulkaninfo not installed -- skipping (not fatal, tier 3 is the decisive check)"
fi

echo "=== Tier 3: minimal SAPIEN offscreen render ==="
if [ ! -f "$VENV_DIR/.setup_complete" ]; then
    uv venv --python 3.10 "$VENV_DIR"
    uv pip install --python "$VENV_DIR/bin/python" "setuptools<81" sapien==2.2.2 numpy
    touch "$VENV_DIR/.setup_complete"
fi

export SVULKAN2_CPU_COPY=1
export SVULKAN2_DISABLE_DENOISER=1
export DISPLAY=
# Under SLURM, stdout goes to a file rather than a TTY, so Python defaults
# to full block buffering -- without this, print() output (including the
# "Creating engine..."/"Creating scene..." progress lines below) sits
# invisible in a buffer until the process exits, making a genuine hang
# indistinguishable from normal-but-silent progress. Also wrapped in an
# explicit `timeout` so a real Vulkan-init hang reports clearly as its own
# timeout rather than only being caught by SLURM's job-level time limit.
export PYTHONUNBUFFERED=1

timeout 300 "$VENV_DIR/bin/python" -u - <<'PYEOF'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import sapien.core as sapien

print("Creating engine + offscreen renderer...")
engine = sapien.Engine()
renderer = sapien.SapienRenderer(offscreen_only=True)
engine.set_renderer(renderer)

print("Creating scene...")
scene = engine.create_scene()
scene.set_timestep(1 / 100.0)
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
scene.add_ground(altitude=0)

width, height = 640, 480
camera = scene.add_camera(name="camera", width=width, height=height, fovy=1.0, near=0.1, far=100)
camera.set_pose(sapien.Pose(p=[-2, 0, 1]))

print("Stepping + rendering one frame...")
scene.step()
scene.update_render()
camera.take_picture()
rgba = camera.get_float_texture("Color")

assert rgba is not None, "get_float_texture returned None"
assert rgba.shape == (height, width, 4), f"unexpected shape {rgba.shape}"
assert np.isfinite(rgba).all(), "rendered image has non-finite values"
nonzero_frac = float((rgba != 0).mean())
assert nonzero_frac > 0.01, f"rendered frame is essentially all-zero (nonzero_frac={nonzero_frac:.4f}) -- likely a black/broken render, not a real scene"

print(f"SUCCESS: rendered {rgba.shape} frame, mean={rgba.mean():.4f}, nonzero_frac={nonzero_frac:.4f}")
PYEOF
sapien_status=$?

echo "=== Result ==="
if [ "$sapien_status" -eq 0 ]; then
    echo "SAPIEN/Vulkan rendering WORKS on this node ($(hostname)) -- safe to proceed with full SimplerEnv setup."
elif [ "$sapien_status" -eq 124 ]; then
    echo "SAPIEN/Vulkan rendering HUNG on this node ($(hostname)) -- killed after 300s with no output past the ICD/venv setup. Likely a genuine Vulkan device-enumeration/driver hang, not just slow first-run compilation."
else
    echo "SAPIEN/Vulkan rendering FAILED on this node ($(hostname)) -- see traceback above before investing in the full SimplerEnv port."
fi
exit "$sapien_status"
