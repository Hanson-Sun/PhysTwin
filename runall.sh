#!/usr/bin/env bash

# Exit on:
# - any error
# - undefined variable
# - pipeline failure
set -euo pipefail

# Print helpful error message
trap 'rc=$?; echo "runall.sh failed on line ${LINENO} with exit code ${rc}" >&2; exit ${rc}' ERR

# Kill entire process group on Ctrl-C
# trap 'echo; echo "Interrupted. Killing all child processes..."; kill 0' INT TERM

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Load conda into this non-interactive shell ----
source "$(conda info --base)/etc/profile.d/conda.sh"

# Helper to print step headers
step() {
  echo
  echo "========================================"
  echo "STEP: $*"
  echo "========================================"
}

# =====================================================
# STEP 1: Different environment
# =====================================================

# step "Parse depth data"
# conda activate phystwin_data
# python -u script_depth_inference.py

# # Switch back to main env
# conda activate phystwin

# # =====================================================
# # Main pipeline (sequential)
# # =====================================================

# step "Process the data"
# python -u script_process_data.py

# # step "Detect environment planes"
# python script_detect_environment_planes.py

# # step "Align extrinsics to detected plane (delete .aligned_to_plane to run again)"
# python -u script_align_to_plane.py

# # step "Calibrate camera extrinsics"
# python -u script_calibrate_camera_extrinsics.py 

# step "Export Gaussian data"
python -u export_gaussian_data.py

# step "Export human mask data"
python -u export_video_human_mask.py

# step "Zero-order Optimization"
python -u script_optimize.py

step "First-order Optimization"
python -u script_train.py

step "Inference"
python -u script_inference.py

step "Train Gaussian (first-frame)"
bash gs_run.sh

step "LBS dynamic video rendering"
bash gs_run_simulate.sh

step "Export render eval data"
python -u export_render_eval_data.py

step "Quantitative evaluation"
bash evaluate.sh

step "Qualitative results"
bash gs_run_simulate_white.sh

python -u visualize_render_results.py

echo
echo "========================================"
echo "ALL STEPS COMPLETED SUCCESSFULLY"
echo "========================================"