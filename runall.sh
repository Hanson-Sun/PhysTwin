#!/usr/bin/env bash

# Fail fast: exit on error, undefined var, or pipeline failure
set -euo pipefail

# Print a helpful message when any command fails
trap 'rc=$?; echo "runall.sh failed on line ${LINENO} with exit code ${rc}" >&2; exit ${rc}' ERR

export WANDB_MODE=offline

# Process the data
python script_process_data.py

# need to run calibrate_camera_extrinsics.py after
python script_calibrate_camera_extrinsics.py

# Further get the data for first-frame Gaussian
python export_gaussian_data.py

# Get human mask data for visualization and rendering evaluation
python export_video_human_mask.py

# Zero-order Optimization
python script_optimize.py

# First-order Optimization
python script_train.py

# Inference with the constructed models
python script_inference.py

# Train the Gaussian with the first-frame data
bash gs_run.sh

# Use LBS to render the dynamic videos (The final videos in ./gaussian_output_dynamic folder)
bash gs_run_simulate.sh
python export_render_eval_data.py
# Get the quantative results
bash evaluate.sh

# Get the qualitative results
bash gs_run_simulate_white.sh
python visualize_render_results.py