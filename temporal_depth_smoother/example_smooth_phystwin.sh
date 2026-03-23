#!/bin/bash
# Example: Smooth depth data for a PhysTwin case

set -e

CASE_NAME="double_stretch_sloth_smooth"
CHECKPOINT="temporal_depth_output/training/best_model.pt"

python -m temporal_depth_smoother.smooth_phystwin_data \
  "$CASE_NAME" \
  --checkpoint "$CHECKPOINT" \
  --window-size 16 \
  --device cuda \
  --evaluate
