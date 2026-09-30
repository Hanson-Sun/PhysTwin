#!/usr/bin/env bash
# Run the full pipeline for the sim cases.
# Usage: ./run.sh [extra args passed to run_case_pipeline.py]
# e.g.  ./run.sh --skip_process
# Visualization is enabled by default; pass --no_visualize for a headless run.
# Live Open3D preview has been removed from the simulation path.
# sim_rigid_box sim_rigid_box_grip_lift sim_rigid_box_heavy_end sim_rope sim_soft_ball sim_soft_ball_grip_lift

set -euo pipefail

CASES=(sim_soft_sloth_grip_lift sim_soft_seal_grip_lift sim_soft_octopus_grip_lift sim_soft_teddy_bear_grip_lift)

for case_name in "${CASES[@]}"; do
    echo ">>> Running case: $case_name"
    python scripts/run_case_pipeline.py --case_name "$case_name" "$@" || echo "FAILED: $case_name (continuing...)"
done

echo "All cases complete."