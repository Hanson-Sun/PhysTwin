#!/usr/bin/env bash
# Run the full pipeline for the sim cases.
# Usage: ./run.sh [extra args passed to run_case_pipeline.py]
# e.g.  ./run.sh --skip_process --visualize
# --visualize enables preprocessing previews and saved Warp training videos
# for every case. Live Open3D preview has been removed from the simulation path.

set -euo pipefail

CASES=(sim_rope sim_rigid_box sim_rigid_box_heavy_end)

for case_name in "${CASES[@]}"; do
    echo ">>> Running case: $case_name"
    python scripts/run_case_pipeline.py --case_name "$case_name" "$@"
done

echo "All cases complete."
