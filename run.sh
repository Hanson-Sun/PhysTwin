#!/usr/bin/env bash
# Run the full pipeline for the sim cases.
# Usage: ./run.sh [extra args passed to run_case_pipeline.py]
# e.g.  ./run.sh --skip_process
# Visualization is enabled by default; pass --no_visualize for a headless run.
# Live Open3D preview has been removed from the simulation path.
# All output is also written to logs/run_<timestamp>.log (override with LOG_DIR=...).
# sim_rigid_box sim_rigid_box_grip_lift sim_rigid_box_heavy_end sim_rope sim_soft_ball sim_soft_sloth_grip_lift sim_soft_seal_grip_lift sim_soft_octopus_grip_lift sim_soft_teddy_bear_grip_lift

set -euo pipefail

CASES=(sim_soft_ball_grip_lift)

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# Matches tqdm progress lines (extended regex), any of:
#   1) the stats suffix: "[00:10<00:12,  4.50it/s" / "[1:02:03, 1.5s/it" / "[00:05, ?it/s"
#      - elapsed time, optional "<remaining", then a rate of a number or "?",
#        followed by any unit name in either "<unit>/s" or "s/<unit>" form
#        (it, step, iter, kit, ...)
#   2) the bar prefix: " 45%|████   |"
TQDM_RE='\[[0-9:]+(<[0-9:?]+)?, +(\?|[0-9.]+)[a-zA-Z_]*(/s|s/[a-zA-Z_]+)|[0-9]+%[|]'

# Send everything (stdout + stderr) to the terminal untouched, and to the log
# file with tqdm's carriage-return progress updates split into lines and dropped.
exec > >(tee >(tr '\r' '\n' | grep --line-buffered -Ev "$TQDM_RE" >> "$LOG_FILE")) 2>&1
export PYTHONUNBUFFERED=1  # flush prints immediately so the log stays in order

echo "Logging to $LOG_FILE"

FAILED=()
for case_name in "${CASES[@]}"; do
    echo ">>> Running case: $case_name"
    if ! python scripts/run_case_pipeline.py --case_name "$case_name" "$@"; then
        echo "FAILED: $case_name (continuing...)"
        FAILED+=("$case_name")
    fi
done

echo "All cases complete."
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Failed cases: ${FAILED[*]}"
fi