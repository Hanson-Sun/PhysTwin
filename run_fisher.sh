#!/usr/bin/env bash
# Every ASID Fisher component on one case, then a rendered MuJoCo trajectory.
#
# Usage:  ./run_fisher.sh                 # sim_soft_sloth_push, default sizes
#         CASE=sim_soft_ball_push ./run_fisher.sh
#         FIM_GROUPS=30 PLAN_GROUPS=20 ./run_fisher.sh
#
# Stages, in order: exact grouped FIM -> selection -> exact per-action FIM
# validation -> per-point gain heatmaps -> sketch study -> planned push replay
# through the MuJoCo generator. One action per ASID iteration: `plan_sequential`
# (a greedy set of pushes) is deliberately not chained here.
#
# Sizes are chosen to finish in roughly 1.5-2 h on one GPU at the default 60
# frames. Everything is overridable from the environment; see the variables
# below. As with run.sh, the terminal gets the output untouched and
# logs/fisher_<case>_<timestamp>.log gets the same text with tqdm's
# carriage-return progress updates flattened and dropped.
set -euo pipefail

cd "$(dirname "$0")"

CASE=${CASE:-sim_soft_sloth_push}
BASE=${BASE:-data/different_types}
ROOT=${ROOT:-experiments-fim}
# Exact FIM tolerates more groups than the rank-12 sketch used for planning.
FIM_GROUPS=${FIM_GROUPS:-20}
PLAN_GROUPS=${PLAN_GROUPS:-10}
FRAMES=${FRAMES:-60}
NODES=${NODES:-500}
RANK=${RANK:-12}
AZIMUTH=${AZIMUTH:-6}
TILTS=${TILTS:-"0 -20"}
# One ASID iteration commits to a single best trajectory, so the default
# shortlist is one action; raise SELECT only for an offline set study.
SELECT=${SELECT:-1}
PLAN_FRAMES=${PLAN_FRAMES:-60}
OFFSET_LIMIT=${OFFSET_LIMIT:-0.05}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/fisher_${CASE}_${STAMP}.log"

# Matches tqdm progress lines (extended regex), any of:
#   1) the stats suffix: "[00:10<00:12,  4.50it/s" / "[1:02:03, 1.5s/it" / "[00:05, ?it/s"
#   2) the bar prefix: " 45%|████   |"
# Kept identical to run.sh so both logs are readable the same way.
TQDM_RE='\[[0-9:]+(<[0-9:?]+)?, +(\?|[0-9.]+)[a-zA-Z_]*(/s|s/[a-zA-Z_]+)|[0-9]+%[|]'

exec > >(tee >(tr '\r' '\n' | grep --line-buffered -Ev "$TQDM_RE" >> "$LOG")) 2>&1
export PYTHONUNBUFFERED=1  # flush prints immediately so the log stays in order

if [[ "${CONDA_DEFAULT_ENV:-}" == "phystwin-cu132" ]]; then
    PY=(python)
else
    # Non-interactive shells often skip the conda hook; fall back to the usual
    # install location before giving up.
    if ! command -v conda >/dev/null; then
        for candidate in "$HOME/miniconda3/bin" "$HOME/anaconda3/bin" /opt/conda/bin; do
            if [[ -x "$candidate/conda" ]]; then
                PATH="$candidate:$PATH"
                break
            fi
        done
    fi
    command -v conda >/dev/null || { echo "conda not found; activate phystwin-cu132" >&2; exit 1; }
    PY=(conda run --no-capture-output -n phystwin-cu132 env PYTHONPATH=.)
fi
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

FIM_RUN="$ROOT/$CASE/fim_g${FIM_GROUPS}_f${FRAMES}"
PLAN_RUN="$ROOT/$CASE/mix_g${PLAN_GROUPS}_r${RANK}"
TRAJECTORY="$ROOT/$CASE/trajectory_${STAMP}.json"
CAPTURE="$ROOT/$CASE/plan_${STAMP}"

stage() {
    local name=$1
    shift
    local start=$SECONDS
    echo
    echo ">>> $name   [$(date +%H:%M:%S)]"
    "$@"
    echo "<<< $name ok in $((SECONDS - start))s"
}

echo "Logging to $LOG"
echo "case $CASE"
echo "fim $FIM_RUN"
echo "plan $PLAN_RUN"
echo

# 1. Exact grouped Fisher at the nominal action, plus its figures and the
#    per-node local gain map. ~7 min at g=20, f=60.
stage "exact FIM" "${PY[@]}" scripts/run_fim_case.py \
    --case "$BASE/$CASE" --root "$ROOT" --nodes "$NODES" --frames "$FRAMES" \
    --groups "$FIM_GROUPS" --run-id "fim_g${FIM_GROUPS}_f${FRAMES}"

stage "FIM overlay (reload path)" "${PY[@]}" scripts/plot_fim.py "$FIM_RUN"

# 2. Candidate pushes scored by a rank-R sketch; the argmax is the ASID
#    iteration's next trajectory.
stage "selection" "${PY[@]}" scripts/plan_actions.py \
    --case "$BASE/$CASE" --root "$ROOT" --nodes "$NODES" --frames "$FRAMES" \
    --groups "$PLAN_GROUPS" --rank "$RANK" --azimuth-count "$AZIMUTH" \
    --tilts $TILTS --select "$SELECT" --run-id "mix_g${PLAN_GROUPS}_r${RANK}"

# 3. Exact Fisher for every candidate: how good the sketch ranking really was.
stage "exact validation" "${PY[@]}" scripts/validate_actions.py \
    "$PLAN_RUN" --select "$SELECT"

# 4. Where each action puts information, per body point.
stage "gain heatmaps" "${PY[@]}" scripts/plot_fim_heatmaps.py "$PLAN_RUN"
stage "gain heatmaps (relative)" "${PY[@]}" scripts/plot_fim_heatmaps.py \
    "$PLAN_RUN" --relative

# 5. Sketch study: capture against the exact gains saved in stage 3. Needs the
#    two canonical run ids it was written against, so it is skipped otherwise.
if [[ -d "$ROOT/$CASE/mix_g10_r12/actions" && -d "$ROOT/$CASE/mix_g20_r12/actions" ]]; then
    stage "sketch study" "${PY[@]}" scripts/analyze_sketch.py \
        --root "$ROOT" --case "$CASE" --rank "$RANK" --select "$SELECT"
else
    echo
    echo ">>> sketch study skipped: needs mix_g10_r12 and mix_g20_r12 exact runs"
fi

# 6. Screen the push family, plan one spline tool motion with CEM, then replay
#    it through the normal data-generation path, so the capture has the same
#    model, cameras, 848x480 resolution and fps as the dataset. Open-loop: the
#    plan comes from the reduced Warp model, not a MuJoCo contact response.
stage "plan trajectory" "${PY[@]}" scripts/plan_push_trajectory.py \
    --case "$BASE/$CASE" --frames "$PLAN_FRAMES" --nodes "$NODES" \
    --azimuth-count "$AZIMUTH" --tilts $TILTS --knots 3 --samples 8 --iterations 3 \
    --offset-limit "$OFFSET_LIMIT" --out "$TRAJECTORY"

stage "generate planned capture" "${PY[@]}" scripts/generate_sim_data.py models.json \
    --case "$CASE" --trajectory "$TRAJECTORY" \
    --output_dir "$ROOT/$CASE" --output_name "$(basename "$CAPTURE")"

echo
echo "Figures:"
find "$FIM_RUN" "$PLAN_RUN" -path '*/figs/*' -name '*.png' | sort
echo "Videos:"
ls "$CAPTURE"/color/*.mp4 2>/dev/null || echo "  (none)"
echo "Planned trajectory:"
echo "  $TRAJECTORY"
echo "Log:"
echo "  $LOG"
