# Full run: `sim_soft_sloth_push`

Every command below starts from the repository root in an interactive WSL shell.
These are multi-minute-to-hour jobs, so they live here rather than in a bounded
test. Run the stages in order; each one only needs the artifacts of the previous.

```bash
cd /root/asid_digital_twin
conda activate phystwin-cu132
export PYTHONPATH=.
export WANDB_MODE=disabled          # trainer calls wandb.init; disable it for batch runs
mkdir -p logs

CASE=sim_soft_sloth_push
BASE=data/different_types
TAG=$(date +%Y%m%d_%H%M)
```

MuJoCo rendering uses the default GLFW path in this environment. Do **not** set
`MUJOCO_GL=egl`; the installed `OpenGL` bindings raise
`AttributeError: module 'OpenGL.EGL' has no attribute 'EGLDeviceEXT'`.

The case already exists on disk (124 frames, 4779 tracked object points,
`final_data.pkl`, `experiments_optimization/sim_soft_sloth_push/optimal_params.pkl`),
so stages 2–6 are runnable now. Stage 1 is only for regenerating the raw capture.

## 0. Sanity checks (seconds)

```bash
python -m unittest -v tests.test_sim_export tests.test_planning_mujoco tests.test_continue_warp
```

## 1. Regenerate the raw MuJoCo capture (optional, destructive)

`export_case` clears and rewrites `color/`, `depth/`, `calibrate.pkl`,
`metadata.json` and `split.json` for this case name. Skipping this stage keeps
the existing capture, which is the same manifest entry (`models.json`,
`push`, `young: 350`, `cellcount: 4 6 1`, `capture_every: 8`).

```bash
python scripts/generate_sim_data.py models.json \
  --output_dir $BASE --case $CASE --overwrite 2>&1 | tee logs/gen_$TAG.log
```

Regeneration invalidates everything downstream: rerun stage 2 without
`--skip_process`, and expect new `checkpoint "resume_geometry" ...` values, so
prior checkpoints can no longer be resumed.

## 2. Preprocessing, CMA initialization, and Warp training

Full path from raw RGB-D (segmentation, tracking, lift-to-3D, CMA, training):

```bash
python scripts/run_case_pipeline.py --base_path $BASE --case_name $CASE \
  --no_visualize --skip_gaussians --cma_max_iter 20 --warp_iterations 20 \
  2>&1 | tee logs/pipeline_$TAG.log
```

Category (`sloth stuffed animal`), the shape prior and the `interior` generator
come from `data_config.csv`, so `--category` is not needed here.

Two shortcuts, depending on what you are re-running:

```bash
# keep the existing tracked final_data.pkl, redo CMA + training
python scripts/run_case_pipeline.py --base_path $BASE --case_name $CASE \
  --no_visualize --skip_gaussians --skip_process \
  --cma_max_iter 20 --warp_iterations 20

# keep the existing CMA optimum, training only
python scripts/train_warp_case.py --base_path $BASE --case_name $CASE \
  --skip_cma --iterations 20 --no_visualize
```

`--warp_iterations` matters: without it `configs/soft.yaml` supplies
`iterations: 200`, which is many hours on the full 4779-point, 86-frame model.
20 epochs is a reasonable first pass; raise it once the loss is stable. Outputs
land in `experiments/$CASE/train/` (`best_*.pth`, `iter_*.pth`,
`training_manifest.json`).

## 3. Exact grouped Fisher on the reduced model (7–17 min)

```bash
python scripts/run_fim_case.py --case $BASE/$CASE \
  --nodes 500 --frames 60 --groups 20 --run-id fim_g20_f60 \
  2>&1 | tee logs/fim_g20_$TAG.log
```

Writes `experiments-fim/$CASE/fim_g20_f60/`: `fisher.npy`, `fim_diag.npy`,
`point_information.npy`, `node_groups.npy`, `phi0.npy`, `summary.json` and
`figs/{fim_matrix,point_information,fim_partition}.png`. Measured cost is
5.15 s per 60-frame/500-node rollout, so the run is `2P + 1` rollouts: about
4 min at `--groups 10` (`P = 21`), 7 min at 20 (`P = 41`), 17 min at 50
(`P = 101`). Redraw the overlay at any time with:

```bash
python scripts/plot_fim.py experiments-fim/$CASE/fim_g20_f60
```

## 4. Candidate selection

Sketch screen over the azimuth/tilt push family (about 2.1 min per action).
`--select 1` is the ASID iteration: one action per round. A larger `--select`
only produces an offline set study, which is not part of the loop:

```bash
python scripts/plan_actions.py --case $BASE/$CASE \
  --nodes 500 --frames 60 --groups 20 --rank 12 \
  --azimuth-count 6 --tilts 0 --select 1 --run-id plan_g20_$TAG \
  2>&1 | tee logs/plan_actions_$TAG.log
```

Exact re-check of those candidates (24 min at `g = 10`, 54 min at `g = 20`):

```bash
python scripts/validate_actions.py experiments-fim/$CASE/plan_g20_$TAG
```

`--exact-top N` is what decides the push: the sketch only proposes a shortlist,
because at 124 frames and `g = 20` its ordering disagrees with the exact Fisher
(it ranked az270 first, the exact check ranked az090 first by 64 nats). Each
exact Fisher costs `2P` rollouts (~18 min at `g = 20`, 124 frames) and is cached
under `<root>/<case>/.exact_cache`, so raising `N` is a one-off cost.

`plan_sequential.py`/`plot_sequential.py` pick a *set* of pushes by greedy
re-simulation from each deformed state. That answers "which three pushes would
I pick at once", which is not the loop: the loop plans one trajectory per
iteration, executes it, refits, and recomputes. Keep the pair for offline
comparison only, and expect 24 rollouts per candidate per round.

Read `gains.json` for per-action sketch gains and `selected`, and
`comparison.json` for sketch-versus-exact rank agreement. Sketch gains order
actions; only the exact numbers are reportable magnitudes.

## 5. Sim2sim smoke test and a planned capture

```bash
python scripts/plan_mujoco_smoke.py 2>&1 | tee logs/mujoco_smoke_$TAG.log

python scripts/plan_push_trajectory.py --case $BASE/$CASE \
  --frames 60 --nodes 500 --groups 20 --azimuth-count 6 --tilts 0 \
  --knots 3 --samples 8 --iterations 3 --offset-limit 0.05 --exact-top 4 \
  --out experiments-fim/$CASE/trajectory_$TAG.json \
  2>&1 | tee logs/trajectory_$TAG.log
python scripts/generate_sim_data.py models.json --case $CASE \
  --trajectory experiments-fim/$CASE/trajectory_$TAG.json \
  --output_dir experiments-fim/$CASE --output_name plan_$TAG
```

The first command scores three coarse candidates and executes the winner for
six RGB-D frames; the tool is two pad centres there, so it checks plumbing only.

The second pair screens the azimuth/tilt family, refines the winner with CEM
over a spline (waypoint offsets, time warp, ±30° claw yaw/pitch) and replays the
plan through the generator, so the capture has the case's own model, cameras,
848x480 resolution, `capture_every` and fps: `color/{0,1,2}.mp4` is comparable
to `data/different_types/<case>/color/`. Allow ~65 min at the defaults: a
six-action screen (~13 min, ~4 min on a `.factor_cache` hit) plus 25 CEM
evaluations at ~2.1 min each. Lower `--samples`/`--iterations` to budget a
shorter plan; the screen is the part the cache removes. See
[planning.md](planning.md) for the parameterization and the score.

Read the result honestly. The objective is the sketch gain at the nominal
grouped parameters with the prior only (`basis` in the JSON), so it is not
conditioned on a fitted estimate. It is exactly flat before the claw reaches the
object (measured: bit-identical observations at 24 frames), which is why
candidates that never contact are rejected outright. After contact the score is
dominated by contact sensitivity, so in MuJoCo a plan that just presses harder
can look like a plan that is more informative. Compare `trajectory_$TAG.json`
with the recorded push: the plan must be visibly curved or re-oriented, and
`score` must exceed `start_gain` by more than noise, otherwise the frames are
too short to contain contact.

## 6. Warm-start continuation

```bash
cat > logs/continual_$TAG.json <<EOF
{
  "case_name": "$CASE",
  "base_path": "$BASE",
  "output_dir": "experiments-fim/continual_$TAG",
  "frame_prefixes": [70, 90, 110],
  "iterations_per_round": 5,
  "tracked_points": 500
}
EOF
python scripts/continue_warp.py logs/continual_$TAG.json 2>&1 | tee logs/continual_$TAG.log
```

Prefixes must increase from >= 2 and stay within the 124 captured frames. Use
prefixes past ~frame 64 if you want controller contact in the window: earlier
bounded runs on frames 3–12 reported `contacts 0/0`, so nothing was exercised
there. `tracked_points` subsamples the model (500 of 4779) and drops the
interior points, so results are not the full model. Each `round_NNN/` keeps its
input prefix, `training.log`, `iter_*.pth`, `parameters.npz` and `summary.json`;
`history.json` records the run. `output_dir` must not already exist.

## Notes

- Everything above writes new directories; nothing overwrites existing runs
  except stage 1 and a repeated `experiments/$CASE/train` from stage 2.
- Warp prints `Running the tape backwards may produce incorrect gradients ...`
  warnings for `set_contact_points`/`set_control_points`. They come from the
  upstream trainer and are not caused by the ASID additions.
- The planner still scores the nominal grouped parameters (the mean is not
  refit) and `asid/planning.md` documents the open contact-gradient and
  calibrated-sim2sim gaps.
