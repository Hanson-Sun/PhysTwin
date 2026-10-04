# Integration Notes

## Branch convention

Work is split across phase branches, each branched from the previous one:

- `PRSI/1-phystwin-real2sim` — real-to-sim pipeline on PhysTwin/Warp
- `PRSI/2-asid-integration` — all ASID work, built on top of phase 1

Keep phase 2 concerns off the phase 1 branch, so each phase stays reviewable on
its own.

## Background

The PhysTwin integration combines Boba's runtime with the original
PhysTwin preprocessing and warp-training pipeline. Boba's runtime trainer remains
unchanged; the upstream training trainer is isolated under
`qqtt/engine/trainer_warp_upstream.py`.

## Environments

Two conda environments are required:

- `phystwin-cu132` — the main runtime: warp training, Gaussian reconstruction, and RL (PyTorch 2.12 / CUDA 13.2).
- `phystwin-data` — RGB-D data processing only: segmentation, tracking, shape priors. The legacy sam2/GroundingDINO stack needs the older PyTorch 2.4 / CUDA 12.1 toolchain, so it cannot live in the main environment.

Set up `phystwin-data` once:

```bash
conda env create -f env_install/phystwin-data.yml
conda activate phystwin-data
bash env_install/install_data_processing.sh
bash env_install/download_data_checkpoints.sh
```

Run everything else from `phystwin-cu132`. `run_case_pipeline.py` finds the
`phystwin-data` python automatically; pass `--data_python` only if your
environment is in a non-standard location.

## Process RGB-D Videos

RGB and depth videos must be paired and require camera calibration. Video files
do not reliably contain camera intrinsics or camera poses.

```bash
python scripts/process_rgbd_case.py \
	--rgb_video /path/to/rgb.mp4 \
	--depth_video /path/to/depth.mp4 \
	--output_dir data/different_types \
	--case_name my_case \
	--category cloth \
	--intrinsics /path/to/intrinsics.npy \
	--c2w /path/to/camera_to_world.npy \
	--depth_scale 1.0
```

Use `--depth_scale 1000` when the depth video stores metres. The segmentation pipeline defaults to the single robot prompt
`robot gripper.` and recognizes `hand`, `claw`, `gripper`, and
`robot gripper` as controller labels while preserving the downstream
`controller` mask name. Use
`--controller_prompt` and `--controller_names` on `process_data.py` to override
those defaults. Add `--shape_prior` to enable shape-prior processing. During
GroundingDINO segmentation, the highest-confidence non-controller detection is
kept per camera, while controller detections are preserved. This prevents one
object from becoming multiple tracked objects. Shape-prior alignment upsamples
the masked crop and selects render views using matches with valid rendered depth.
The command writes
`final_data.pkl`, `metadata.json`, `calibrate.pkl`, and intermediate data under
`<output_dir>/my_case/`.

## Shape Prior Backends (`trellis` / `interior`)

**What changed:** the shape-prior stage either generates a mesh (`trellis`, generative image-to-3D) or skips mesh generation entirely and populates interior points directly from masks and depth (`interior`, `data_process/interior_sample.py`). The former `carve`/`poisson` mesh backends were removed — their only volumetric consumer was interior-point sampling, which `interior` now serves without a watertight mesh, a mesh-repair stage, or the alignment stage.

**User action:** none required — defaults to `trellis`. Opt in per-case via the 4th column of `data_config.csv` or `--shape_generator {trellis|interior}`.

- `trellis` — generative image-to-3D from a single upscaled, segmented RGB crop
  (`data_process/shape_prior.py`). Fast, but the geometry is hallucinated, so
  thin or obliquely viewed objects can come out wrong. Produces the aligned
  `shape/matching/final_mesh.glb` used by interior sampling and the Gaussian
  stage's `shape_prior.glb`.
- `interior` — mesh-free interior-point sampling (`data_process/interior_sample.py`).
  Carves a voxel occupancy grid (inside every object mask AND never in front of
  the observed depth), erodes one voxel from the free-space boundary, and writes
  one jittered point per kept voxel to `shape/interior_points.npy` (+ `.ply` for
  inspection). No mesh, no alignment stage, no watertightness repair; runs in
  seconds. Thin geometry (cloth, rope) has no interior at voxel resolution and
  yields zero interior points, which is correct. The Gaussian stage falls back
  to the observed point cloud (`observation.ply`) since no mesh exists.

`data_config.csv` selects `interior` for the sim cases:

```csv
sim_rigid_box_heavy_end, rectangle box, True,interior
```

Override the configured backend for one run with `--shape_generator`:

```bash
python scripts/run_case_pipeline.py \
	--case_name sim_rigid_box_heavy_end \
	--shape_generator interior
```

Run the sampler on its own to inspect the points (it writes
`interior_points.npy`, `interior_points.ply`, `interior_points.mp4` and
`interior_stats.json` into `shape/`):

```bash
python data_process/interior_sample.py \
	--base_path data/different_types \
	--case_name sim_rigid_box_heavy_end
```

### Interior-fill diagnostics

Every run prints (and stores in `shape/interior_stats.json`) the numbers that
say whether the fill is good, plus a `interior_points.mp4` turntable of the
carved hull (gray, lower half cut away) with the interior points (red) inside.
Pass `--no_visualize` to skip the video.

- `interior_fill_ratio` — interior volume / occupied volume. Near 1 means a
  solid body; low values mean the object is thin at the current voxel size.
- `blobs` / `largest_blob_fraction` — number of separate interior regions and
  the share of the biggest one (1.0 = one solid blob).
- `mean_observed_distance` — mean distance from an interior point to the nearest
  observed surface point. Large values mean the interior is mostly volume no
  camera ever saw (visual-hull bulge).
- `observed_without_interior` — fraction of observed surface with no interior
  point within `--support_radius` (10 mm). These are thin parts (fingers, rope,
  cloth edges) that cannot hold interior samples at the current voxel size.
- `points_below_floor` — must always be 0; anything else means points are being
  generated underneath the support surface.
- `observed_bbox_extent_m` / `interior_bbox_extent_m` — the interior should not
  bulge past the observation.

### Floor awareness

The support surface is `z = 0` and **`z > 0` is below the floor** in this
dataset's world frame (the same convention `data_process_sample.py` and
`align.py` use when they snap `z > 0` points back to `z = 0`). Carving treats
everything below the floor as free space, and observed points below it are
dropped as depth noise, so no interior point is ever generated inside the
table. Use `--floor_z` if the support surface is not at `z = 0`.

Useful options:

- `--voxel_size` — voxel edge in metres; the detail/resolution trade-off (and
  the effective minimum feature size that can hold interior points).
- `--max_points` — cap on the number of interior points written (default 10000,
  matching the previous `volume_mesh` sampling budget).
- `--mask_erode` — pixels to erode each object mask before carving; raise it to
  trim the segmentation fringe.
- `--depth_margin` — voxels closer than this to the observed surface are not
  carved (depth-noise tolerance).

## Train Warp Parameters

```bash
python scripts/train_warp_case.py \
	--base_path data/different_types \
	--case_name my_case \
	--visualize
```

This runs CMA initialization followed by differentiable warp training. Saved Warp
training videos are enabled by default; use `--no_visualize` for a headless run.
 The best weights are
saved under `experiments/my_case/train/best_*.pth`, with a manifest at
`experiments/my_case/train/training_manifest.json`. Use `--skip_cma` to skip
the initialization stage only when
`experiments_optimization/my_case/optimal_params.pkl` already exists. For a
small headless end-to-end test, use `--iterations 1 --cma_max_iter 1 --no_visualize`.

### Dense controller contact (new, automatic)

**What changed:** training now uses the hollow dense gripper shell (`controller_points_dense`, voxel `0.003`m) as a unilateral collider instead of sparse spring tethers. Contact is a swept-segment penalty `F=k*pen` with Coulomb friction (`μ*F_n` capped via `tanh`), hysteresis (`activation 0.014 / release 0.0175`m), and mass-normalized stiffness (`k_eff=k_base*m_i` so penetration `a=k_base*pen` is `m`-independent). Prevents tunneling at `capture_every=8` and wires into the Boba batched runtime.

**User action:** none. Dense tracks are preserved automatically (`qqtt/data/real_data.py`, `qqtt/utils/controller_collider.py`). CMA reuses the same fixed topology (`controller_radius`/`max_neighbours` not re-optimized). Visualize coverage with `scripts/visualize_controller_points.py --dense --hollow`.

### Differentiable ground & friction (new, automatic)

**What changed:** hard ground impact / `min(μN, m·v/dt)` is replaced by a smooth sigmoid ground activation (`ground_contact_smoothing=2e-5`) and `tanh` Coulomb saturation (`tanh(v·k / μN)`). Object-object collisions keep hard impact timing but smooth the friction cap. Improves gradient flow for `collide_fric` without changing the visual result.

**User action:** none. Enabled in both `spring_mass_warp*.py`. No config required; CMA/Adam learn the same `collide_*` scalars more stably.

### Training visualization & controller overlay (new)

Warp training now saves an MP4 every `vis_interval` (default 20) under `experiments/<case>/train/` via a separate forward graph, and the standalone `visualize_controller_points.py` renders any `inference.pkl`/`final_data.pkl` trajectory as multi-camera Open3D panels with an optional hollow dense gripper inset.

```bash
python scripts/visualize_controller_points.py data/different_types/sim_rope/final_data.pkl data/different_types/sim_rope/vis.mp4 --dense --hollow
```

### Learnable anchored mass field (heavy-end / CoM shift)

Disabled by default; no config change required. Enable with one flag:

```yaml
# configs/real.yaml (or per-case override)
learn_mass: true          # default false — off = uniform mass, fully backward compatible
init_mass: 1.0            # uniform scale baked into all masses at t=0
mass_min: 0.1             # clamp per-vertex mass after exp()
 mass_max: 10.0
mass_reg_weight: 1.0e-4   # variance penalty on log-mass (mean is free)
 mass_smooth_weight: 0.0   # Laplacian on anchor graph (0 = off)
# Anchored (default K=128 FPS, K<=0 or K>=N falls back to per-vertex N):
mass_num_anchors: 128
mass_anchor_knn: 4         # inverse-distance kNN to N points
 mass_anchor_seed: 42
mass_anchor_method: fps    # fps (default) or random
```

YAML keys are optional — unset keys keep the defaults above. Set via YAML or
`cfg.update_from_dict({"learn_mass": True})`. CMA now optimizes an 11th
dimension `global_mass` (normalized in `[mass_min, mass_max]`) automatically and
maps `global_mass → init_mass`; no user action needed. Checkpoints with
`learn_mass: true` save `log_mass` (K=128) + `masses` (N) + `mass_anchor_*`
metadata and reload strictly (no resampling) — from this point forward all
checkpoints are assumed `fps`-anchored.

### Learnable global damping (rope springiness fix) — optional

Disabled by default; enable to let Adam fine-tune the two globals already tuned by CMA:

```yaml
# configs/real.yaml
learn_damping: true       # default false — off = frozen (CMA warm-start only)
dashpot_damping: 100      # initial per-spring Kelvin-Voigt [0,200]
drag_damping: 3           # initial global velocity decay [0,20]
```

Single flag `learn_damping: true` makes both `dashpot` + `drag` trainable in Adam (2 scalars, `clamp[0,200]`/`[0,20]`, `0.2×lr`, graph-baked). No other change; with `false` behavior is identical to before. Useful for `sim_rope` where `chamfer+track+acc(0.01)` position loss under-constrains velocity — check `wandb` `dashpot_damping`/`drag_damping`. Checkpoints always save/restore both values. No anchored/per-spring field — global only (minimal change).

### Which parameters are learnable

There are two independent optimization stages with **different, overlapping** parameter sets. `scripts/train_warp_case.py` runs CMA-ES first, then Adam.

**CMA-ES** (`qqtt/engine/cma_optimize_warp.py`) optimizes 11 normalized dimensions in `[0,1]`, denormalized in `error_func`:

| # | Parameter | Range | Meaning |
|---|---|---|---|
| 0 | `init_spring_Y` | `[spring_Y_min, spring_Y_max]` | global spring Young's modulus |
| 1 | `object_radius` | `[0.01, 0.05]` | object-object spring search radius |
| 2 | `object_max_neighbours` | `[10, 50]` | object-object spring neighbour cap |
| 3 | `collide_elas` | `[0, 1]` | object restitution |
| 4 | `collide_fric` | `[0.3, 2.0]` | ground friction |
| 5 | `collide_object_elas` | `[0, 1]` | object-object restitution |
| 6 | `collide_object_fric` | `[0, 2]` | object-object friction |
| 7 | `collision_dist` | `[0.01, 0.05]` | self-collision query radius |
| 8 | `drag_damping` | `[0, 20]` | global velocity decay |
| 9 | `dashpot_damping` | `[0, 200]` | per-spring Kelvin-Voigt damping |
| 10 | `global_mass` | `[mass_min, mass_max]` | maps to `init_mass` |

`controller_radius` and `controller_max_neighbours` are **explicitly fixed** during CMA — the claw's spring topology is not re-derived per candidate, so candidates stay comparable.

**Adam** (`qqtt/engine/trainer_warp_upstream.py`, `trainable_parameters`) learns per-spring / per-particle fields rather than globals:

- Always: `spring_Y` (per-spring), `collide_elas`, `collide_fric`, `collide_object_elas`, `collide_object_fric`
- Only when `controller_contact_enabled`: `controller_contact_stiffness`, `controller_contact_friction`
- Only when `learn_mass`: `log_mass` (per-vertex or K-anchored)
- Only when `learn_damping`: `dashpot_damping`, `drag_damping` (both at `0.2×lr`)

`log_mass` / `dashpot_damping` / `drag_damping` share a reduced learning rate (`_special` set) because they are global scalars whose gradients are otherwise poorly scaled.

**Not learnable at any stage** (fixed config values): `controller_contact_radius`, `controller_contact_activation_radius`, `controller_contact_release_radius`, `object_total_mass` (CMA tunes the `init_mass` *scale*, not the total), `controller_radius`, `controller_max_neighbours`.

Note the two stages see **different claw geometry**: CMA passes the sparse 30-point FPS proxy as `controller_contact_points`, while Adam trains on the dense hollow shell. Since 30 points cannot cover a 0.26 m claw at an 0.018 m contact radius, CMA's objective is effectively blind to grip capacity — see [Grip capacity is radius-bound](#grip-capacity-is-radius-bound-not-stiffness-bound).

### `controller_radius` / `controller_max_neighbours`

These two build the **legacy spring tethers** from claw to object, in `_init_start` (`qqtt/engine/trainer_warp_upstream.py`). They are *not* contact parameters and have no effect on grip force.

```python
for i in range(len(controller_points)):
    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
        controller_points[i], controller_radius, controller_max_neighbours,
    )
    for j in idx:
        springs.append([num_object_points + i, j])
```

For each of the sparse claw points, find object nodes within `controller_radius` (default `0.04` m) and link up to `controller_max_neighbours` (default `50`) of them. Those links become ordinary springs evaluated by `eval_springs`, pulling the object toward the claw's rest offset — a bilateral attachment, not a unilateral contact.

Consequences worth knowing:

- They use the **sparse** `controller_points` (30 FPS-sampled), not `controller_points_dense`. Denser claw tracking does not add tethers.
- Because they are bilateral, they also pull the object *down* when it rests below the claw — which is why grip failures can look like the object following the claw in one direction but not the other.
- `controller_radius` too large starts grabbing nodes far from the pad surface, effectively welding a wide region of the object to the claw.

The modern path is unilateral dense contact (`controller_contact_*`), which is why the tether parameters are frozen: changing them mid-pipeline would invalidate comparison against previously trained checkpoints.

### Grip capacity is radius-bound, not stiffness-bound

For contact to lift the object, the friction-limited force must clear its weight:

```
mu * K * sum(penetration)  >  m * g          mu = controller_contact_friction
```

`sum(penetration)` is dominated by `controller_contact_radius`, which is **not learnable**. Measured peak grip margin on the recorded soft-lift cases (0.3 kg, so weight = 2.94 N, `mu` = 0.3, **before** the occlusion fix below):

| Case | `sum_pen` @ r=0.018 | margin @ K=250 | margin @ K=2000 |
|---|---|---|---|
| `sim_soft_ball_grip_lift` | 0.1507 | 3.84× | 30.7× |
| `sim_soft_sloth_grip_lift` | **0.0052** | **0.13×** | **1.06×** |
| `sim_soft_seal_grip_lift` | 0.1128 | 2.88× | 23.0× |
| `sim_soft_teddy_bear_grip_lift` | 0.2296 | 5.85× | 46.8× |

The sloth was ~29× weaker than the ball, so no reachable stiffness fixed it: `K=2000` only reached 1.06× and sits at 84% of the explicit-integration ceiling for its 12670 nodes (`K_max = node_mass * (0.5/dt)^2` = 2368 N/m). **Prefer fixing the collider over inflating either knob** — see the next section, which removes the need for both.

### Occluded claw points are refilled, not discarded (new, automatic)

**What changed:** the dense gripper collider used to keep only claw points visible in *every* frame (`mask = np.prod(controller_visibilities, axis=0)` in `filter_motion`). A grasped object hides the pad face that has to make contact, so that intersection discards the gripper's most important geometry exactly during the lift. `get_final_track_data` now keeps every point the gripper mask saw in *at least one* frame and rebuilds the hidden frames with `qqtt.utils.controller_collider.fill_occluded_controller_points`:

- hidden between two observations -> linear interpolation between them,
- hidden with only one side observed -> hold that measurement,
- never observed -> dropped and counted in the log line.

This is safe, unlike simply relaxing the filter. `filter_track` leaves unobserved frames as **zeros**, not estimates, so an interpolated frame is built from two real measurements, and the one-sided case clamps to an observation instead of extrapolating through the occluder. Extrapolating there is what would let the collider reach past the object and "grip" through it.

**Effect** (`sim_soft_sloth_grip_lift`, at the unchanged `controller_contact_radius: 0.018`):

| | claw pts/frame | `sum_pen` | grip margin |
|---|---|---|---|
| all-frames intersection | 2108 | 0.00518 | **0.13×** |
| per-frame observation + fill | ~4300 | 0.08786 | **2.24×** |

The ball also improves (3.84× → 9.13×) because it loses pad points too, just fewer.

**User action:** none, but the tracking stages must be re-run for existing cases — `--skip_process` keeps the old `final_data.pkl` and the fix will not appear. Expect `Dense Controller Point Number` in the log to be larger than before; it now reports the all-frames count alongside it for comparison.

Note this supersedes the radius workaround: with the collider refilled, `controller_contact_radius: 0.018` clears the object's weight on its own, so the raised `0.025` / stiffness `2000` currently in `configs/soft.yaml` are no longer needed and push contact ~7 mm deeper than the real claw reaches. Reverting them is the physically faithful setting.

Inspect with `scripts/visualize_controller_points.py --dense --hollow`, and compare `min(claw→object)` distance per frame: a healthy grip sits well inside `controller_contact_radius`, not just outside it.

### The four claw-point thresholds

Claw points go through two gates. The first asks "did we actually see this point?" The second asks "is it sitting where the claw really is?". Both drop whole tracks, never individual frames, because the collider needs the same point count every frame.

**Gate 1 — visibility.** `controller_visibility_threshold`, default `0.80`. A track must be visible in at least 80% of frames. "Visible" means CoTracker saw it *and* its pixel was still inside the controller mask in `filter_track`. The missing 20% get filled in afterwards (interpolate between two real measurements, or hold the nearest one). No geometry involved — this gate is purely about tracking quality.

**Gate 2 — depth consistency.** Runs after filling, in `controller_depth_consistency`. It projects each point into every camera and compares against the measured depth image. Every (frame, camera) pair ends up in one of three buckets:

- **supported** — lands inside the controller mask and matches measured depth
- **contradicted** — floats in front of the measured surface, or matches depth but the mask says it isn't claw
- **neutral** — buried behind the measured surface, off-screen, or no depth at that pixel. Counts as neither.

Two settings decide those buckets and the final verdict:

| Setting | Default | What it means |
|---|---|---|
| `depth_tolerance` | `0.004` (4 mm) | How close counts as "matching". Not a filter itself — it's the width of the agreement band. |
| `min_support_fraction` | `0.7` | Must be supported in at least 70% of frames. |
| `max_conflict_fraction` | `0.10` | May be contradicted in at most 10% of frames. |

A track survives gate 2 only if it clears both fractions. The `Depth consistency: kept X/Y tracks; median support=…, median conflict=…` log line prints the distributions *before* the cutoffs, so use it to decide what to change instead of guessing.

**Three things worth knowing before you tune these:**

1. **`depth_tolerance` is not a "looser = keep more" knob.** Widen the band and a previously-neutral frame becomes supported *or* contradicted, depending on whether its pixel is inside the mask. It can cut either way, and past ~10 mm it starts accepting points that visibly float off the claw.

2. **`min_support_fraction: 0.7` is in tension with the occlusion refill above.** Support is counted over *all* frames, and neutral frames don't help it. A pad point hidden for 30% of the lift can never exceed ~0.70 support, so it gets rejected — which throws away exactly the geometry the refill exists to recover. If support is the binding constraint in the log, this is the number to lower (try ~0.25–0.35, roughly one minus the fraction of frames typically occluded).

3. **`max_conflict_fraction` is rarely the binding one.** Contradicted frames are uncommon in practice; the median is usually near zero. Loosening it past 0.10 tends to change very little.

Only `controller_visibility_threshold` is a CLI flag (`--controller_visibility_threshold`). The three depth settings are hardcoded at the `controller_depth_consistency(...)` call in `data_process/data_process_track.py`, so changing them means editing that call.

## Reconstruct Gaussians

Run this once after RGB-D processing and before RL:

```bash
python scripts/reconstruct_gaussians.py \
	--base_path data/different_types \
	--case_name my_case \
	--output_dir gaussian_output/my_case \
	--use_masks
```

This prepares the first-frame Gaussian dataset and trains the static appearance
model. The resulting model is written to
`gaussian_output/my_case/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply`.
The command also creates the pruned Boba asset under
`gaussian_output_pruned_policy_30_55/my_case/` by default. During RL, use the
pruned asset with Boba's `batch_prune` renderer and reuse both Gaussian assets
and the trained warp weights; do not reconstruct them for every episode.

## Run Everything

For a case listed in `data_config.csv`, the category and default shape-prior
setting are selected automatically:

```bash
python scripts/run_case_pipeline.py \
	--case_name double_lift_sloth \
	--visualize
```

This runs RGB-D processing, CMA and warp training, Gaussian reconstruction, and
Gaussian pruning in order. Use `--dry_run` to inspect the resolved settings,
`--smoke_test` for a minimal run when a compatible
`optimal_params.pkl` already exists, or `--skip_process`, `--skip_warp`, and
`--skip_gaussians` to resume from an existing stage.

Use `--no_shape_prior` to override a configured shape prior,
or `--shape_prior` to enable it for a case configured without one. Use
`--skip_segmentation` to resume after a completed segmentation stage.
The full pipeline enables preprocessing visualizations and saved Warp training
videos by default. Pass `--no_visualize` for a headless run.
 Warp checkpoints are written under
`experiments/<case_name>/train/`. Runtime playback always selects the highest
numbered available `best_<iteration>.pth` checkpoint rather than relying on
filesystem glob order.

RGB-D processing automatically runs in the `phystwin-data` environment (see
[Environments](#environments)).

## MuJoCo Synthetic RGB-D Export (new)

**What changed:** adds a full MuJoCo → PhysTwin data path (`mujoco_sim/{simulation,scene,interactor,phystwin_export}` + `mujoco_assets/{world,claw,object_box/rope/heavy_end}.xml` + `models.json`/`scripts/generate_sim_data.py`). Useful for controlled ablations (uniform vs heavy-end) without real capture.

**User action:** optional. Normal pipeline unchanged. To generate synthetic data, use either single-case API or manifest. The exporter writes a PhysTwin-compatible case directory, including:

- `color/<camera>/<frame>.png` RGB frames
- `color/<camera>.mp4` videos for dense tracking
- `depth/<camera>/<frame>.npy` uint16 depth in millimetres
- `calibrate.pkl` camera-to-world poses in the OpenCV camera convention
- `metadata.json` with intrinsics, image size, FPS, and frame count
- `split.json` with the standard 70/30 train/test frame ranges

Default example loads the validated rigid `mujoco_assets/object_box.xml`.
Soft meshes are tetrahedralized into MuJoCo flexes on the fly at load time.
Active assets:

- `object_rope.xml` — simple flexible rope/twine scaffold
- `object_box.xml` — validated rigid baseline box
  so alignment and reconstruction have non-trivial visual features
- `sim_rigid_box_grip_lift` — single-claw open/close grip-and-lift case using
  the textured rigid box
- `object_sloth.stl` — watertight sloth flex mesh lying on its back with the
  head toward camera 0, prepared from the repaired shape prior by
  `mujoco_sim/prepare_mesh.py` and shipped with `object_sloth.png` +
  `object_sloth.uvsrc.npz` appearance sidecars
- `sim_soft_sloth_grip_lift` — single-claw grip-and-lift of the textured soft
  sloth (see [Soft sloth case](#soft-sloth-case-new-sim_soft_sloth_grip_lift))
- `object_octopus.stl` / `object_seal.stl` / `object_teddy_bear.stl` — toy GLBs
  scaled with `--target-height 0.225 --max-dim 0.30` after a y-up → z-up roll, each
  with texture sidecars; the octopus additionally yawed 180° so a tentacle
  lands inside the claw's grip span (its head sits at the model centre, so the
  head always lies between the pads)
- `sim_soft_octopus_grip_lift` / `sim_soft_seal_grip_lift` /
  `sim_soft_teddy_bear_grip_lift` — grip-and-lift cases for those three toys

The previous procedural cloth, doll, sloth, zebra, package, and compound-plush
stand-ins are not active in `models.json` because they are not proper connected
volume meshes. The real PhysTwin `shape/object.glb` files are non-watertight,
disconnected surface reconstructions; `mujoco_sim/prepare_mesh.py` repairs the
sloth into the watertight `object_sloth.stl` flex asset (the zebra is still
pending).

Select an active asset via JSON `object_file` or `load_model(object_file=...)`. Preprocessing requires 3 cameras (as in `mujoco_assets/world.xml`):

```python
from mujoco_sim.phystwin_export import export_case

# After collecting frames with DigitalTwinSim:
export_case(sim, frames, "data/different_types/my_sim_case")
```

Generate multiple simulation cases from a JSON manifest:

```json
{
  "models": [
    {
      "case_name": "sim_rigid_box",
      "object_file": "object_box.xml",
      "width": 848,
      "height": 480,
      "steps_per_segment": 300,
      "capture_every": 8,
      "substeps": 4,
      "fps": 30
    },
    {
      "case_name": "sim_rigid_box",
      "object_file": "object_box.xml"
    }
  ]
}
```

Run this from the repository root to write cases under the standard PhysTwin
root `data/different_types/<case_name>/`:

```bash
python scripts/generate_sim_data.py models.json
python scripts/generate_sim_data.py models.json --output_dir data/different_types --overwrite
```

Use `--dry_run` to validate and list every manifest entry without rendering.
The `phystwin-cu132` environment includes the MuJoCo package's bundled
elasticity plugin libraries, and the scene loader loads them automatically.

### New stuffed animal from a GLB (new)

Everything needed to turn an arbitrary toy `.glb` into a PhysTwin-format sim
case. None of the PhysTwin capture stages (segmentation, tracking, shape
prior, calibration) are involved: the three cameras, claw, and floor come
from `mujoco_assets/world.xml`.

**What you provide — one `.glb`:**

- **Real-world scale in metres**, roughly 0.05–0.5 m. `prepare_mesh` prints
  the extents; if they look like millimetres, scale the file first — or scale
  at prepare time with `--target-height 0.225 --max-dim 0.30` (height target
  bounded by the longest extent; applied before the voxel fill, and keep the
  grip-station cross-section under the claw's 0.20 m open gap).- Open or fragmented meshes are fine — `prepare_mesh` voxel-fills them watertight. Already-watertight inputs skip the repair entirely — GLB texture-seam
  vertices are welded by position first, so only genuinely open meshes take
  the voxel path (pass `--remesh` to force it, e.g. to decimate a dense mesh). The fill rounds off features below ~2 voxels, so the default `--pitch` is 0.003 m (features below ~6 mm still smooth out); pass e.g. `--pitch 0.0015` for an even finer repair at higher mesh/sim cost.
- A UV-unwrapped base-colour texture is optional: present → ported as
  `<stem>.png` + `<stem>.uvsrc.npz` sidecars; absent → flat colour from
  `soft.rgba` in the manifest.
- The pose only has to rest stably on the floor; orient it with
  `--rotate-x-deg/--rotate-y-deg/--rotate-z-deg` (extrinsic, applied x → y → z).

**1. Build the asset** (writes the STL plus appearance sidecars):

```bash
python -m mujoco_sim.prepare_mesh path/to/toy.glb mujoco_assets/object_toy.stl
```

**2. Add a manifest entry** to `models.json`:

```json
{
  "case_name": "sim_soft_toy_grip_lift",
  "object_mesh": "object_toy.stl",
  "trajectory": "grip_lift",
  "n_interactors": 1,
  "soft": {
    "rgba": "0.47 0.42 0.37 1",
    "young": 350,
    "cellcount": "4 6 1"
  },
  "width": 848,
  "height": 480,
  "steps_per_segment": 300,
  "capture_every": 8,
  "substeps": 4,
  "fps": 30
}
```

`soft` is passed straight to `soft_body.soft_object` (see the sloth section
for why `cellcount` and `young` matter). `rgba` only applies when there is no
texture sidecar — when the texture exists the flex is forced white so the
material does not tint it. The soft asset is re-centred on the origin at load
time, and the grip contact height is the object's vertical centre (with a
palm-clearance rule for objects taller than the claw's reach).

**3. Generate and check:**

```bash
python scripts/generate_sim_data.py models.json --dry_run   # validate only
python scripts/generate_sim_data.py models.json \
    --case sim_soft_toy_grip_lift --overwrite
python -m unittest tests.test_soft_body tests.test_soft_sloth
```

Output lands in `data/different_types/<case_name>/` in the standard PhysTwin
layout listed above.

**Grip constraint** (`grip_lift` only): the claw closes along world x with a
0.20 m open gap, so the toy's cross-section at the grip station (the middle
of the body) must be narrower than that — otherwise use `"trajectory":
"push"`, which has no such limit.

### Soft sloth case (new): `sim_soft_sloth_grip_lift`

The real PhysTwin sloth runs as a textured soft flex. Regenerate the asset
(and its appearance sidecars) from the repaired shape prior, then the case:

```bash
python -m mujoco_sim.prepare_mesh \
    data/different_types/double_lift_sloth/shape/matching/final_mesh.glb \
    mujoco_assets/object_sloth.stl \
    --rotate-x-deg 180 --rotate-z-deg 90

python scripts/generate_sim_data.py models.json \
    --case sim_soft_sloth_grip_lift --overwrite
```

- **Texture**: `prepare_mesh` writes `<stem>.png` + `<stem>.uvsrc.npz` beside
  the mesh. `soft_body.soft_object` detects them, emits the texture/material
  block, and passes the flex texcoords through the asset dict — MJCF cannot
  store texcoords on a `flexcomp`, so `scene.compile_scene` attaches them via
  `MjSpec` and forces the flex rgba white (rgba would tint the texture).
  Texcoords come from a nearest-triangle projection of the source UV atlas
  (best of 24 candidate triangles per node by projection distance), so texture
  only mislands where the voxel fill fused separate layers.
- **Pose**: the asset lies on its back (belly up) with the head pointing at
  camera 0 (`cam_front`, -y); the claw grips the torso, whose mid-body section
  (0.156 m) fits the pads' 0.20 m open gap. Rotations are extrinsic and apply
  in x, then y, then z order.
- **Material** (`models.json` → `soft`): `cellcount "4 6 1"` splits the flex
  into a real grid — MuJoCo's default single 8-node cell can only warp the
  whole body as one affine blob (limbs cannot articulate and the body bulges
  at rest). `young: 350` (vs the `1e4` default) lets the limbs sag under
  gravity: rest z-span 0.096 m grows to ~0.30 m while suspended, for a +0.15 m
  lift.

To inspect one of the exported depth maps:

```bash
python -m mujoco_sim.visualize_depth \
    --input data/different_types/my_sim_case/depth/0/0.npy \
    --output data/different_types/my_sim_case/depth/0/0_preview.png
```

The visualizer treats PhysTwin integer depth files as millimetres and floating
point depth files as metres by default. Use `--unit meters` or
`--unit millimeters` to override that behavior.

## Env modifications

1. `pip install mujoco`
2. `pip install gymnasium`

## Controller Point Visualization

`scripts/visualize_controller_points.py` renders any `inference.pkl` (or `final_data.pkl`) trajectory as one calibrated Open3D panel per camera plus an optional hollow dense gripper overlay. Videos are stitched horizontally at native resolution.

```bash
python scripts/visualize_controller_points.py data/different_types/sim_rope/final_data.pkl data/different_types/sim_rope/vis.mp4 --dense --hollow
# opts: --trajectory PATH --trail-length N --inset-scale 0.34 --voxel-size 0.003
```
