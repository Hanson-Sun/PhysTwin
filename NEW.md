# Integration Notes

The `asid-phystwin-integration` branch combines Boba's runtime with the original
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

## Shape Prior Backends (new: `carve` / `poisson` alongside `trellis`)

**What changed:** in addition to `trellis` (generative image-to-3D), two deterministic backends were added. All write the same watertight `shape/object.glb` consumed by alignment / interior sampling.

**User action:** none required — defaults to `trellis`. Opt in per-case via the 4th column of `data_config.csv` or `--shape_generator {carve|poisson|trellis}`.

Shape-prior generation writes the watertight object mesh that alignment and
interior-point sampling consume (`shape/object.glb`). The backend is chosen per
case by the optional fourth column of `data_config.csv`; a missing or empty
column falls back to `trellis`.

- `trellis` — generative image-to-3D from a single upscaled, segmented RGB crop
  (`data_process/shape_prior.py`). Fast, but the geometry is hallucinated, so
  thin or obliquely viewed objects can come out wrong.
- `carve` — deterministic depth-carved space carving
  (`data_process/shape_carve.py`). Carves free space from the object masks, depth
  maps, and camera poses, force-occupies the observed points, and extracts a
  watertight mesh. It also preserves the calibrated observed surface points for
  inspection and downstream refinement. It uses only masks, depth, and calibration, so it does not
  depend on the generative stage. Well-observed faces are accurate, but faces
  seen at grazing incidence (e.g. a horizontal top from near-horizontal cameras)
  and hidden sides are only weakly constrained, so they come out slightly domed
  or approximate rather than perfectly flat.
- `poisson` — smooth Poisson reconstruction from the calibrated back-projected
  surface points. It estimates normals, reconstructs a closed surface, trims
  low-density extrapolation, and rejects the result unless it is a valid
  watertight volume. Select it with `--shape_generator poisson`.

`data_config.csv` selects `carve` for the heavy-end box:

```csv
sim_rigid_box_heavy_end, rectangle box, True,carve
```

Override the configured backend for one run with `--shape_generator`:

```bash
python scripts/run_case_pipeline.py \
	--case_name sim_rigid_box_heavy_end \
	--shape_generator carve
```

Run the carver on its own to inspect the mesh (it writes `object.glb`,
`object.ply`, the calibrated observed points in `observed_points.ply` and
`observed_points_filtered.ply`, and, with `--visualize`, a turntable video):

```bash
python data_process/shape_carve.py \
	--base_path data/different_types \
	--case_name sim_rigid_box_heavy_end \
	--output_dir data/different_types/sim_rigid_box_heavy_end/shape
```

Useful options:

- `--voxel_size` — voxel edge in metres; the detail/resolution trade-off.
- Ground flattening is enabled by default at `z=0`; use `--ground_z` to change
  the support height or `--no_flatten_ground` to disable it.
- `--mask_erode` — pixels to erode each object mask before carving; raise it to
  trim the segmentation fringe and tighten the mesh.
- `--extract` — `isosurface` (default; marching cubes on the signed distance
  field, smooth and watertight) or `blocky` (exact but visibly stepped).
- `--allow_open` — write a non-watertight mesh instead of failing the stage.

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
  grip-station cross-section under the claw's 0.20 m open gap).- Open or fragmented meshes are fine — `prepare_mesh` voxel-fills them watertight. Already-watertight inputs skip the repair entirely (pass `--remesh` to force it, e.g. to decimate a dense mesh). The fill rounds off features below ~2 voxels, so the default `--pitch` is 0.003 m (features below ~6 mm still smooth out); pass e.g. `--pitch 0.0015` for an even finer repair at higher mesh/sim cost.
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
