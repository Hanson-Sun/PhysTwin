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

## Shape Prior Backends

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

## MuJoCo Synthetic RGB-D Export

The MuJoCo exporter writes a PhysTwin-compatible case directory, including:

- `color/<camera>/<frame>.png` RGB frames
- `color/<camera>.mp4` videos for dense tracking
- `depth/<camera>/<frame>.npy` uint16 depth in millimetres
- `calibrate.pkl` camera-to-world poses in the OpenCV camera convention
- `metadata.json` with intrinsics, image size, FPS, and frame count
- `split.json` with the standard 70/30 train/test frame ranges

The default MuJoCo example now loads the validated rigid
`mujoco_assets/object_box.xml`. Proper soft-body assets will be added after
surface repair and tetrahedral volume generation. The active representative
assets are:

- `object_rope.xml` — simple flexible rope/twine scaffold
- `object_box.xml` — validated rigid baseline box
  so alignment and reconstruction have non-trivial visual features
- `sim_rigid_box_grip_lift` — single-claw open/close grip-and-lift case using
  the textured rigid box

The previous procedural cloth, doll, sloth, zebra, package, and compound-plush
stand-ins are not active in `models.json` because they are not proper connected
volume meshes. The real PhysTwin sloth/zebra `shape/object.glb` files are also
non-watertight, disconnected surface reconstructions and must be repaired and
tetrahedralized before they can be used as MuJoCo `dim=3` flex objects.

Select an active asset with the JSON generator's `object_file` field or with
`load_model(object_file=...)`.
The current PhysTwin preprocessing scripts require three cameras, matching the
three cameras in `mujoco_assets/world.xml`:

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

Use `scripts/visualize_controller_points.py` to create an MP4 from the
simulated vertex trajectory. It renders one calibrated headless Open3D panel per
camera and appends the panels horizontally, preserving each panel's native
resolution. The script prefers `inference.pkl` next to `final_data.pkl`; if it
is missing, it falls back to the object trajectory in `final_data.pkl`.

```bash
python scripts/visualize_controller_points.py \
  data/different_types/sim_rope/final_data.pkl \
  data/different_types/sim_rope/inference_with_controller_points.mp4 \
  --dense --hollow
```

Options:

- `--trajectory PATH`: use a different simulated vertex trajectory.
- `--dense`: show `controller_points_dense` instead of sparse points.
- `--hollow`: remove duplicate/interior dense voxels before visualization.
- `--inset-scale SIZE`: set the inset size as a fraction of the video; default is `0.34`.
- `--trail-length N`: show the previous N controller positions in orange.
- `--voxel-size SIZE`: configure hollow-shell voxel size; default is `0.003` meters.
