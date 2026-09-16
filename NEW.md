# Integration Notes

The `asid-phystwin-integration` branch combines Boba's runtime with the original
PhysTwin preprocessing and warp-training pipeline. Boba's runtime trainer remains
unchanged; the upstream training trainer is isolated under
`qqtt/engine/trainer_warp_upstream.py`.

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

Use `--depth_scale 1000` when the depth video stores metres. Add
`--shape_prior` to enable shape-prior processing. The command writes
`final_data.pkl`, `metadata.json`, `calibrate.pkl`, and intermediate data under
`<output_dir>/my_case/`.

## Train Warp Parameters

```bash
python scripts/train_warp_case.py \
	--base_path data/different_types \
	--case_name my_case
```

This runs CMA initialization followed by differentiable warp training. The best
weights are saved under `experiments/my_case/train/best_*.pth`, with a manifest
at `experiments/my_case/train/training_manifest.json`. Use `--skip_cma` to skip
the initialization stage only when
`experiments_optimization/my_case/optimal_params.pkl` already exists. Use
`--iterations 1 --cma_max_iter 1 --no_visualize` for a small end-to-end test.

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
	--case_name double_lift_sloth
```

This runs RGB-D processing, CMA and warp training, Gaussian reconstruction, and
Gaussian pruning in order. Use `--dry_run` to inspect the resolved settings,
`--smoke_test` for a minimal run when a compatible
`optimal_params.pkl` already exists, or `--skip_process`, `--skip_warp`, and
`--skip_gaussians` to resume from an existing stage.

Use `--no_shape_prior` to override a configured shape prior,
or `--shape_prior` to enable it for a case configured without one. Use
`--skip_segmentation` to resume after a completed segmentation stage.
Warp training is headless by default and writes checkpoints under
`experiments/<case_name>/train/`.

RGB-D processing uses a separate `phystwin-data` environment because the legacy
GroundingDINO extension is incompatible with Boba's PyTorch 2.12/CUDA 13.2
environment:

```bash
conda env create -f env_install/phystwin-data.yml
conda activate phystwin-data
bash env_install/install_data_processing.sh
bash env_install/download_data_checkpoints.sh
python scripts/run_case_pipeline.py \
	--case_name double_lift_sloth \
	--data_python "$(conda info --base)/envs/phystwin-data/bin/python"
```## MuJoCo Synthetic RGB-D Export

The MuJoCo exporter writes a PhysTwin-compatible case directory, including:

- `color/<camera>/<frame>.png` RGB frames
- `color/<camera>.mp4` videos for dense tracking
- `depth/<camera>/<frame>.npy` uint16 depth in millimetres
- `calibrate.pkl` camera-to-world poses in the OpenCV camera convention
- `metadata.json` with intrinsics, image size, FPS, and frame count
- `split.json` with the standard 70/30 train/test frame ranges

The current PhysTwin preprocessing scripts require three cameras, matching the
three cameras in `mujoco_assets/world.xml`:

```python
from mujoco_sim.phystwin_export import export_case

# After collecting frames with DigitalTwinSim:
export_case(sim, frames, "data/different_types/my_sim_case")
```

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
