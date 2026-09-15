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
`data/different_types/my_case/`.

## Train Warp Parameters

```bash
python scripts/train_warp_case.py \
	--base_path data/different_types \
	--case_name my_case
```

This runs CMA initialization followed by differentiable warp training. The best
weights are saved under `experiments/my_case/train/best_*.pth`, with a manifest
at `experiments/my_case/train/training_manifest.json`. Use `--skip_cma` to skip
the initialization stage. Use `--iterations 1` for a quick smoke test.

## Reconstruct Gaussians

Run this once after RGB-D processing and before RL:

```bash
python scripts/reconstruct_gaussians.py \
	--base_path data/different_types \
	--case_name my_case \
	--output_dir gaussian_output/my_case
```

This prepares the first-frame Gaussian dataset and trains the static appearance
model. The resulting model is written to
`gaussian_output/my_case/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply`.
The command also creates the pruned Boba asset under
`gaussian_output_pruned_policy_30_55/my_case/` by default. During RL, use the
pruned asset with Boba's `batch_prune` renderer and reuse both Gaussian assets
and the trained warp weights; do not reconstruct them for every episode.

## Run Everything

For a case listed in `data_config.csv`, the category and shape-prior setting are
selected automatically:

```bash
python scripts/run_case_pipeline.py \
	--case_name double_lift_sloth
```

This runs RGB-D processing, CMA and warp training, Gaussian reconstruction, and
Gaussian pruning in order. Use `--dry_run` to inspect the resolved settings,
`--smoke_test` for a minimal run, or `--skip_process`, `--skip_warp`, and
`--skip_gaussians` to resume from an existing stage.


## Env modifications

1. `pip install mujoco`
2. `pip install gymnasium`