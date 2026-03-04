#!/usr/bin/env python3
"""
Multi-Camera Depth Estimation with Pi3X Chunk-Based Streaming

Uses Pi3X — the multimodal-conditioned variant of Pi3 — which accepts
calibrated camera intrinsics and extrinsics (poses) as conditioning inputs.
This grounds the depth estimation in the known camera geometry so that:
  * XY rays per pixel follow the real lens model (not Pi3's guessed focal).
  * All cameras and frames share the same metric, calibrated world frame.
  * Cross-camera and cross-chunk depth scales are consistent.

Reuses the DA3 streaming scaffolding (ChunkConfig, ChunkAligner, calibration
I/O, save helpers, etc.) unchanged.
"""
import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Shared scaffolding — imported from the DA3 streaming pipeline
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from parse_depth_da3_streaming_final import (
    ChunkConfig,
    ChunkAligner,
    load_calibration,
    save_calibration,
    create_camera_configs_from_calibration,
    find_synchronized_frames_from_dir,
    save_chunk_results,
)
from da3_camera_config import CameraConfig, ensure_4x4_matrix

# Pi3X model (repository must be on PYTHONPATH or adjacent as Pi3/)
sys.path.insert(0, str(Path(__file__).parent.parent / "Pi3"))
from pi3.models.pi3x import Pi3X




_PIXEL_LIMIT = 255_000  # Per-frame pixel budget (same as Pi3's own default)

def _load_images_for_pi3x(
    image_paths: List[str],
    device: torch.device,
    pixel_limit: int = _PIXEL_LIMIT,
):
    """Load images, resize to the largest multiple-of-14 within *pixel_limit*.
    """
    to_tensor = transforms.ToTensor()

    # Compute uniform target size from the first image
    first = PILImage.open(image_paths[0]).convert("RGB")
    W_orig, H_orig = first.size
    scale = math.sqrt(pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1.0
    k = round(W_orig * scale / 14)
    m = round(H_orig * scale / 14)
    while (k * 14) * (m * 14) > pixel_limit:
        if k / m > W_orig / H_orig:
            k -= 1
        else:
            m -= 1
    inf_W, inf_H = max(1, k) * 14, max(1, m) * 14

    tensors = []
    for path in image_paths:
        img = PILImage.open(path).convert("RGB")
        img = img.resize((inf_W, inf_H), PILImage.Resampling.LANCZOS)
        tensors.append(to_tensor(img))

    return torch.stack(tensors).to(device), inf_H, inf_W  # (N, 3, H', W')


# ---------------------------------------------------------------------------
# Core inference — Pi3X conditioned on calibrated intrinsics + poses
# ---------------------------------------------------------------------------
def process_chunk(
    model: Pi3X,
    camera_configs: List[CameraConfig],
    camera_ids: List[int],
    frame_names: List[str],
):
    """Run Pi3X depth inference conditioned on calibrated intrinsics and poses.
    """
    num_cams = len(camera_ids)
    num_frames = len(frame_names)
    N_total = num_frames * num_cams

    # Build flat image list: frame-major, camera-minor
    all_image_paths = [
        str(config.image_dir / frame_name)
        for frame_name in frame_names
        for config in camera_configs
    ]

    # Load images and record both resolutions
    device = next(model.parameters()).device
    imgs, inf_H, inf_W = _load_images_for_pi3x(all_image_paths, device)  # (N_total, 3, H', W')

    # Get original image size for resizing depth back
    first_img = PILImage.open(all_image_paths[0])
    orig_W, orig_H = first_img.size  # PIL gives (W, H)

    # ------------------------------------------------------------------
    # Build intrinsics tensor — scaled from original → inference resolution
    # Pi3X expects pixel-coordinate K at the actual input image resolution.
    # Layout mirrors image order: frame-major, camera-minor.
    # ------------------------------------------------------------------
    scale_x = inf_W / orig_W
    scale_y = inf_H / orig_H
    K_scaled_list = []
    for _ in range(num_frames):
        for cfg in camera_configs:
            K = cfg.intrinsics.copy().astype(np.float32)  # (3, 3)
            K[0] *= scale_x  # fx, cx
            K[1] *= scale_y  # fy, cy
            K_scaled_list.append(K)

    # (1, N_total, 3, 3)
    intrinsics_t = torch.from_numpy(np.stack(K_scaled_list)).float().to(device)[None]

    # ------------------------------------------------------------------
    # Build poses tensor — c2w in OpenCV convention (Right-Down-Forward)
    # Our CameraConfig stores w2c (extrinsics), so we invert.
    # Same c2w is reused for every frame of the same camera (fixed rig).
    # ------------------------------------------------------------------
    c2w_list = []
    for _ in range(num_frames):
        for cfg in camera_configs:
            w2c = ensure_4x4_matrix(cfg.extrinsics).astype(np.float32)
            c2w = np.linalg.inv(w2c)
            c2w_list.append(c2w)

    # (1, N_total, 4, 4)
    poses_t = torch.from_numpy(np.stack(c2w_list)).float().to(device)[None]

    # ------------------------------------------------------------------
    # All conditioning masks → True (always condition on rays + poses)
    # ------------------------------------------------------------------
    mask_ray   = torch.ones(1, N_total, dtype=torch.bool, device=device)
    mask_pose  = torch.ones(1, N_total, dtype=torch.bool, device=device)
    mask_depth = torch.zeros(1, N_total, dtype=torch.bool, device=device)  # no prior depths

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            res = model(
                imgs=imgs[None],          # (1, N_total, 3, H', W')
                intrinsics=intrinsics_t,  # (1, N_total, 3, 3)
                poses=poses_t,            # (1, N_total, 4, 4)
                mask_add_ray=mask_ray,
                mask_add_pose=mask_pose,
                mask_add_depth=mask_depth,
            )

    # ------------------------------------------------------------------
    # Z-only depth maps — resize to original resolution for saving
    # local_points[..., 2] is camera-space Z (metric, conditioned scale)
    # ------------------------------------------------------------------
    local_pts_np = res["local_points"][0].float().cpu().numpy()  # (N_total, H', W', 3)

    chunk_depths_z: Dict[int, List[np.ndarray]] = {cam_id: [] for cam_id in camera_ids}
    for frame_idx in range(num_frames):
        for cam_idx, cam_id in enumerate(camera_ids):
            flat_idx = frame_idx * num_cams + cam_idx
            z = local_pts_np[flat_idx, :, :, 2]  # (H', W')
            if z.shape != (orig_H, orig_W):
                z = cv2.resize(z, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
            chunk_depths_z[cam_id].append(z)

    # ------------------------------------------------------------------
    # World-space points — kept at inference resolution for visualisation
    # res['points'] is metric and in the calibrated world frame
    # (relative to poses[:, 0] = frame0_cam0)
    # ------------------------------------------------------------------
    world_pts_np = res["points"][0].float().cpu().numpy()  # (N_total, H', W', 3)

    chunk_world_pts: Dict[int, List[np.ndarray]] = {cam_id: [] for cam_id in camera_ids}
    for frame_idx in range(num_frames):
        for cam_idx, cam_id in enumerate(camera_ids):
            flat_idx = frame_idx * num_cams + cam_idx
            chunk_world_pts[cam_id].append(world_pts_np[flat_idx])  # (H', W', 3)

    # Pi3X-predicted camera poses for all views (used for frustum visualisation)
    chunk_cam_poses = res["camera_poses"][0].float().cpu().numpy()  # (N_total, 4, 4)

    return chunk_depths_z, chunk_world_pts, chunk_cam_poses


def process_multi_camera_streaming(
    model: Pi3X,
    camera_configs: List[CameraConfig],
    camera_ids: List[int],
    frame_names: List[str],
    output_root: Path,
    args: argparse.Namespace,
) -> None:
    """Process the full video in overlapping chunks with temporal blending.

    Mirrors the DA3 streaming orchestrator but omits calibration-consistency
    checks that are only meaningful when the model consumes camera parameters.
    """
    chunk_config = ChunkConfig(chunk_size=args.chunk_size, overlap=args.overlap)
    aligner = ChunkAligner(blend_mode=args.blend_mode)

    num_frames = len(frame_names)
    stride = chunk_config.chunk_size - chunk_config.overlap

    if num_frames <= chunk_config.chunk_size:
        num_chunks = 1
    else:
        remaining = num_frames - chunk_config.chunk_size
        num_chunks = 1 + (remaining + stride - 1) // stride

    print(f"\nProcessing {num_frames} frames in {num_chunks} chunks")
    print(f"Chunk size: {chunk_config.chunk_size}, Overlap: {chunk_config.overlap}, Stride: {stride}")
    print("")

    try:
        chunk_idx = 0
        start_idx = 0
        pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk")

        while start_idx < num_frames:
            end_idx = min(start_idx + chunk_config.chunk_size, num_frames)
            chunk_frames = frame_names[start_idx:end_idx]
            if not chunk_frames:
                break

            pbar.set_postfix({"frames": f"{start_idx}-{end_idx - 1}", "count": len(chunk_frames)})

            chunk_depths_z, chunk_world_pts, chunk_cam_poses = process_chunk(
                model=model,
                camera_configs=camera_configs,
                camera_ids=camera_ids,
                frame_names=chunk_frames,
            )

            is_first = chunk_idx == 0

            if not is_first and len(chunk_frames) < chunk_config.overlap:
                tqdm.write(
                    f"  Warning: last chunk ({len(chunk_frames)} frames) < overlap "
                    f"({chunk_config.overlap}) — skipping blend"
                )
                aligned_depths = chunk_depths_z
            else:
                aligned_depths = aligner.align_and_blend(
                    chunk_depths=chunk_depths_z,
                    chunk_start_idx=start_idx,
                    overlap_size=chunk_config.overlap if not is_first else 0,
                    is_first_chunk=is_first,
                )

            save_chunk_results(
                chunk_depths=aligned_depths,
                frame_names=chunk_frames,
                output_root=output_root,
                overlap_size=chunk_config.overlap if not is_first else 0,
                is_first_chunk=is_first,
                visualize=args.visualize,
                verbose=args.verbose,
            )

            if is_first and args.visualize_3d:
                _visualize_first_frame_world(
                    chunk_world_pts=chunk_world_pts,
                    chunk_cam_poses=chunk_cam_poses,
                    camera_ids=camera_ids,
                    num_cams=len(camera_ids),
                    verbose=args.verbose,
                )

            start_idx += stride
            chunk_idx += 1
            pbar.update(1)
            torch.cuda.empty_cache()

        pbar.close()
    finally:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-camera chunk-based streaming depth estimation with Pi3X",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--case_dir", type=Path, required=True,
                        help="Case directory containing color/ subdirectory")
    parser.add_argument("--output_root", type=Path, default=Path("parsed_depth_pi3"),
                        help="Root output directory")

    # Calibration (optional — not used by Pi3 for inference, but copied to
    # the output directory so downstream consumers remain compatible)
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Path to calibrate.pkl (defaults to case_dir/calibrate.pkl)")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Path to metadata.json (defaults to case_dir/metadata.json)")

    # Model
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Pi3X checkpoint (.safetensors or .pt). "
                             "Omit to download the pretrained model from HuggingFace.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Chunk configuration
    parser.add_argument("--chunk_size", type=int, default=3,
                        help="Number of frames per chunk")
    parser.add_argument("--overlap", type=int, default=2,
                        help="Overlapping frames between consecutive chunks")
    parser.add_argument("--blend_mode", choices=["linear", "avg", "last"], default="linear",
                        help="Temporal blending mode for overlapping frames")

    # Misc
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--visualize", action="store_true",
                        help="Show depth colour maps during processing")
    parser.add_argument("--visualize_3d", action="store_true",
                        help="Visualize first-frame depth in 3-D (requires Open3D)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {args.case_dir}")

    case_name = args.case_dir.name
    output_dir = args.output_root / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 80)
    print("Multi-Camera Pi3X Chunk-Based Streaming Depth Estimation")
    print("=" * 80)
    print(f"Case:       {case_name}")
    print(f"Output:     {output_dir}")
    print(f"Device:     {args.device}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Overlap:    {args.overlap}")
    print(f"Blend mode: {args.blend_mode}")
    print("=" * 80)

    print("\n[1/3] Finding synchronized frames...")
    frame_names = find_synchronized_frames_from_dir(args.case_dir)
    print(f"      {len(frame_names)} frames found")

    print(f"\n[2/3] Loading Pi3X model...")
    torch.set_grad_enabled(False)
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()

    print("\n[3/3] Loading calibration & processing chunks...")
    calib_result = load_calibration(args.case_dir, args.calibration, args.metadata)

    if calib_result is not None:
        camera_configs, camera_ids = calib_result
        print(f"      ✓ Loaded calibration ({len(camera_configs)} cameras)")

        # Copy calibration to output directory.
        # cfg.intrinsics from load_calibration is already at original image resolution.
        out_intrinsics = np.stack([cfg.intrinsics for cfg in camera_configs], axis=0)
        out_extrinsics = np.stack([cfg.extrinsics for cfg in camera_configs], axis=0)
        sample_img = cv2.imread(str(camera_configs[0].image_dir / frame_names[0]))
        img_h, img_w = sample_img.shape[:2] if sample_img is not None else (None, None)
        save_calibration(
            output_dir, out_intrinsics, out_extrinsics,
            fps=args.fps, frame_count=len(frame_names),
            image_h=img_h, image_w=img_w,
        )
    else:
        print("      ! No calibration found — using identity stubs (depth will not be metric)")
        color_dir = args.case_dir / "color"
        cam_dirs = sorted(d for d in color_dir.iterdir() if d.is_dir())
        camera_configs = [
            CameraConfig(
                intrinsics=np.eye(3, dtype=np.float32),
                extrinsics=np.eye(4, dtype=np.float32),
                image_dir=cam_dir,
            )
            for cam_dir in cam_dirs
        ]
        camera_ids = list(range(len(camera_configs)))

    process_multi_camera_streaming(
        model=model,
        camera_configs=camera_configs,
        camera_ids=camera_ids,
        frame_names=frame_names,
        output_root=output_dir,
        args=args,
    )

    print("\n" + "=" * 80)
    print("✓ Complete!")
    print(f"✓ Output: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
