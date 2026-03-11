#!/usr/bin/env python3
"""Entry point for multi-camera chunk-based depth streaming."""

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from da3_camera_config import (
    CameraConfig,
    ensure_4x4_matrix,
    normalize_extrinsics_scale,
)
from DA3ChunkProcessor import DA3ChunkProcessor
from DA3PoseFinder import DA3PoseFinder
from depth_anything_3.api import DepthAnything3
from depth_inference_classes import ChunkProcessor
from depth_video_renderer import visualize_depth
from DUSt3RPoseFinder import DUSt3RPoseFinder
from tqdm import tqdm


@dataclass
class ChunkConfig:
    chunk_size: int = 60
    overlap: int = 30

    def __post_init__(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be < chunk_size ({self.chunk_size})"
            )


def extract_frame_number(filename: str) -> int:
    numbers = re.findall(r"\d+", filename)
    if not numbers:
        raise ValueError(f"No frame number found in filename: {filename}")
    return int(numbers[0])


class ChunkAligner:
    """Blends overlapping regions between consecutive chunks for temporal consistency."""

    def __init__(self, blend_mode: str = "linear"):
        self.blend_mode = blend_mode
        self.prev_chunk_depths: Dict[int, List[np.ndarray]] = {}

    def align_and_blend(
        self,
        chunk_depths: Dict[int, List[np.ndarray]],
        chunk_start_idx: int,
        overlap_size: int,
        is_first_chunk: bool = False,
    ) -> Dict[int, List[np.ndarray]]:
        if is_first_chunk or not self.prev_chunk_depths:
            self.prev_chunk_depths = {k: v.copy() for k, v in chunk_depths.items()}
            return chunk_depths

        aligned_depths = {}
        for cam_id, curr_depths in chunk_depths.items():
            if cam_id not in self.prev_chunk_depths:
                aligned_depths[cam_id] = curr_depths
                continue

            prev_overlap = self.prev_chunk_depths[cam_id][-overlap_size:]
            curr_overlap = curr_depths[:overlap_size]

            if len(prev_overlap) != overlap_size or len(curr_overlap) != overlap_size:
                print(
                    f"Warning: overlap size mismatch for camera {cam_id} "
                    f"(prev={len(prev_overlap)}, curr={len(curr_overlap)})"
                )
                aligned_depths[cam_id] = curr_depths
                continue

            aligned_depths[cam_id] = (
                self._blend_overlap(prev_overlap, curr_overlap)
                + curr_depths[overlap_size:]
            )

        self.prev_chunk_depths = {k: v.copy() for k, v in aligned_depths.items()}
        return aligned_depths

    def _blend_overlap(
        self, prev: List[np.ndarray], curr: List[np.ndarray]
    ) -> List[np.ndarray]:
        n = len(prev)
        if self.blend_mode == "avg":
            return [0.5 * (prev[i] + curr[i]) for i in range(n)]
        if self.blend_mode == "last":
            return list(curr)
        if self.blend_mode == "linear":
            return [
                ((n - i) / (n + 1)) * prev[i] + ((i + 1) / (n + 1)) * curr[i]
                for i in range(n)
            ]
        raise ValueError(f"Unknown blend mode: {self.blend_mode}")


def load_calibration(
    case_dir: Path,
    calib_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Optional[Tuple[List[CameraConfig], List[int]]]:
    """Load calibration from calibrate.pkl + metadata.json. Returns None if files missing."""
    calib_path = calib_path or case_dir / "calibrate.pkl"
    meta_path = meta_path or case_dir / "metadata.json"

    if not calib_path.exists() or not meta_path.exists():
        return None

    with open(calib_path, "rb") as f:
        c2w_list = pickle.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)

    # Two-tier convention:
    #   case_dir/metadata.json   → K at original image resolution (WH = image dims)
    #   output_dir/metadata.json → K at depth-model resolution   (WH = depth dims)
    stored_wh = metadata.get("WH")
    configs = []
    for cam_id in range(len(c2w_list)):
        K = np.asarray(metadata["intrinsics"][cam_id], dtype=np.float32)

        if stored_wh is not None:
            img_dir = case_dir / "color" / str(cam_id)
            sample_imgs = sorted(img_dir.iterdir()) if img_dir.exists() else []
            if sample_imgs:
                orig_img = cv2.imread(str(sample_imgs[0]))
                if orig_img is not None:
                    orig_h, orig_w = orig_img.shape[:2]
                    depth_w, depth_h = int(stored_wh[0]), int(stored_wh[1])
                    if orig_w != depth_w or orig_h != depth_h:
                        K = K.copy()
                        K[0] *= orig_w / depth_w
                        K[1] *= orig_h / depth_h

        c2w = ensure_4x4_matrix(np.asarray(c2w_list[cam_id], dtype=np.float32))
        configs.append(
            CameraConfig(
                intrinsics=K,
                extrinsics=np.linalg.inv(c2w),
                image_dir=case_dir / "color" / str(cam_id),
            )
        )

    return configs, list(range(len(c2w_list)))


def save_calibration(
    case_dir: Path,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    calib_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    fps: int = 30,
    image_h: Optional[int] = None,
    image_w: Optional[int] = None,
    frame_count: Optional[int] = None,
):
    """Save calibration to calibrate.pkl + metadata.json (preserves existing metadata fields)."""
    calib_path = calib_path or case_dir / "calibrate.pkl"
    meta_path = meta_path or case_dir / "metadata.json"

    print("\n[*] Saving calibration...")

    c2w_list = [np.linalg.inv(ensure_4x4_matrix(w2c)) for w2c in extrinsics]
    with open(calib_path, "wb") as f:
        pickle.dump(c2w_list, f)
    print(f"    ✓ Saved {calib_path}")

    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    metadata.update(
        {
            "intrinsics": [ixt.tolist() for ixt in intrinsics],
            "serial_numbers": list(range(len(intrinsics))),
            "fps": fps,
        }
    )
    if image_h is not None and image_w is not None:
        metadata["WH"] = [image_w, image_h]
    if frame_count is not None:
        metadata["frame_num"] = frame_count

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"    ✓ Saved {meta_path}")


def create_camera_configs_from_calibration(
    case_dir: Path, intrinsics: np.ndarray, extrinsics: np.ndarray
) -> Tuple[List[CameraConfig], List[int]]:
    configs, camera_ids = [], []
    for cam_id in range(len(intrinsics)):
        img_dir = case_dir / "color" / str(cam_id)
        if not img_dir.exists():
            continue
        camera_ids.append(cam_id)
        configs.append(
            CameraConfig(
                intrinsics=intrinsics[cam_id],
                extrinsics=extrinsics[cam_id],
                image_dir=img_dir,
            )
        )
    return configs, camera_ids


def find_synchronized_frames_from_dir(case_dir: Path) -> List[str]:
    """Return filenames present in every camera subdirectory, sorted by frame number."""
    color_dir = case_dir / "color"
    if not color_dir.exists():
        raise FileNotFoundError(f"Color directory not found: {color_dir}")

    camera_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    if not camera_dirs:
        raise ValueError("No camera directories found in color/")

    frame_sets = [
        {
            f.name
            for f in cam_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        for cam_dir in camera_dirs
    ]
    common = set.intersection(*frame_sets)
    if not common:
        raise ValueError("No synchronized frames found across all cameras")

    return sorted(common, key=extract_frame_number)


def visualize_first_frame_3d(
    output_root: Path,
    intrinsics: Optional[np.ndarray] = None,
    extrinsics: Optional[np.ndarray] = None,
    verbose: bool = False,
):
    """Visualize the first saved depth frame in 3D via Open3D (best-effort)."""
    try:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from visualize_3d_scene import visualize_depth_scene

        cam_0_dir = output_root / "0"
        if not cam_0_dir.exists():
            return

        depth_files = sorted(f.name for f in cam_0_dir.iterdir() if f.suffix == ".npy")
        if not depth_files:
            return

        frame_name = Path(depth_files[0]).stem
        if verbose:
            tqdm.write(f"  → Visualizing frame: {frame_name}")

        if intrinsics is not None and extrinsics is not None:
            visualize_depth_scene(
                depth_root=str(output_root),
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                frame_name=frame_name,
                depth_scale=None,
                auto_scale=True,
                show_cameras=True,
                frustum_scale=0.05,
                axis_scale=0.01,
                max_depth=100.0,
            )
        else:
            visualize_depth_scene(
                depth_root=str(output_root),
                calibrate_path=str(output_root / "calibrate.pkl"),
                metadata_path=str(output_root / "metadata.json"),
                frame_name=frame_name,
                depth_scale=None,
                auto_scale=True,
                show_cameras=True,
                frustum_scale=0.05,
                axis_scale=0.01,
                max_depth=100.0,
            )
    except Exception as e:
        tqdm.write(f"  ✗ Error visualizing 3D: {e}")


def save_chunk_results(
    chunk_depths: Dict[int, List[np.ndarray]],
    frame_names: List[str],
    output_root: Path,
    overlap_size: int,
    is_first_chunk: bool,
    visualize: bool = False,
    verbose: bool = False,
):
    """Save depth .npy files, skipping already-saved overlap frames on non-first chunks."""
    start_idx = 0 if is_first_chunk else overlap_size

    if verbose and start_idx < len(frame_names):
        tqdm.write(f"  → Saving frames {frame_names[start_idx]} … {frame_names[-1]}")

    for cam_id, depths in chunk_depths.items():
        save_dir = output_root / str(cam_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i in range(start_idx, len(depths)):
            np.save(save_dir / f"{Path(frame_names[i]).stem}.npy", depths[i])
            if visualize and i == start_idx:
                cv2.imshow(f"Camera {cam_id}", visualize_depth(depths[i]))
                cv2.waitKey(1)


def process_multi_camera_streaming(
    chunk_processor: ChunkProcessor,
    camera_configs: List[CameraConfig],
    camera_ids: List[int],
    frame_names: List[str],
    output_root: Path,
    args: argparse.Namespace,
    aligned_intrinsics: Optional[np.ndarray] = None,
    aligned_extrinsics: Optional[np.ndarray] = None,
):
    chunk_config = ChunkConfig(chunk_size=args.chunk_size, overlap=args.overlap)
    aligner = ChunkAligner(blend_mode=args.blend_mode)
    stride = chunk_config.chunk_size - chunk_config.overlap
    num_frames = len(frame_names)

    if num_frames <= chunk_config.chunk_size:
        num_chunks = 1
    else:
        num_chunks = 1 + (num_frames - chunk_config.chunk_size + stride - 1) // stride

    print(
        f"\nProcessing {num_frames} frames in {num_chunks} chunks "
        f"(size={chunk_config.chunk_size}, overlap={chunk_config.overlap})\n"
    )

    ref_K = camera_configs[0].intrinsics.copy()
    ref_E = camera_configs[0].extrinsics.copy()

    try:
        pbar = tqdm(total=num_chunks, desc="Processing chunks", unit="chunk")
        start_idx, chunk_idx = 0, 0

        while start_idx < num_frames:
            end_idx = min(start_idx + chunk_config.chunk_size, num_frames)
            chunk_frames = frame_names[start_idx:end_idx]
            if not chunk_frames:
                break

            pbar.set_postfix(
                {"frames": f"{start_idx}-{end_idx - 1}", "n": len(chunk_frames)}
            )

            chunk_depths = chunk_processor.process_chunk(
                camera_configs=camera_configs,
                camera_ids=camera_ids,
                frame_names=chunk_frames,
            )

            # Warn if calibration has drifted (should never happen)
            if not np.allclose(camera_configs[0].intrinsics, ref_K, rtol=1e-5):
                tqdm.write(f"  ⚠ WARNING: Intrinsics changed in chunk {chunk_idx}!")
            if not np.allclose(camera_configs[0].extrinsics, ref_E, rtol=1e-5):
                tqdm.write(f"  ⚠ WARNING: Extrinsics changed in chunk {chunk_idx}!")

            is_first = chunk_idx == 0
            ov = chunk_config.overlap if not is_first else 0

            if not is_first and len(chunk_frames) < chunk_config.overlap:
                tqdm.write(
                    f"  Warning: last chunk ({len(chunk_frames)} frames) < overlap {chunk_config.overlap}"
                )
                aligned_depths = chunk_depths
            else:
                aligned_depths = aligner.align_and_blend(
                    chunk_depths, start_idx, ov, is_first_chunk=is_first
                )

            save_chunk_results(
                aligned_depths,
                chunk_frames,
                output_root,
                ov,
                is_first,
                args.visualize,
                args.verbose,
            )

            if is_first and args.visualize_3d:
                visualize_first_frame_3d(
                    output_root, aligned_intrinsics, aligned_extrinsics, args.verbose
                )

            start_idx += stride
            chunk_idx += 1
            pbar.update(1)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-camera chunk-based streaming depth estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case_dir", type=Path, required=True)
    parser.add_argument(
        "--output_root", type=Path, default=Path("parsed_depth_DA3_streaming")
    )
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument(
        "--blend_mode", choices=["linear", "avg", "last"], default="linear"
    )
    parser.add_argument(
        "--calib_frames",
        type=int,
        default=3,
        help="Frames to average for calibration inference (no calibrate.pkl only)",
    )
    parser.add_argument(
        "--camera_baseline",
        type=float,
        default=1.0,
        help="Target max camera separation (m) when inferring calibration via DUSt3R",
    )
    parser.add_argument("--model", default="DA3")
    parser.add_argument(
        "--pose_calibration_model", default="DUSt3R", choices=["DUSt3R", "DA3"]
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_3d", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not args.case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {args.case_dir}")

    output_dir = args.output_root / args.case_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Case:    {args.case_dir.name}  →  {output_dir}")
    print(f"Model:   {args.model}   Device: {args.device}")
    print(
        f"Chunks:  size={args.chunk_size}  overlap={args.overlap}  blend={args.blend_mode}"
    )
    print("=" * 60)

    print("\n[1/5] Finding synchronized frames...")
    frame_names = find_synchronized_frames_from_dir(args.case_dir)
    print(f"      {len(frame_names)} frames found")

    print(f"\n[2/5] Loading model ({args.model})...")
    if args.model == "DA3":
        depth_model = (
            DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
            .to(args.device)
            .eval()
        )
        depth_pose_finder = DA3PoseFinder(depth_model)
        chunk_processor = DA3ChunkProcessor(depth_model)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print("\n[3/5] Loading/inferring calibration...")
    calib_result = load_calibration(args.case_dir, args.calibration, args.metadata)
    inferred = False

    if calib_result is not None:
        camera_configs, camera_ids = calib_result
        print(f"      ✓ Loaded calibration ({len(camera_configs)} cameras)")
    else:
        print(
            f"      ! No calibration found — inferring via {args.pose_calibration_model}..."
        )
        if args.pose_calibration_model == "DUSt3R":
            calib_finder = DUSt3RPoseFinder()
        else:
            calib_finder = depth_pose_finder

        intrinsics, extrinsics = calib_finder.infer_calibration(
            args.case_dir, frame_names, args.calib_frames
        )
        if args.pose_calibration_model == "DUSt3R":
            del calib_finder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        extrinsics, _ = normalize_extrinsics_scale(extrinsics, args.camera_baseline)
        camera_configs, camera_ids = create_camera_configs_from_calibration(
            args.case_dir, intrinsics, extrinsics
        )
        inferred = True

    print("\n[4/5] Determining depth-model output resolution...")
    out_extrinsics = np.stack([cfg.extrinsics for cfg in camera_configs])
    aligned_intrinsics, depth_hw = depth_pose_finder.align_intrinsics(
        camera_configs, camera_ids, frame_names
    )
    if aligned_intrinsics is None:
        raise RuntimeError("Depth model returned no intrinsics — cannot continue")

    if depth_hw:
        print(f"      ✓ Depth resolution: {depth_hw[1]}×{depth_hw[0]}")
        print(
            f"      ✓ fx: {camera_configs[0].intrinsics[0, 0]:.1f} (image) "
            f"→ {aligned_intrinsics[0, 0, 0]:.1f} (depth)"
        )

    orig_intrinsics = np.stack([cfg.intrinsics for cfg in camera_configs])
    sample_img = cv2.imread(str(camera_configs[0].image_dir / frame_names[0]))
    img_h, img_w = sample_img.shape[:2] if sample_img is not None else (None, None)

    if inferred:
        save_calibration(
            args.case_dir,
            orig_intrinsics,
            out_extrinsics,
            fps=args.fps,
            image_h=img_h,
            image_w=img_w,
            frame_count=len(frame_names),
        )

    save_calibration(
        output_dir,
        orig_intrinsics,
        out_extrinsics,
        fps=args.fps,
        image_h=img_h,
        image_w=img_w,
        frame_count=len(frame_names),
    )

    print("\n[5/5] Processing chunks...")
    process_multi_camera_streaming(
        chunk_processor,
        camera_configs,
        camera_ids,
        frame_names,
        output_dir,
        args,
        aligned_intrinsics,
        out_extrinsics,
    )

    print("\n" + "=" * 60)
    print("✓ Complete!")
    print(f"✓ Output:      {output_dir}")
    print(f"✓ Calibration: {output_dir / 'calibrate.pkl'}")
    if inferred:
        print("  (calibration also saved to case_dir for future use)")
    print("=" * 60)


if __name__ == "__main__":
    main()
