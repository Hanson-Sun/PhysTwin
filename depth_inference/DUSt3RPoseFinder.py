from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from depth_inference_classes import PoseFinder
from PIL import Image


class DUSt3RPoseFinder(PoseFinder):
    """PoseFinder implementation for DUSt3R (model loaded lazily)."""

    def _rescale_intrinsics(
        self, K: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]
    ) -> np.ndarray:
        """Rescale K from src resolution to dst resolution."""
        K = K.copy()
        K[0, 0] *= dst_hw[1] / src_hw[1]  # fx
        K[1, 1] *= dst_hw[0] / src_hw[0]  # fy
        K[0, 2] *= dst_hw[1] / src_hw[1]  # cx
        K[1, 2] *= dst_hw[0] / src_hw[0]  # cy
        return K

    def _get_frame_paths(self, case_dir: Path, frame: int) -> List[str]:
        """Return sorted image paths for each camera at the given frame index."""
        color_dir = case_dir / "color"
        cam_dirs = sorted(
            [d for d in color_dir.iterdir() if d.is_dir()],
            key=lambda d: int(d.name),
        )
        paths = []
        for cam_dir in cam_dirs:
            img_path = cam_dir / f"{frame}.png"
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Frame {frame} not found for cam {cam_dir.name}: {img_path}"
                )
            paths.append(str(img_path))
        return paths

    def infer_calibration(
        self,
        case_dir: Path,
        frame_names: List[str],
        calib_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        sys.path.insert(0, str(Path(__file__).parent.parent / "dust3r"))

        from da3_camera_config import average_extrinsics, average_intrinsics
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.device import to_numpy
        from dust3r.utils.image import load_images

        num_calib_frames = max(1, min(calib_frames, len(frame_names)))
        print(
            f"\n[*] DUSt3R calibration — averaging over {num_calib_frames} frame(s)..."
        )

        if num_calib_frames == 1:
            sample_indices = [0]
        else:
            sample_indices = [
                int(round(i * (len(frame_names) - 1) / (num_calib_frames - 1)))
                for i in range(num_calib_frames)
            ]

        sampled_frames = [frame_names[i] for i in sample_indices]
        print(f"    sampled frames: {sampled_frames}")

        # Parse frame indices from filenames (e.g. "0042.png" → 42)
        frame_indices = []
        for fname in sampled_frames:
            stem = Path(fname).stem.lstrip("0") or "0"
            try:
                frame_indices.append(int(stem))
            except ValueError:
                print(f"    ⚠ Could not parse frame index from {fname}, using 0")
                frame_indices.append(0)

        # Lazily load model
        if self.model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = (
                AsymmetricCroCo3DStereo.from_pretrained(
                    "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
                )
                .to(device)
                .eval()
            )

        device = str(next(self.model.parameters()).device)
        first_paths = self._get_frame_paths(case_dir, frame_indices[0])
        n_cams = len(first_paths)
        W_orig, H_orig = Image.open(first_paths[0]).size
        print(f"    {n_cams} cameras · frame indices: {frame_indices}")

        all_Ks: List[np.ndarray] = []
        all_c2ws: List[np.ndarray] = []

        for frame_idx in frame_indices:
            img_paths = self._get_frame_paths(case_dir, frame_idx)
            print(f"\n    Processing frame {frame_idx}...")

            imgs = load_images(img_paths, size=512, verbose=False)
            pairs = make_pairs(
                imgs, scene_graph="complete", prefilter=None, symmetrize=True
            )
            print(f"      Inference on {len(pairs)} pairs...")
            output = inference(pairs, self.model, device, batch_size=1, verbose=False)

            print("      Global alignment (300 iterations)...")
            mode = (
                GlobalAlignerMode.PointCloudOptimizer
                if n_cams > 2
                else GlobalAlignerMode.PairViewer
            )
            scene = global_aligner(output, device=device, mode=mode, verbose=False)
            if mode == GlobalAlignerMode.PointCloudOptimizer:
                scene.compute_global_alignment(
                    init="mst", niter=300, schedule="cosine", lr=0.01
                )

            K_dust3r = to_numpy(scene.get_intrinsics())  # (N, 3, 3)
            c2ws_dust = to_numpy(scene.get_im_poses())  # (N, 4, 4)

            # true_shape: (1, 2) tensor with [H, W] at DUSt3R's inference resolution
            dust3r_H, dust3r_W = (int(x) for x in imgs[0]["true_shape"][0])

            all_Ks.append(
                np.stack(
                    [
                        self._rescale_intrinsics(
                            K_dust3r[i],
                            (dust3r_H, dust3r_W),
                            (H_orig, W_orig),
                        )
                        for i in range(n_cams)
                    ]
                ).astype(np.float32)
            )
            all_c2ws.append(c2ws_dust.astype(np.float32))

        if not all_Ks:
            raise RuntimeError("DUSt3R failed to process all frames")

        K_stack = np.stack(all_Ks)  # (S, N, 3, 3)
        c2w_stack = np.stack(all_c2ws)  # (S, N, 4, 4)

        intrinsics_array = np.stack(
            [average_intrinsics(K_stack[:, i]) for i in range(n_cams)]
        ).astype(np.float32)

        c2ws_averaged = np.stack(
            [average_extrinsics(c2w_stack[:, i]) for i in range(n_cams)]
        ).astype(np.float32)

        print(f"\n    ✓ Averaged over {len(all_Ks)} frame(s)")

        # DUSt3R returns cam-to-world; invert to world-to-camera
        return intrinsics_array, np.linalg.inv(c2ws_averaged)

    def align_intrinsics(
        self,
        camera_configs,
        camera_ids,
        frame_names: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """DUSt3R does not expose a depth-resolution intrinsics processor."""
        return None, None
