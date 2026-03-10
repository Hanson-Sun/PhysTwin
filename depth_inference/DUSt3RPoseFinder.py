from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import PIL

from depth_inference_classes import PoseFinder


class DUSt3RPoseFinder(PoseFinder):
    """PoseFinder implementation for DUSt3R model.

    model may be None; if so, DUSt3R is loaded lazily on first call.
    """

    def _rescale_intrinsics(self, K_dust3r, dust3r_hw, orig_hw):
        """Rescale K from DUSt3R's resolution back to original resolution."""
        H_d, W_d = dust3r_hw
        H_o, W_o = orig_hw
        sx = W_o / W_d
        sy = H_o / H_d
        K = K_dust3r.copy()
        K[0, 0] *= sx      # fx
        K[1, 1] *= sy      # fy
        K[0, 2] *= sx      # cx
        K[1, 2] *= sy      # cy
        return K

    def _get_frame_paths(self, case_dir: str, frame: int) -> List[str]:
        """Return sorted image paths for each camera at given frame."""
        from pathlib import Path as PathlibPath
        color_dir = PathlibPath(case_dir) / "color"
        cam_dirs = sorted(
            [d for d in color_dir.iterdir() if d.is_dir()],
            key=lambda d: int(d.name)
        )
        paths = []
        for cam_dir in cam_dirs:
            img_path = cam_dir / f"{frame}.png"
            if not img_path.exists():
                raise FileNotFoundError(f"Frame {frame} not found for cam {cam_dir.name}: {img_path}")
            paths.append(str(img_path))
        return paths

    def infer_calibration(
        self,
        case_dir: Path,
        frame_names: List[str],
        calib_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        _sys.path.insert(0, str(Path(__file__).parent.parent / "dust3r"))
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        from dust3r.utils.device import to_numpy
        from da3_camera_config import average_intrinsics, average_extrinsics

        # Sample frames similar to DA3PoseFinder
        num_calib_frames = max(1, min(calib_frames, len(frame_names)))
        print(f"\n[*] Inferring camera calibration with DUSt3R by averaging over {num_calib_frames} frame(s)...")
        
        if num_calib_frames == 1:
            sample_indices = [0]
        else:
            sample_indices = [
                int(round(i * (len(frame_names) - 1) / (num_calib_frames - 1)))
                for i in range(num_calib_frames)
            ]
        
        sampled_frames = [frame_names[i] for i in sample_indices]
        print(f"    sampled frames: {sampled_frames}")
        
        # Convert frame names to frame indices
        frame_indices = []
        for fname in sampled_frames:
            try:
                frame_idx = int(fname.replace(".png", "").lstrip("0") or "0")
                frame_indices.append(frame_idx)
            except ValueError:
                print(f"    ⚠ Could not parse frame index from {fname}, using 0")
                frame_indices.append(0)
        
        # Load or initialize model
        model = self.model
        if model is None:
            model = AsymmetricCroCo3DStereo.from_pretrained(
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
            self.model = model

        # Get device and camera count
        device = str(next(model.parameters()).device)
        img_paths_first = self._get_frame_paths(str(case_dir), frame_indices[0])
        n_cams = len(img_paths_first)
        W_orig, H_orig = PIL.Image.open(img_paths_first[0]).size
        print(f"    {n_cams} cameras · frame indices: {frame_indices}")

        all_Ks = []      # Collect (n_cams, 3, 3) for each frame
        all_c2ws = []    # Collect (n_cams, 4, 4) for each frame

        # Process each frame
        for frame_idx in frame_indices:
            img_paths = self._get_frame_paths(str(case_dir), frame_idx)
            print(f"\n    Processing frame {frame_idx}...")

            imgs = load_images(img_paths, size=512, verbose=False)
            
            pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
            print(f"      Inference on {len(pairs)} pairs...")
            output = inference(pairs, model, device, batch_size=1, verbose=False)

            print(f"      Global alignment (300 iterations)...")
            mode = GlobalAlignerMode.PointCloudOptimizer if n_cams > 2 else GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=device, mode=mode, verbose=False)

            if mode == GlobalAlignerMode.PointCloudOptimizer:
                scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

            K_dust3r = to_numpy(scene.get_intrinsics())    # (N, 3, 3)
            c2ws_dust = to_numpy(scene.get_im_poses())      # (N, 4, 4)

            dust3r_H = imgs[0]["true_shape"][0][0].item() if hasattr(imgs[0]["true_shape"][0], "item") else int(imgs[0]["true_shape"][0][0])
            dust3r_W = imgs[0]["true_shape"][0][1].item() if hasattr(imgs[0]["true_shape"][0], "item") else int(imgs[0]["true_shape"][0][1])

            # Rescale to original resolution
            Ks_frame = np.stack([
                self._rescale_intrinsics(K_dust3r[i], (dust3r_H, dust3r_W), (H_orig, W_orig))
                for i in range(n_cams)
            ]).astype(np.float32)
            
            c2ws_frame = c2ws_dust.astype(np.float32)

            all_Ks.append(Ks_frame)
            all_c2ws.append(c2ws_frame)

        # Average across frames
        if len(all_Ks) == 0:
            raise RuntimeError("DUSt3R failed to process all frames")
        
        K_stack = np.stack(all_Ks, axis=0)      # (num_frames, n_cams, 3, 3)
        c2w_stack = np.stack(all_c2ws, axis=0)  # (num_frames, n_cams, 4, 4)

        intrinsics_array = np.stack([
            average_intrinsics(K_stack[:, cam_id])
            for cam_id in range(n_cams)
        ]).astype(np.float32)
        
        c2ws_averaged = np.stack([
            average_extrinsics(c2w_stack[:, cam_id])
            for cam_id in range(n_cams)
        ]).astype(np.float32)
        
        print(f"\n    ✓ Averaged over {len(all_Ks)} frame(s)")
        
        # DUSt3R returns cam-to-world; invert to world-to-camera
        extrinsics_array = np.linalg.inv(c2ws_averaged)
        
        return intrinsics_array, extrinsics_array