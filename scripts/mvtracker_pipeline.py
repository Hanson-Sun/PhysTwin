#!/usr/bin/env python3
"""
MVTracker 3D point tracking pipeline with Rerun visualization.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MVTrackerPipeline:
    """3D point tracking and visualization pipeline."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str = "./mvtracker_results",
        device: str = "cuda",
        resolution: Optional[Tuple[int, int]] = None,
        frame_skip: int = 1,
    ):
        """
        Initialize pipeline with memory optimization options.

        Args:
            data_dir: Dataset directory with color/ subdirectory
            output_dir: Output directory for results
            device: "cuda" or "cpu"
            resolution: Target resolution (H, W) for downsampling. If None, use original.
            frame_skip: Load every Nth frame (e.g., 2 = every 2nd frame).
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.resolution = resolution
        self.frame_skip = frame_skip
        
        if frame_skip > 1:
            logger.info(f"Frame skipping enabled: loading every {frame_skip}th frame")
        if resolution:
            logger.info(f"Resolution downsampling enabled: target {resolution}")

    def _load_video_frames(self, num_views: int = 3) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Load RGB video frames from color/ directory."""
        color_dir = self.data_dir / "color"
        all_frames = []
        
        for view_id in range(num_views):
            video_path = color_dir / f"{view_id}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            
            frames = np.stack(frames)  # (T, H, W, 3)
            
            # Apply frame skipping
            if self.frame_skip > 1:
                frames = frames[::self.frame_skip]
            
            # Apply downsampling
            if self.resolution:
                h, w = self.resolution
                frames = np.stack([
                    cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
                    for f in frames
                ])
            
            all_frames.append(frames)
            logger.info(f"View {view_id}: {frames.shape}")
        
        # Sync frame counts
        min_frames = min(f.shape[0] for f in all_frames)
        all_frames = [f[:min_frames] for f in all_frames]
        
        rgbs = np.stack(all_frames)  # (V, T, H, W, 3)
        H, W = rgbs.shape[2:4]
        logger.info(f"Loaded {rgbs.shape[0]} views, {rgbs.shape[1]} frames, {H}x{W}")
        return rgbs, (H, W)

    def _get_default_cameras(self, num_views: int, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create default camera parameters."""
        focal = max(H, W)
        intrs = np.array([
            [[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]]
            for _ in range(num_views)
        ], dtype=np.float32)
        
        extrs = []
        for v in range(num_views):
            angle = 2 * np.pi * v / num_views
            extr = np.eye(4, dtype=np.float32)
            extr[0, 3] = np.cos(angle)
            extr[1, 3] = np.sin(angle)
            extr[2, 3] = 0.5
            extrs.append(extr[:3])
        
        return intrs, np.array(extrs, dtype=np.float32)

    @torch.no_grad()
    def track(
        self,
        query_points: Optional[np.ndarray] = None,
        num_views: int = 3,
        iters: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Track 3D points across views using MVTracker.
        
        Args:
            query_points: (N, 3) 3D query point coordinates. If None, use defaults.
            num_views: Number of camera views
            iters: Refinement iterations
            
        Returns:
            traj: (T, N, 3) tracked point trajectories
            vis: (T, N) visibility confidence scores
            metadata: Dictionary with scene metadata
        """
        
        # Load videos
        rgbs, (H, W) = self._load_video_frames(num_views)
        T = rgbs.shape[1]

        # Get cameras
        intrs, extrs = self._get_default_cameras(num_views, H, W)
        intrs = np.tile(intrs[:, None], (1, T, 1, 1))
        extrs = np.tile(extrs[:, None], (1, T, 1, 1))

        # Query points (4 values: id, x, y, z)
        if query_points is None:
            query_points = np.array([
                [0, 0.0, 0.0, 0.0],
                [0, 0.1, 0.0, 0.0],
                [0, 0.0, 0.1, 0.0],
            ], dtype=np.float32)

        # Zero depths (MVTracker handles this)
        depths = np.zeros((num_views, T, 1, H, W), dtype=np.float32)

        # Convert to tensors
        rgbs_t = torch.from_numpy(rgbs).float().permute(0, 1, 4, 2, 3)[None] / 255.0
        depths_t = torch.from_numpy(depths).float()[None]
        intrs_t = torch.from_numpy(intrs).float()[None]
        extrs_t = torch.from_numpy(extrs).float()[None]
        query_t = torch.from_numpy(query_points).float()[None]

        # Move to device
        rgbs_t, depths_t, intrs_t, extrs_t, query_t = [
            x.to(self.device) for x in [rgbs_t, depths_t, intrs_t, extrs_t, query_t]
        ]

        logger.info("Loading MVTracker...")
        model = torch.hub.load(
            "ethz-vlg/mvtracker",
            "mvtracker_predictor",
            pretrained=True,
            device=self.device,
            predictor_kwargs={"n_iters": iters},
        )
        model.eval()

        logger.info(f"Running MVTracker (iters={iters})...")
        results = model(
            rgbs=rgbs_t,
            depths=depths_t,
            intrs=intrs_t,
            extrs=extrs_t,
            query_points_3d=query_t,
        )

        traj = results["traj_e"][0].cpu().numpy()
        vis = results["vis_e"][0].cpu().numpy()

        logger.info(f"Tracking complete: {traj.shape} (frames, points, xyz)")

        metadata = {
            'intrinsics': intrs.tolist(),
            'extrinsics': extrs.tolist(),
            'query_points': query_points.tolist(),
            'num_frames': int(T),
            'num_views': int(num_views),
            'height': int(H),
            'width': int(W),
        }

        return traj, vis, metadata

    def visualize_3d_rerun(
        self,
        rgbs: np.ndarray,
        trajectories: np.ndarray,
        visibilities: np.ndarray,
        metadata: Dict,
        output_path: Optional[str] = None,
        mode: str = "save",
        fps: float = 10.0,
    ):
        """
        Visualize with Rerun - interactive 3D viewer.
        
        Args:
            rgbs: (V, T, H, W, 3) RGB frames
            trajectories: (T, N, 3) tracked points
            visibilities: (T, N) confidence scores
            metadata: Camera and scene metadata
            output_path: Path to save .rrd recording (if mode=="save")
            mode: "save" (file), "spawn" (window), or "stream" (TCP)
            fps: Playback frame rate
        """
        try:
            import rerun as rr
            from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun, log_tracks_to_rerun
        except ImportError:
            logger.error("Rerun required: pip install rerun-sdk==0.21.0")
            return

        logger.info(f"Visualizing with Rerun ({mode} mode)...")
        
        if output_path is None:
            output_path = str(self.output_dir / "tracking.rrd")
        
        # Initialize Rerun
        rr.init("mvtracker", recording_id="tracking")
        if mode == "stream":
            rr.connect_tcp()
        elif mode == "spawn":
            rr.spawn()

        # Convert to torch tensors
        rgbs_t = torch.from_numpy(rgbs).float().permute(0, 1, 4, 2, 3)[None] / 255.0  # (1, V, T, 3, H, W)
        intrs = np.array(metadata['intrinsics'], dtype=np.float32)
        extrs = np.array(metadata['extrinsics'], dtype=np.float32)
        
        # Broadcast to all frames if needed
        if intrs.ndim == 3:
            intrs = np.tile(intrs[:, None], (1, rgbs.shape[1], 1, 1))
        if extrs.ndim == 3:
            extrs = np.tile(extrs[:, None], (1, rgbs.shape[1], 1, 1))
        
        intrs_t = torch.from_numpy(intrs).float()[None]  # (1, V, T, 3, 3)
        extrs_t = torch.from_numpy(extrs).float()[None]  # (1, V, T, 3, 4)

        # Zero depths for point cloud
        depths = np.zeros((rgbs.shape[0], rgbs.shape[1], 1, rgbs.shape[2], rgbs.shape[3]), dtype=np.float32)
        depths_t = torch.from_numpy(depths).float()[None]  # (1, V, T, 1, H, W)

        # Log point clouds and camera positions
        log_pointclouds_to_rerun(
            dataset_name="scene",
            datapoint_idx=0,
            rgbs=rgbs_t,
            depths=depths_t,
            intrs=intrs_t,
            extrs=extrs_t,
            fps=fps,
            log_camera_frustrum=True,
            log_rgb_pointcloud=True,
            log_rgb_image=False,
        )

        # Convert trajectories to torch for logging
        query_points = torch.from_numpy(np.array(metadata['query_points'])).float()
        traj_t = torch.from_numpy(trajectories).float()
        vis_t = torch.from_numpy(visibilities).float()
        
        # Log tracked points
        log_tracks_to_rerun(
            dataset_name="MVTracker",
            datapoint_idx=0,
            predictor_name="MVTracker",
            query_points_3d=query_points[None],
            pred_trajectories=traj_t[None],
            pred_visibilities=vis_t[None],
            gt_trajectories_3d_worldspace=None,
            gt_visibilities_any_view=None,
            fps=fps,
        )

        if mode == "save":
            logger.info(f"Saving Rerun recording to {output_path}")
            rr.save(output_path)
            logger.info(f"Open with: rerun {output_path}")
        
        logger.info("Visualization complete!")

    def save_results(self, trajectories: np.ndarray, visibilities: np.ndarray, metadata: Dict):
        """Save tracking results."""
        results_path = self.output_dir / "results.npz"
        np.savez(results_path, trajectories=trajectories, visibilities=visibilities, **metadata)
        logger.info(f"Results saved: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="MVTracker 3D point tracking pipeline with Rerun visualization")
    parser.add_argument("data_dir", help="Dataset directory with color/ subdirectory")
    parser.add_argument("--output_dir", default="./mvtracker_results", help="Output directory")
    parser.add_argument("--iters", type=int, default=4, help="Refinement iterations")
    parser.add_argument("--resolution", help="Target resolution (e.g., '360,640' for H,W)")
    parser.add_argument("--frame_skip", type=int, default=1, help="Load every Nth frame")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device")
    parser.add_argument("--visualize", action="store_true", help="Enable Rerun visualization")
    parser.add_argument("--rerun_mode", choices=["save", "spawn", "stream"], default="save", 
                        help="Rerun mode: save to file, spawn window, or stream TCP")
    parser.add_argument("--rrd_output", help="Path to save .rrd file (Rerun recording)")
    
    args = parser.parse_args()

    resolution = None
    if args.resolution:
        h, w = map(int, args.resolution.split(","))
        resolution = (h, w)

    # Initialize pipeline
    pipeline = MVTrackerPipeline(
        args.data_dir,
        args.output_dir,
        device=args.device,
        resolution=resolution,
        frame_skip=args.frame_skip,
    )

    # Run tracking
    traj, vis, meta = pipeline.track(num_views=3, iters=args.iters)
    
    logger.info(f"Tracking complete: {traj.shape}")
    logger.info(f"  Trajectories: {traj.shape} (frames, points, xyz)")
    logger.info(f"  Visibilities: {vis.shape} (frames, points)")

    # Save results
    pipeline.save_results(traj, vis, meta)

    # Visualize if requested
    if args.visualize:
        rgbs, _ = pipeline._load_video_frames(num_views=3)
        rrd_path = args.rrd_output or str(pipeline.output_dir / "tracking.rrd")
        pipeline.visualize_3d_rerun(rgbs, traj, vis, meta, output_path=rrd_path, mode=args.rerun_mode)

    logger.info(f"Done! Results in {pipeline.output_dir}")


if __name__ == "__main__":
    main()
