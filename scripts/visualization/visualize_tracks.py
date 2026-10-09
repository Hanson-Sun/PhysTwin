#!/usr/bin/env python3
"""
Unified visualization script for 3D tracks from multiple trackers.
Supports: MVTracker, SpaTrackerV2, and CoTracker
Navigate frames with arrow keys or 'n'/'p', toggle playback with space.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import argparse


class TrackVisualizer:
    def __init__(self, obj_pts, obj_vis, ctrl_pts, ctrl_vis, tracker_name="Unknown"):
        self.obj_pts = obj_pts
        self.obj_vis = obj_vis
        self.ctrl_pts = ctrl_pts
        self.ctrl_vis = ctrl_vis
        self.tracker_name = tracker_name
        
        self.frame_idx = 0
        self.num_frames = obj_pts.shape[0]
        self.is_playing = False
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name=f"{tracker_name} - 3D Track Visualization",
            width=1200,
            height=800
        )
        
        self.pcd_obj = o3d.geometry.PointCloud()
        self.pcd_ctrl = o3d.geometry.PointCloud()
        self.axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        
        self.vis.add_geometry(self.pcd_obj)
        self.vis.add_geometry(self.pcd_ctrl)
        self.vis.add_geometry(self.axes)
        
        # Register keyboard callbacks
        self.vis.register_key_callback(ord('n'), self._on_next)
        self.vis.register_key_callback(ord('p'), self._on_prev)
        self.vis.register_key_callback(ord(' '), self._on_play_pause)
        self.vis.register_key_callback(262, self._on_next)  # Right arrow
        self.vis.register_key_callback(263, self._on_prev)  # Left arrow
        self.vis.register_key_callback(ord('q'), self._on_quit)
        
        self.should_quit = False
        self.update_frame(0)
    
    def _on_next(self, vis):
        self.frame_idx = (self.frame_idx + 1) % self.num_frames
        self.update_frame(self.frame_idx)
    
    def _on_prev(self, vis):
        self.frame_idx = (self.frame_idx - 1) % self.num_frames
        self.update_frame(self.frame_idx)
    
    def _on_play_pause(self, vis):
        self.is_playing = not self.is_playing
        status = "PLAYING" if self.is_playing else "PAUSED"
        print(f"{status} (Frame {self.frame_idx}/{self.num_frames-1})")
    
    def _on_quit(self, vis):
        self.should_quit = True
        self.vis.destroy_window()
    
    def update_frame(self, frame_idx):
        self.frame_idx = frame_idx
        
        # Update object points (cyan)
        if self.obj_pts.shape[1] > 0 and self.obj_vis.shape[0] > frame_idx:
            obj_valid = self.obj_vis[frame_idx] > 0
            if obj_valid.any():
                pts = self.obj_pts[frame_idx][obj_valid].astype(np.float64)
                # Handle 2D points (e.g., from CoTracker) by padding with z=0
                if pts.shape[-1] == 2:
                    pts = np.pad(pts, ((0, 0), (0, 1)), mode='constant')
                self.pcd_obj.points = o3d.utility.Vector3dVector(pts)
                colors = np.tile([0, 1, 1], (len(pts), 1))
                self.pcd_obj.colors = o3d.utility.Vector3dVector(colors)
            else:
                self.pcd_obj.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                self.pcd_obj.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        
        # Update controller points (yellow)
        if self.ctrl_pts.shape[1] > 0 and self.ctrl_vis.shape[0] > frame_idx:
            ctrl_valid = self.ctrl_vis[frame_idx] > 0
            if ctrl_valid.any():
                pts = self.ctrl_pts[frame_idx][ctrl_valid].astype(np.float64)
                # Handle 2D points (e.g., from CoTracker) by padding with z=0
                if pts.shape[-1] == 2:
                    pts = np.pad(pts, ((0, 0), (0, 1)), mode='constant')
                self.pcd_ctrl.points = o3d.utility.Vector3dVector(pts)
                colors = np.tile([1, 1, 0], (len(pts), 1))
                self.pcd_ctrl.colors = o3d.utility.Vector3dVector(colors)
            else:
                self.pcd_ctrl.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                self.pcd_ctrl.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        
        self.vis.update_geometry(self.pcd_obj)
        self.vis.update_geometry(self.pcd_ctrl)
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def run(self):
        while not self.should_quit:
            if self.is_playing:
                self._on_next(self.vis)
            self.vis.poll_events()
            self.vis.update_renderer()


def load_mvtrack_tracks(base_path, case_name):
    """Load MVTracker tracks from object_tracks.npz and controller_tracks.npz."""
    track_path = Path(base_path) / case_name / "mvtrack"
    
    obj_file = track_path / "object_tracks.npz"
    ctrl_file = track_path / "controller_tracks.npz"
    
    assert obj_file.exists(), f"Object tracks not found: {obj_file}"
    assert ctrl_file.exists(), f"Controller tracks not found: {ctrl_file}"
    
    obj_data = np.load(obj_file)
    object_tracks = obj_data["tracks"]
    object_vis = obj_data["visibility"]
    
    ctrl_data = np.load(ctrl_file)
    controller_tracks = ctrl_data["tracks"]
    controller_vis = ctrl_data["visibility"]
    
    print(f"✓ Loaded MVTracker object tracks: {object_tracks.shape}, vis: {object_vis.shape}")
    print(f"✓ Loaded MVTracker controller tracks: {controller_tracks.shape}, vis: {controller_vis.shape}")
    
    return object_tracks, object_vis, controller_tracks, controller_vis


def load_spatracker_tracks(base_path, case_name, num_cams):
    """Load SpaTrackerV2 tracks from separate camera directories."""
    object_tracks = []
    controller_tracks = []
    object_vis = []
    controller_vis = []
    
    for cam_id in range(num_cams):
        cam_dir = Path(base_path) / case_name / "spatracker" / f"camera_{cam_id}"
        
        obj_file = cam_dir / "object_tracks.npz"
        ctrl_file = cam_dir / "controller_tracks.npz"
        
        if obj_file.exists():
            obj_data = np.load(obj_file)
            object_tracks.append(obj_data["tracks"])
            vis = obj_data["visibility"]
            vis = np.squeeze(vis, axis=-1) if vis.ndim > 2 else vis
            object_vis.append(vis)
            print(f"✓ Loaded camera {cam_id} object tracks: {obj_data['tracks'].shape}, vis: {vis.shape}")
        
        if ctrl_file.exists():
            ctrl_data = np.load(ctrl_file)
            controller_tracks.append(ctrl_data["tracks"])
            vis = ctrl_data["visibility"]
            vis = np.squeeze(vis, axis=-1) if vis.ndim > 2 else vis
            controller_vis.append(vis)
            print(f"✓ Loaded camera {cam_id} controller tracks: {ctrl_data['tracks'].shape}, vis: {vis.shape}")
    
    # Concatenate across cameras, trim to minimum frame count
    if object_tracks:
        min_frames = min(t.shape[0] for t in object_tracks)
        object_tracks = [t[:min_frames] for t in object_tracks]
        object_vis = [v[:min_frames] for v in object_vis]
        object_tracks = np.concatenate(object_tracks, axis=1)
        object_vis = np.concatenate(object_vis, axis=1)
        object_vis = np.squeeze(object_vis, axis=-1) if object_vis.ndim > 2 else object_vis
    else:
        object_tracks = np.zeros((0, 0, 3))
        object_vis = np.zeros((0, 0))
    
    if controller_tracks:
        min_frames = min(t.shape[0] for t in controller_tracks)
        controller_tracks = [t[:min_frames] for t in controller_tracks]
        controller_vis = [v[:min_frames] for v in controller_vis]
        controller_tracks = np.concatenate(controller_tracks, axis=1)
        controller_vis = np.concatenate(controller_vis, axis=1)
        controller_vis = np.squeeze(controller_vis, axis=-1) if controller_vis.ndim > 2 else controller_vis
    else:
        controller_tracks = np.zeros((0, 0, 3))
        controller_vis = np.zeros((0, 0))
    
    return object_tracks, object_vis, controller_tracks, controller_vis


def load_cotracker_tracks(base_path, case_name, num_cams):
    """Load CoTracker tracks from per-camera files.
    
    CoTracker stores 2D pixel coordinates. This loader attempts to load them
    but note: proper 3D visualization requires depth lifting (see data_process_track.py).
    """
    object_tracks = []
    controller_tracks = []
    object_vis = []
    controller_vis = []
    
    track_path = Path(base_path) / case_name / "cotracker"
    
    for cam_id in range(num_cams):
        track_file = track_path / f"{cam_id}.npz"
        
        if track_file.exists():
            data = np.load(track_file)
            tracks = data["tracks"]  # [T, N, 2] - 2D pixel coords
            visibility = data["visibility"]
            
            # For CoTracker, all tracks are currently treated as object nodes
            # (requires depth lifting for proper 3D coordinates)
            object_tracks.append(tracks)
            object_vis.append(visibility)
            print(f"✓ Loaded camera {cam_id} CoTracker tracks: {tracks.shape}, vis: {visibility.shape}")
            print(f"  Note: CoTracker 2D pixels will be visualized as xy-plane (z=0). Use data_process_track.py for 3D lifting.")
    
    # Concatenate across cameras, trim to minimum frame count
    if object_tracks:
        min_frames = min(t.shape[0] for t in object_tracks)
        object_tracks = [t[:min_frames] for t in object_tracks]
        object_vis = [v[:min_frames] for v in object_vis]
        object_tracks = np.concatenate(object_tracks, axis=1)
        object_vis = np.concatenate(object_vis, axis=1)
    else:
        object_tracks = np.zeros((0, 0, 2))
        object_vis = np.zeros((0, 0))
    
    # Create dummy controller tracks
    controller_tracks = np.zeros((0, 0, 3))
    controller_vis = np.zeros((0, 0))
    
    return object_tracks, object_vis, controller_tracks, controller_vis


def load_tracks(base_path, case_name, track_method, num_cams=3):
    """Load tracks using specified method.
    
    Args:
        base_path: Base directory path
        case_name: Case name
        track_method: One of 'mvtrack', 'spatracker', 'cotracker'
        num_cams: Number of cameras (for multi-camera trackers)
    
    Returns:
        Tuple of (object_tracks, object_vis, controller_tracks, controller_vis)
    """
    print(f"\n{'='*60}")
    print(f"Loading {track_method.upper()} tracks from {base_path}/{case_name}")
    print(f"{'='*60}\n")
    
    if track_method == "mvtrack":
        return load_mvtrack_tracks(base_path, case_name)
    elif track_method == "spatracker":
        return load_spatracker_tracks(base_path, case_name, num_cams)
    elif track_method == "cotracker":
        return load_cotracker_tracks(base_path, case_name, num_cams)
    else:
        raise ValueError(f"Unknown tracking method: {track_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified visualization for 3D tracks from multiple trackers"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base data directory"
    )
    parser.add_argument(
        "--case_name",
        type=str,
        required=True,
        help="Case name"
    )
    parser.add_argument(
        "--track_method",
        type=str,
        choices=["mvtrack", "spatracker", "cotracker"],
        default="spatracker",
        help="Tracking method to visualize (default: spatracker)"
    )
    parser.add_argument(
        "--num_cams",
        type=int,
        default=3,
        help="Number of cameras (for multi-camera trackers)"
    )
    
    args = parser.parse_args()
    
    obj_pts, obj_vis, ctrl_pts, ctrl_vis = load_tracks(
        args.base_path, args.case_name, args.track_method, args.num_cams
    )
    
    print(f"\n✓ Object tracks: {obj_pts.shape}")
    print(f"✓ Controller tracks: {ctrl_pts.shape}\n")
    print("Controls: 'n'/'→' = next, 'p'/'←' = prev, 'space' = play/pause, 'q' = quit\n")
    
    visualizer = TrackVisualizer(obj_pts, obj_vis, ctrl_pts, ctrl_vis, args.track_method)
    visualizer.run()
