#!/usr/bin/env python3
"""
Visualization script for SpaTrackerV2 3D tracks with smooth Open3D viewer.
Navigate frames with arrow keys or 'n'/'p', toggle playback with space.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import argparse


class TrackVisualizer:
    def __init__(self, obj_pts, obj_vis, ctrl_pts, ctrl_vis):
        self.obj_pts = obj_pts
        self.obj_vis = obj_vis
        self.ctrl_pts = ctrl_pts
        self.ctrl_vis = ctrl_vis
        
        self.frame_idx = 0
        self.num_frames = obj_pts.shape[0]
        self.is_playing = False
        
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="SpaTrackerV2 Tracks", width=1200, height=800)
        
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
    
    # Concatenate across cameras
    if object_tracks:
        object_tracks = np.concatenate(object_tracks, axis=1)
        object_vis = np.concatenate(object_vis, axis=1)
        object_vis = np.squeeze(object_vis, axis=-1) if object_vis.ndim > 2 else object_vis
    else:
        object_tracks = np.zeros((0, 0, 3))
        object_vis = np.zeros((0, 0))
    
    if controller_tracks:
        controller_tracks = np.concatenate(controller_tracks, axis=1)
        controller_vis = np.concatenate(controller_vis, axis=1)
        controller_vis = np.squeeze(controller_vis, axis=-1) if controller_vis.ndim > 2 else controller_vis
    else:
        controller_tracks = np.zeros((0, 0, 3))
        controller_vis = np.zeros((0, 0))
    
    return object_tracks, object_vis, controller_tracks, controller_vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SpaTrackerV2 3D tracks")
    parser.add_argument("--base_path", type=str, required=True, help="Base data directory")
    parser.add_argument("--case_name", type=str, required=True, help="Case name")
    parser.add_argument("--num_cams", type=int, default=3, help="Number of cameras")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Loading SpaTrackerV2 tracks from {args.base_path}/{args.case_name}")
    print(f"{'='*60}\n")
    
    obj_pts, obj_vis, ctrl_pts, ctrl_vis = load_spatracker_tracks(
        args.base_path, args.case_name, args.num_cams
    )
    
    print(f"\n✓ Object tracks: {obj_pts.shape}")
    print(f"✓ Controller tracks: {ctrl_pts.shape}\n")
    print("Controls: 'n'/'→' = next, 'p'/'←' = prev, 'space' = play/pause, 'q' = quit\n")
    
    visualizer = TrackVisualizer(obj_pts, obj_vis, ctrl_pts, ctrl_vis)
    visualizer.run()
