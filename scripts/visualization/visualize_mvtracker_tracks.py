#!/usr/bin/env python3
"""
Generic 3D visualization of object tracking outputs without external dependencies.
Supports MVTracker, CoTracker, and other multi-view or monocular tracking formats.
Uses matplotlib for interactive 3D point cloud visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import argparse
import sys
import glob


def load_tracks(track_file):
    """Load tracks from .npz file. Handles multiple track formats."""
    data = np.load(track_file)
    available_keys = list(data.keys())
    
    tracks = None
    vis = None
    
    # Try to find tracks and visibility in common key formats
    key_pairs = [
        ('tracks', 'visibility'),
        ('tracks', 'vis'),
        ('object_tracks', 'visibility'),
        ('object_tracks', 'vis'),
        ('trajectories', 'visibility'),
        ('trajectories', 'vis'),
    ]
    
    for track_key, vis_key in key_pairs:
        if track_key in available_keys and vis_key in available_keys:
            tracks = data[track_key]
            vis = data[vis_key]
            break
    
    # If no matching pair, try to infer from available keys
    if tracks is None:
        # Look for any key that looks like tracks
        track_candidates = [k for k in available_keys if 'track' in k.lower() or 'traj' in k.lower()]
        vis_candidates = [k for k in available_keys if 'vis' in k.lower()]
        
        if track_candidates and vis_candidates:
            tracks = data[track_candidates[0]]
            vis = data[vis_candidates[0]]
        elif track_candidates:
            tracks = data[track_candidates[0]]
            # Create dummy visibility (all visible)
            vis = np.ones(tracks.shape[:2])
        else:
            raise KeyError(f"Could not find tracks in {track_file}. Available keys: {available_keys}")
    
    return tracks, vis


def find_track_files(base_path):
    """Find all track .npz files in a directory, returning them with descriptive names."""
    track_files = {}
    
    # Search for various track file patterns
    patterns = [
        "*_tracks.npz",          # object_tracks.npz, controller_tracks.npz
        "tracks.npz",
        "*_track.npz",
        "[0-9].npz",             # 0.npz, 1.npz, etc. (per-camera tracks)
    ]
    
    found_files = set()
    for pattern in patterns:
        files = glob.glob(str(base_path / pattern))
        found_files.update(files)
    
    # Sort and assign names
    for fpath in sorted(found_files):
        fname = Path(fpath).stem
        
        # Create descriptive name
        if "object" in fname.lower():
            name = "Object"
        elif "controller" in fname.lower():
            name = "Controller"
        elif "hand" in fname.lower():
            name = "Hand"
        elif fname.isdigit():
            name = f"Camera {fname}"
        else:
            name = fname.replace("_", " ").title()
        
        track_files[name] = Path(fpath)
    
    return track_files


# Color palette for different track types
COLORS = {
    'Object': 'cyan',
    'Controller': 'magenta',
    'Hand': 'red',
    'default': 'blue',
}


def get_color(track_name):
    """Get color for a track type."""
    for key in COLORS:
        if key.lower() in track_name.lower():
            return COLORS[key]
    return COLORS['default']


def plot_frame(ax, track_data_dict, frame_idx=0, title="", show_history=5, max_points_to_draw=100):
    """Plot a single frame of tracks with trajectory lines.
    
    Args:
        track_data_dict: Dict of {track_name: (tracks [T,N,3], visibility [T,N])}
        show_history: Number of previous frames to show as history (trajectory lines)
        max_points_to_draw: Maximum number of points to draw (sample if more available)
    """
    ax.clear()
    
    # Determine history range
    history_start = max(0, frame_idx - show_history)
    
    # Plot each track type
    for track_name, (tracks, vis) in track_data_dict.items():
        if tracks is None or tracks.shape[0] <= frame_idx:
            continue
        
        color = get_color(track_name)
        n_points = tracks.shape[1]
        
        # Sample points if there are too many
        sample_indices = np.linspace(0, n_points - 1, 
                                    min(max_points_to_draw, n_points), dtype=int)
        
        # Draw trajectory lines for visible points
        for point_idx in sample_indices:
            if vis[frame_idx, point_idx] > 0.5:
                pts = tracks[history_start:frame_idx+1, point_idx, :]
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.2, linewidth=0.5)
        
        # Plot current visible points (bright)
        vis_mask = vis[frame_idx, sample_indices] > 0.5
        visible_pts = tracks[frame_idx][sample_indices][vis_mask]
        if len(visible_pts) > 0:
            ax.scatter(visible_pts[:, 0], visible_pts[:, 1], visible_pts[:, 2], 
                      c=color, s=20, alpha=0.8, label=f'{track_name} (visible)')
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title} (Frame {frame_idx}) - History: {show_history} frames")
    ax.legend(fontsize=8, loc='upper right')
    
    # Auto-scale with padding
    all_points = []
    for tracks, vis in track_data_dict.values():
        if tracks is not None and tracks.shape[0] > frame_idx:
            all_points.append(tracks[frame_idx])
    
    if all_points:
        all_pts = np.vstack(all_points)
        bounds = np.array([all_pts.min(axis=0), all_pts.max(axis=0)])
        center = bounds.mean(axis=0)
        extent = (bounds.max() - bounds.min()).max() / 2
        
        ax.set_xlim(center[0] - extent, center[0] + extent)
        ax.set_ylim(center[1] - extent, center[1] + extent)
        ax.set_zlim(center[2] - extent, center[2] + extent)
    
    ax.grid(False)


def interactive_viewer(track_data_dict, case_name, num_frames):
    """Interactive 3D viewer with frame slider and trajectory lines."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # State variables
    current_frame = [0]
    show_history = [3]
    max_points = [100]
    
    # Initial plot
    plot_frame(ax, track_data_dict, frame_idx=0, title=case_name, 
              show_history=show_history[0], max_points_to_draw=max_points[0])
    
    # Add text display
    text_str = f"Frame: 0 / {num_frames-1}  |  History: {show_history[0]}  |  Points: {max_points[0]}"
    status_text = fig.text(0.5, 0.02, text_str, ha='center', fontsize=11, weight='bold')
    
    def update_frame(frame_idx):
        current_frame[0] = max(0, min(frame_idx, num_frames - 1))
        plot_frame(ax, track_data_dict, frame_idx=current_frame[0], title=case_name,
                  show_history=show_history[0], max_points_to_draw=max_points[0])
        
        # Update text
        text = f"Frame: {current_frame[0]} / {num_frames-1}  |  History: {show_history[0]} frames  |  Points sampled: {max_points[0]}"
        status_text.set_text(text)
        
        fig.canvas.draw()
    
    def update_history(delta):
        show_history[0] = max(1, min(num_frames - 1, show_history[0] + delta))
        update_frame(current_frame[0])
    
    def update_points(delta):
        max_points[0] = max(10, min(500, max_points[0] + delta))
        update_frame(current_frame[0])
    
    # Keyboard controls
    def on_key(event):
        if event.key == 'left':
            update_frame(current_frame[0] - 1)
        elif event.key == 'right':
            update_frame(current_frame[0] + 1)
        elif event.key == 'home':
            update_frame(0)
        elif event.key == 'end':
            update_frame(num_frames - 1)
        elif event.key == 'up':
            update_history(2)
        elif event.key == 'down':
            update_history(-2)
        elif event.key == '+' or event.key == '=':
            update_points(20)
        elif event.key == '-' or event.key == '_':
            update_points(-20)
        elif event.key == 'q':
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Instructions
    print(f"\n{'='*60}")
    print(f"3D Track Visualization: {case_name}")
    print(f"Total frames: {num_frames}")
    print(f"Track types: {', '.join(track_data_dict.keys())}")
    print(f"{'='*60}")
    print(f"\nControls:")
    print(f"  LEFT/RIGHT arrow:  Previous/Next frame")
    print(f"  HOME/END:          First/Last frame")
    print(f"  UP/DOWN arrow:     Increase/Decrease trajectory history (±2 frames)")
    print(f"  +/- :              Increase/Decrease number of points shown (±20)")
    print(f"  Q:                 Quit")
    print(f"  Mouse:             Rotate/Pan/Zoom view")
    print(f"{'='*60}\n")
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D tracking outputs (MVTracker, CoTracker, etc.)")
    parser.add_argument("--base_path", type=str, required=True, help="Base data path")
    parser.add_argument("--case_name", type=str, required=True, help="Case name")
    parser.add_argument("--track_dir", type=str, default=None, help="Track directory (default: {base_path}/{case_name}/mvtrack)")
    
    args = parser.parse_args()
    
    base_path = Path(args.base_path)
    case_name = args.case_name
    
    # Determine track directory
    if args.track_dir:
        track_dir = Path(args.track_dir)
    else:
        track_dir = base_path / case_name / "mvtrack"
    
    if not track_dir.exists():
        print(f"✗ Track directory not found: {track_dir}")
        sys.exit(1)
    
    # Find all track files
    track_files = find_track_files(track_dir)
    
    if not track_files:
        print(f"✗ No track files (.npz) found in {track_dir}")
        sys.exit(1)
    
    print(f"✓ Found {len(track_files)} track file(s): {', '.join(track_files.keys())}")
    
    # Load all tracks
    track_data_dict = {}
    num_frames = None
    
    for track_name, track_file in track_files.items():
        try:
            tracks, vis = load_tracks(track_file)
            track_data_dict[track_name] = (tracks, vis)
            print(f"  ✓ {track_name}: {tracks.shape[0]} frames, {tracks.shape[1]} points")
            
            if num_frames is None:
                num_frames = tracks.shape[0]
            elif num_frames != tracks.shape[0]:
                print(f"  ⚠ Warning: {track_name} has different number of frames ({tracks.shape[0]} vs {num_frames})")
        except Exception as e:
            print(f"  ✗ Failed to load {track_name}: {e}")
    
    if not track_data_dict:
        print(f"✗ Failed to load any tracks")
        sys.exit(1)
    
    if num_frames is None:
        print(f"✗ Could not determine number of frames")
        sys.exit(1)
    
    # Launch viewer
    interactive_viewer(track_data_dict, case_name, num_frames)


if __name__ == "__main__":
    main()
