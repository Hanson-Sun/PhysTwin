#!/usr/bin/env python3
"""
Compare two final_data.pkl files from different tracking runs with detailed metrics and plots.

Usage Examples:
  python compare_data.py --pkl1 data/case1/final_data.pkl --pkl2 data/case2/final_data.pkl
  python compare_data.pkl --base_path data --case1 run_a --case2 run_b --output compare_result --max_frames 100
"""

import pickle
import numpy as np
import cv2
import argparse
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
import csv


def load_track_data(pkl_path):
    """Load track data from pickle file"""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def compute_track_metrics(tracks1, tracks2, visibilities1, visibilities2):
    """Compute resolution-invariant metrics between two tracks"""
    T = min(tracks1.shape[0], tracks2.shape[0])
    
    # 1. Center of mass trajectory
    com1 = np.nanmean(tracks1[:T], axis=1)  # [T, 3]
    com2 = np.nanmean(tracks2[:T], axis=1)  # [T, 3]
    
    com_diff = np.linalg.norm(com1 - com2, axis=1)  # [T]
    com_mae = np.nanmean(com_diff)
    com_rmse = np.sqrt(np.nanmean(com_diff ** 2))
    
    # Correlation of COM trajectories
    if len(com1) > 2 and np.std(com1) > 1e-6 and np.std(com2) > 1e-6:
        com_corr = np.corrcoef(com1.flatten(), com2.flatten())[0, 1]
    else:
        com_corr = np.nan
    
    # 2. Visibility comparison
    vis1_count = (visibilities1[:T] > 0.5).sum(axis=1)  # points visible per frame
    vis2_count = (visibilities2[:T] > 0.5).sum(axis=1)
    vis_diff = np.abs(vis1_count - vis2_count).mean()
    
    # Relative visibility difference (normalized by average)
    avg_vis = (vis1_count.mean() + vis2_count.mean()) / 2
    if avg_vis > 0:
        vis_relative_diff = np.abs(vis1_count - vis2_count).mean() / avg_vis * 100
    else:
        vis_relative_diff = 0
    
    # 3. Spatial spread (variance of points in each frame)
    spread1 = np.nanstd(tracks1[:T], axis=1).mean(axis=1)  # [T]
    spread2 = np.nanstd(tracks2[:T], axis=1).mean(axis=1)  # [T]
    spread_diff = np.nanmean(np.abs(spread1 - spread2))
    
    # Relative spread difference
    avg_spread = (spread1.mean() + spread2.mean()) / 2
    if avg_spread > 0:
        spread_relative_diff = spread_diff / avg_spread * 100
    else:
        spread_relative_diff = 0
    
    # 4. Velocity/motion comparison
    vel1 = np.linalg.norm(np.diff(com1, axis=0), axis=1)
    vel2 = np.linalg.norm(np.diff(com2, axis=0), axis=1)
    velocity_mae = np.nanmean(np.abs(vel1 - vel2))
    
    # 5. Point density alignment (relative)
    density1 = vis1_count.astype(float)
    density2 = vis2_count.astype(float)
    # Normalize densities
    if density1.max() > 0:
        density1_norm = density1 / density1.max()
    else:
        density1_norm = density1
    if density2.max() > 0:
        density2_norm = density2 / density2.max()
    else:
        density2_norm = density2
    
    # Correlation of normalized densities
    if np.std(density1_norm) > 1e-6 and np.std(density2_norm) > 1e-6:
        density_corr = np.corrcoef(density1_norm, density2_norm)[0, 1]
    else:
        density_corr = np.nan
    
    return {
        'com_mae': com_mae,
        'com_rmse': com_rmse,
        'com_corr': com_corr,
        'velocity_mae': velocity_mae,
        'visibility_diff': vis_diff,
        'visibility_relative_diff': vis_relative_diff,
        'spread_diff': spread_diff,
        'spread_relative_diff': spread_relative_diff,
        'density_corr': density_corr,
        'num_points_1': tracks1.shape[1],
        'num_points_2': tracks2.shape[1],
        'com1': com1,
        'com2': com2,
        'vel1': vel1,
        'vel2': vel2,
        'density1': density1,
        'density2': density2,
        'spread1': spread1,
        'spread2': spread2,
    }


def create_comparison_plots(metrics_dict, track_type, output_dir):
    """Create and save comparison plots"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Center of Mass Trajectory Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    com1 = metrics_dict['com1']
    com2 = metrics_dict['com2']
    
    # 3D trajectory projection
    axes[0].scatter(com1[:, 0], com1[:, 1], c=np.arange(len(com1)), cmap='Blues', label='Run 1', alpha=0.7, s=20)
    axes[0].scatter(com2[:, 0], com2[:, 1], c=np.arange(len(com2)), cmap='Reds', label='Run 2', alpha=0.7, s=20)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title(f'{track_type} Center of Mass (XY plane)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # COM distance over time
    com_diff = np.linalg.norm(com1 - com2, axis=1)
    axes[1].plot(com_diff, linewidth=2, color='purple')
    axes[1].fill_between(range(len(com_diff)), 0, com_diff, alpha=0.3, color='purple')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('COM Distance (m)')
    axes[1].set_title(f'{track_type} Center of Mass Alignment Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{track_type.lower()}_com_trajectory.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # 2. Velocity Comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    
    vel1 = metrics_dict['vel1']
    vel2 = metrics_dict['vel2']
    
    ax.plot(vel1, label='Run 1', linewidth=2, alpha=0.7)
    ax.plot(vel2, label='Run 2', linewidth=2, alpha=0.7)
    ax.fill_between(range(len(vel1)), vel1, vel2, alpha=0.2, color='gray', label='Difference')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (m/frame)')
    ax.set_title(f'{track_type} Velocity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{track_type.lower()}_velocity.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # 3. Point Density (Visibility) Comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    
    density1 = metrics_dict['density1']
    density2 = metrics_dict['density2']
    
    ax.plot(density1, label='Run 1', marker='o', markersize=3, linewidth=2, alpha=0.7)
    ax.plot(density2, label='Run 2', marker='s', markersize=3, linewidth=2, alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Visible Points')
    ax.set_title(f'{track_type} Point Density Per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{track_type.lower()}_density.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    # 4. Spatial Spread Comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    
    spread1 = metrics_dict['spread1']
    spread2 = metrics_dict['spread2']
    
    ax.plot(spread1, label='Run 1', linewidth=2, alpha=0.7)
    ax.plot(spread2, label='Run 2', linewidth=2, alpha=0.7)
    ax.fill_between(range(len(spread1)), spread1, spread2, alpha=0.2, color='gray')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Spatial Spread (m)')
    ax.set_title(f'{track_type} Point Cloud Spatial Spread')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{track_type.lower()}_spread.png', dpi=100, bbox_inches='tight')
    plt.close()


def save_metrics_to_file(metrics_dict, track_type, output_dir):
    """Save metrics to CSV and JSON files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f'{track_type.lower()}_metrics.csv'
    json_path = output_dir / f'{track_type.lower()}_metrics.json'
    
    # Prepare data for CSV
    csv_data = {
        'Metric': [],
        'Value': [],
    }
    
    for key, value in metrics_dict.items():
        if not isinstance(value, np.ndarray):  # Skip array values
            csv_data['Metric'].append(key)
            if isinstance(value, float):
                csv_data['Value'].append(f"{value:.6f}")
            else:
                csv_data['Value'].append(str(value))
    
    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Metric', 'Value'])
        writer.writeheader()
        for i in range(len(csv_data['Metric'])):
            writer.writerow({'Metric': csv_data['Metric'][i], 'Value': csv_data['Value'][i]})
    
    # Write JSON
    json_compatible = {}
    for key, value in metrics_dict.items():
        if isinstance(value, np.ndarray):
            json_compatible[key] = value.tolist()
        elif isinstance(value, (np.floating, float)):
            json_compatible[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            json_compatible[key] = int(value)
        else:
            json_compatible[key] = str(value)
    
    with open(json_path, 'w') as f:
        json.dump(json_compatible, f, indent=2)
    
    return csv_path, json_path


def render_comparison_video(data1, data2, output_path, max_frames=-1):
    """Render side-by-side comparison video"""
    obj1 = data1.get('object_points', data1.get('trajectories', None))
    obj2 = data2.get('object_points', data2.get('trajectories', None))
    ctrl1 = data1.get('controller_points', None)
    ctrl2 = data2.get('controller_points', None)
    
    if obj1 is None or obj2 is None:
        print("✗ No object tracks found")
        return
    
    T = min(obj1.shape[0], obj2.shape[0])
    if max_frames > 0:
        T = min(T, max_frames)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = Path(output_path).resolve()
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (1536, 768))
    
    print(f"Rendering {T} frames to {output_path}...")
    
    for t in range(T):
        # Create two 768x768 images side by side
        img1 = np.ones((768, 768, 3), dtype=np.uint8) * 255
        img2 = np.ones((768, 768, 3), dtype=np.uint8) * 255
        
        img1 = draw_tracks_on_image(img1, obj1[t], ctrl1[t] if ctrl1 is not None else None, title="Run 1")
        img2 = draw_tracks_on_image(img2, obj2[t], ctrl2[t] if ctrl2 is not None else None, title="Run 2")
        
        # Concatenate
        frame = np.hstack([img1, img2])
        writer.write(frame)
        
        if (t + 1) % max(1, T // 10) == 0:
            print(f"  Frame {t+1}/{T}")
    
    writer.release()
    print(f"\n✓ Saved comparison video to:")
    print(f"  {output_path}\n")


def render_overlay_comparison_video(data1, data2, output_path, max_frames=-1, history_frames=8):
    """Render overlay comparison video with ghost trails showing both runs simultaneously"""
    obj1 = data1.get('object_points', data1.get('trajectories', None))
    obj2 = data2.get('object_points', data2.get('trajectories', None))
    ctrl1 = data1.get('controller_points', None)
    ctrl2 = data2.get('controller_points', None)
    
    if obj1 is None or obj2 is None:
        print("✗ No object tracks found")
        return
    
    T = min(obj1.shape[0], obj2.shape[0])
    if max_frames > 0:
        T = min(T, max_frames)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = Path(output_path).resolve()
    writer = cv2.VideoWriter(str(output_path), fourcc, 30.0, (1024, 1024))
    
    print(f"Rendering overlay comparison with ghost trails: {T} frames to {output_path}...")
    
    for t in range(T):
        # Create single overlay image
        img = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
        
        # Compute common bounds for both runs
        all_pts_1 = [obj1[t]] if obj1 is not None else []
        all_pts_2 = [obj2[t]] if obj2 is not None else []
        if ctrl1 is not None and t < ctrl1.shape[0]:
            all_pts_1.append(ctrl1[t])
        if ctrl2 is not None and t < ctrl2.shape[0]:
            all_pts_2.append(ctrl2[t])
        
        all_pts = np.vstack(all_pts_1 + all_pts_2)
        bounds_min = all_pts.min(axis=0)
        bounds_max = all_pts.max(axis=0)
        bounds_range = bounds_max - bounds_min + 1e-6
        
        # Helper function to project 3D to 2D
        def project_to_2d(pts_3d):
            pts_2d = np.zeros((pts_3d.shape[0], 2))
            pts_2d[:, 0] = (pts_3d[:, 0] - bounds_min[0]) / bounds_range[0] * (1024 - 100) + 50
            pts_2d[:, 1] = (pts_3d[:, 1] - bounds_min[1]) / bounds_range[1] * (1024 - 100) + 50
            return pts_2d.astype(int)
        
        # Draw history trails for Run 1 (blue/cyan)
        history_start = max(0, t - history_frames)
        if obj1 is not None:
            for hist_t in range(history_start, t):
                if hist_t < obj1.shape[0]:
                    pts_current = project_to_2d(obj1[hist_t])
                    pts_next = project_to_2d(obj1[hist_t + 1]) if hist_t + 1 < obj1.shape[0] else pts_current
                    
                    # Fade out older trails
                    alpha = (hist_t - history_start + 1) / (t - history_start + 1)
                    color_intensity = int(200 * alpha)
                    
                    # Sample points for trails
                    for idx in range(0, pts_current.shape[0], max(1, pts_current.shape[0] // 20)):
                        cv2.line(img, tuple(pts_current[idx]), tuple(pts_next[idx]), 
                                (0, color_intensity, color_intensity), 1)  # Cyan trails
        
        # Draw history trails for Run 2 (red)
        if obj2 is not None:
            for hist_t in range(history_start, t):
                if hist_t < obj2.shape[0]:
                    pts_current = project_to_2d(obj2[hist_t])
                    pts_next = project_to_2d(obj2[hist_t + 1]) if hist_t + 1 < obj2.shape[0] else pts_current
                    
                    alpha = (hist_t - history_start + 1) / (t - history_start + 1)
                    color_intensity = int(200 * alpha)
                    
                    for idx in range(0, pts_current.shape[0], max(1, pts_current.shape[0] // 20)):
                        cv2.line(img, tuple(pts_current[idx]), tuple(pts_next[idx]), 
                                (color_intensity, 0, color_intensity), 1)  # Magenta trails
        
        # Draw current points for both runs
        # Run 1: blue object points
        if obj1 is not None and t < obj1.shape[0]:
            pts_2d = project_to_2d(obj1[t])
            for pt in pts_2d:
                cv2.circle(img, tuple(pt), 4, (255, 100, 0), -1)  # Cyan
        
        # Run 2: red object points
        if obj2 is not None and t < obj2.shape[0]:
            pts_2d = project_to_2d(obj2[t])
            for pt in pts_2d:
                cv2.circle(img, tuple(pt), 4, (0, 100, 255), -1)  # Red
        
        # Draw controller points if available
        if ctrl1 is not None and t < ctrl1.shape[0]:
            pts_2d = project_to_2d(ctrl1[t])
            for pt in pts_2d:
                cv2.circle(img, tuple(pt), 3, (255, 200, 0), -1)  # Light cyan
        
        if ctrl2 is not None and t < ctrl2.shape[0]:
            pts_2d = project_to_2d(ctrl2[t])
            for pt in pts_2d:
                cv2.circle(img, tuple(pt), 3, (0, 200, 255), -1)  # Light red
        
        # Add legend
        cv2.putText(img, f"Frame {t}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Blue: Run 1", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        cv2.putText(img, "Red: Run 2", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
        cv2.putText(img, f"History: {history_frames} frames", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        
        writer.write(img)
        
        if (t + 1) % max(1, T // 10) == 0:
            print(f"  Frame {t+1}/{T}")
    
    writer.release()
    print(f"\n✓ Saved overlay video to:")
    print(f"  {output_path}\n")


def draw_tracks_on_image(img, object_pts, controller_pts, title=""):
    """Draw 3D tracks as 2D projection on image"""
    h, w = img.shape[:2]
    
    # Normalize 3D coordinates to image space (simple orthographic projection)
    obj_pts = object_pts.copy()
    if obj_pts is not None:
        # Normalize to pixel range
        obj_min, obj_max = obj_pts.min(axis=0), obj_pts.max(axis=0)
        obj_range = obj_max - obj_min + 1e-6
        
        # Project to 2D (use X-Y plane, ignore Z)
        pts_2d = np.zeros((obj_pts.shape[0], 2))
        pts_2d[:, 0] = (obj_pts[:, 0] - obj_min[0]) / obj_range[0] * (w - 50) + 25
        pts_2d[:, 1] = (obj_pts[:, 1] - obj_min[1]) / obj_range[1] * (h - 50) + 25
        
        # Draw object points (blue)
        for pt in pts_2d:
            cv2.circle(img, tuple(pt.astype(int)), 3, (255, 0, 0), -1)
    
    # Draw controller points (red)
    if controller_pts is not None:
        ctrl_pts = controller_pts.copy()
        ctrl_min, ctrl_max = ctrl_pts.min(axis=0), ctrl_pts.max(axis=0)
        ctrl_range = ctrl_max - ctrl_min + 1e-6
        
        pts_2d = np.zeros((ctrl_pts.shape[0], 2))
        pts_2d[:, 0] = (ctrl_pts[:, 0] - ctrl_min[0]) / ctrl_range[0] * (w - 50) + 25
        pts_2d[:, 1] = (ctrl_pts[:, 1] - ctrl_min[1]) / ctrl_range[1] * (h - 50) + 25
        
        for pt in pts_2d:
            cv2.circle(img, tuple(pt.astype(int)), 3, (0, 0, 255), -1)
    
    # Add title
    cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Compare two tracking runs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pkl1", help="Path to first pkl file")
    group.add_argument("--base_path", help="Base path for case-based loading")
    
    parser.add_argument("--pkl2", help="Path to second pkl file")
    parser.add_argument("--case1", help="Case name for first run")
    parser.add_argument("--case2", help="Case name for second run")
    parser.add_argument("--output", default="track_comparison", help="Output directory")
    parser.add_argument("--max_frames", type=int, default=-1)
    
    args = parser.parse_args()
    
    # Determine pkl paths
    if args.pkl1:
        pkl1, pkl2 = Path(args.pkl1), Path(args.pkl2)
        case_suffix = f"{pkl1.parent.name}_vs_{pkl2.parent.name}"
    else:
        pkl1 = Path(args.base_path) / args.case1 / "final_data.pkl"
        pkl2 = Path(args.base_path) / args.case2 / "final_data.pkl"
        case_suffix = f"{args.case1}_vs_{args.case2}"
    
    # Create output directory
    output_dir = Path(args.output) / case_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading {pkl1}...")
    data1 = load_track_data(pkl1)
    print(f"Loading {pkl2}...")
    data2 = load_track_data(pkl2)
    
    # Compute metrics
    print("\n" + "="*60)
    print("OBJECT TRACKS COMPARISON")
    print("="*60)
    
    obj1, obj2 = data1['object_points'], data2['object_points']
    vis1 = data1.get('object_visibilities', np.ones((obj1.shape[0], obj1.shape[1])))
    vis2 = data2.get('object_visibilities', np.ones((obj2.shape[0], obj2.shape[1])))
    
    metrics_obj = compute_track_metrics(obj1, obj2, vis1, vis2)
    
    print(f"Points: {metrics_obj['num_points_1']} vs {metrics_obj['num_points_2']}")
    print(f"Center of mass MAE:           {metrics_obj['com_mae']:.4f} m")
    print(f"Center of mass RMSE:          {metrics_obj['com_rmse']:.4f} m")
    print(f"COM trajectory correlation:   {metrics_obj['com_corr']:.4f}")
    print(f"Velocity MAE:                 {metrics_obj['velocity_mae']:.4f} m/frame")
    print(f"Visibility count diff:        {metrics_obj['visibility_diff']:.1f} points")
    print(f"Visibility relative diff:     {metrics_obj['visibility_relative_diff']:.1f}%")
    print(f"Spatial spread diff:          {metrics_obj['spread_diff']:.4f} m")
    print(f"Spatial spread relative diff: {metrics_obj['spread_relative_diff']:.1f}%")
    print(f"Point density correlation:    {metrics_obj['density_corr']:.4f}")
    
    # Save object plots and metrics
    create_comparison_plots(metrics_obj, "Object", output_dir / "plots")
    csv_path, json_path = save_metrics_to_file(metrics_obj, "Object", output_dir / "metrics")
    print(f"\n✓ Saved object metrics to {csv_path} and {json_path}")
    
    if 'controller_points' in data1 and 'controller_points' in data2:
        print("\n" + "="*60)
        print("CONTROLLER TRACKS COMPARISON")
        print("="*60)
        
        ctrl1, ctrl2 = data1['controller_points'], data2['controller_points']
        vis1 = data1.get('controller_visibilities', np.ones((ctrl1.shape[0], ctrl1.shape[1])))
        vis2 = data2.get('controller_visibilities', np.ones((ctrl2.shape[0], ctrl2.shape[1])))
        
        metrics_ctrl = compute_track_metrics(ctrl1, ctrl2, vis1, vis2)
        
        print(f"Points: {metrics_ctrl['num_points_1']} vs {metrics_ctrl['num_points_2']}")
        print(f"Center of mass MAE:           {metrics_ctrl['com_mae']:.4f} m")
        print(f"Center of mass RMSE:          {metrics_ctrl['com_rmse']:.4f} m")
        print(f"COM trajectory correlation:   {metrics_ctrl['com_corr']:.4f}")
        print(f"Velocity MAE:                 {metrics_ctrl['velocity_mae']:.4f} m/frame")
        print(f"Visibility count diff:        {metrics_ctrl['visibility_diff']:.1f} points")
        print(f"Visibility relative diff:     {metrics_ctrl['visibility_relative_diff']:.1f}%")
        print(f"Spatial spread diff:          {metrics_ctrl['spread_diff']:.4f} m")
        print(f"Spatial spread relative diff: {metrics_ctrl['spread_relative_diff']:.1f}%")
        print(f"Point density correlation:    {metrics_ctrl['density_corr']:.4f}")
        
        # Save controller plots and metrics
        create_comparison_plots(metrics_ctrl, "Controller", output_dir / "plots")
        csv_path, json_path = save_metrics_to_file(metrics_ctrl, "Controller", output_dir / "metrics")
        print(f"\n✓ Saved controller metrics to {csv_path} and {json_path}")
    
    # Render comparison videos
    print("\n" + "="*60)
    video_output = output_dir / "comparison_sidebyside.mp4"
    render_comparison_video(data1, data2, video_output, args.max_frames)
    
    print("\n" + "="*60)
    overlay_output = output_dir / "comparison_overlay_ghosts.mp4"
    render_overlay_comparison_video(data1, data2, overlay_output, args.max_frames, history_frames=8)
    print("="*60)
    
    print(f"\n✓ All comparison results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
