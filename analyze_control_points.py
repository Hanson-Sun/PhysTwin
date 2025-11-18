"""
Diagnostic script to analyze the spatial relationship between control points and the object geometry.
This helps identify why some control points aren't being connected during initialization.
"""

import numpy as np
import torch
import pickle
from qqtt.utils import logger, cfg
import open3d as o3d
from scipy.spatial.distance import cdist

# Configuration - auto-detect data path
import glob
import os

# Find the first available final_data.pkl
data_paths = glob.glob("./data/different_types/*/final_data.pkl")
if not data_paths:
    raise FileNotFoundError("No final_data.pkl found in ./data/different_types/")

cfg.data_path = data_paths[0]
cfg.base_dir = "./experiments_optimization/test_analysis"
cfg.device = "cuda:0"

print(f"Using data from: {cfg.data_path}\n")

# Load the data
with open(cfg.data_path, "rb") as f:
    data = pickle.load(f)

object_points = data["object_points"]  # (n_frames, n_points, 3)
controller_points = data["controller_points"]  # (n_frames, n_control_points, 3)
other_surface_points = data["surface_points"]
interior_points = data["interior_points"]

# Build the structure_points (same as in real_data.py)
structure_points = np.concatenate(
    [object_points[0], other_surface_points, interior_points], axis=0
)

print("=" * 80)
print("CONTROL POINT ANALYSIS")
print("=" * 80)
print(f"\nStructure points shape: {structure_points.shape}")
print(f"  - Object points (frame 0): {object_points[0].shape}")
print(f"  - Surface points: {other_surface_points.shape}")
print(f"  - Interior points: {interior_points.shape}")
print(f"\nController points shape: {controller_points.shape}")
print(f"  - Number of frames: {controller_points.shape[0]}")
print(f"  - Number of control points: {controller_points.shape[1]}")

# Analyze frame 0 controller points (first frame used for initialization)
first_frame_controllers = controller_points[0]  # (n_control_points, 3)

# Build KDTree from structure points
structure_pcd = o3d.geometry.PointCloud()
structure_pcd.points = o3d.utility.Vector3dVector(structure_points)
pcd_tree = o3d.geometry.KDTreeFlann(structure_pcd)

print(f"\n{'='*80}")
print("DISTANCE ANALYSIS - Frame 0 Controller Points to Nearest Structure Points")
print(f"{'='*80}\n")

# Check distances for each control point
controller_radius = cfg.controller_radius
distances_per_control = []

for i, ctrl_point in enumerate(first_frame_controllers):
    # Find nearest point in structure
    [k, idx, dists] = pcd_tree.search_knn_vector_3d(ctrl_point, 1)
    nearest_dist = np.sqrt(dists[0])  # Open3D returns squared distances
    distances_per_control.append(nearest_dist)
    
    # Also check how many neighbors within radius
    [k_radius, idx_radius, _] = pcd_tree.search_hybrid_vector_3d(
        ctrl_point, controller_radius, 50
    )
    
    status = "✓ CONNECTED" if len(idx_radius) > 0 else "✗ DISCONNECTED"
    print(f"Control point {i}:")
    print(f"  Distance to nearest: {nearest_dist:.6f} m")
    print(f"  Controller radius:   {controller_radius:.6f} m")
    print(f"  Neighbors in radius: {len(idx_radius)}")
    print(f"  Status: {status}")
    print()

# Summary statistics
distances = np.array(distances_per_control)
print(f"{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}")
print(f"Controller radius: {controller_radius} m\n")
print(f"Distance statistics:")
print(f"  Min:    {distances.min():.6f} m")
print(f"  Max:    {distances.max():.6f} m")
print(f"  Mean:   {distances.mean():.6f} m")
print(f"  Median: {np.median(distances):.6f} m")
print(f"  Std:    {distances.std():.6f} m\n")

# Count how many are beyond radius
beyond_radius = np.sum(distances > controller_radius)
print(f"Control points beyond radius ({controller_radius}): {beyond_radius}/{len(distances)}")
print(f"Percentage disconnected: {100*beyond_radius/len(distances):.1f}%\n")

# Show which ones are problematic
print(f"{'='*80}")
print("PROBLEMATIC CONTROL POINTS (beyond radius)")
print(f"{'='*80}\n")

problematic_indices = np.where(distances > controller_radius)[0]
if len(problematic_indices) > 0:
    for i in problematic_indices:
        print(f"Control point {i}: {distances[i]:.6f} m (excess: {distances[i] - controller_radius:.6f} m)")
else:
    print("None - all control points are within radius!")

# Additional analysis: Check the spatial extent
print(f"\n{'='*80}")
print("SPATIAL ANALYSIS")
print(f"{'='*80}\n")

print("Structure points extent:")
print(f"  X: [{structure_points[:, 0].min():.3f}, {structure_points[:, 0].max():.3f}]")
print(f"  Y: [{structure_points[:, 1].min():.3f}, {structure_points[:, 1].max():.3f}]")
print(f"  Z: [{structure_points[:, 2].min():.3f}, {structure_points[:, 2].max():.3f}]")

print("\nController points extent (frame 0):")
print(f"  X: [{first_frame_controllers[:, 0].min():.3f}, {first_frame_controllers[:, 0].max():.3f}]")
print(f"  Y: [{first_frame_controllers[:, 1].min():.3f}, {first_frame_controllers[:, 1].max():.3f}]")
print(f"  Z: [{first_frame_controllers[:, 2].min():.3f}, {first_frame_controllers[:, 2].max():.3f}]")

# Check if control points are "outside" the object bounds
print("\nControl point positions relative to structure bounds:")
for i, ctrl_point in enumerate(first_frame_controllers):
    x_outside = (ctrl_point[0] < structure_points[:, 0].min() or 
                 ctrl_point[0] > structure_points[:, 0].max())
    y_outside = (ctrl_point[1] < structure_points[:, 1].min() or 
                 ctrl_point[1] > structure_points[:, 1].max())
    z_outside = (ctrl_point[2] < structure_points[:, 2].min() or 
                 ctrl_point[2] > structure_points[:, 2].max())
    
    outside_status = "OUTSIDE" if (x_outside or y_outside or z_outside) else "INSIDE"
    print(f"  Control point {i}: {outside_status}")
