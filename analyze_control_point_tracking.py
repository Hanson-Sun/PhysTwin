"""
Analyze the spatial distribution and tracking quality of control points.
This helps identify why control points are clustered instead of distributed.
"""

import numpy as np
import torch
import pickle
from qqtt.utils import logger, cfg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob

# Configuration - auto-detect data path
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

controller_points = data["controller_points"]  # (n_frames, n_control_points, 3)

print("=" * 80)
print("CONTROL POINT TRACKING ANALYSIS")
print("=" * 80)
print(f"\nController points shape: {controller_points.shape}")
print(f"  - Number of frames: {controller_points.shape[0]}")
print(f"  - Number of control points: {controller_points.shape[1]}")
print(f"  - Spatial dimension: {controller_points.shape[2]}\n")

# Analyze frame 0
frame_0 = controller_points[0]  # (n_control_points, 3)

print(f"{'='*80}")
print("FRAME 0 ANALYSIS - Control Point Positions")
print(f"{'='*80}\n")

for i, point in enumerate(frame_0):
    print(f"Control point {i:2d}: X={point[0]:7.4f}, Y={point[1]:7.4f}, Z={point[2]:7.4f}")

# Calculate spatial statistics
print(f"\n{'='*80}")
print("SPATIAL STATISTICS - Frame 0")
print(f"{'='*80}\n")

print("Per-dimension extent:")
print(f"  X: [{frame_0[:, 0].min():.4f}, {frame_0[:, 0].max():.4f}] range={frame_0[:, 0].max() - frame_0[:, 0].min():.4f}")
print(f"  Y: [{frame_0[:, 1].min():.4f}, {frame_0[:, 1].max():.4f}] range={frame_0[:, 1].max() - frame_0[:, 1].min():.4f}")
print(f"  Z: [{frame_0[:, 2].min():.4f}, {frame_0[:, 2].max():.4f}] range={frame_0[:, 2].max() - frame_0[:, 2].min():.4f}")

# Calculate centroid
centroid = frame_0.mean(axis=0)
print(f"\nCentroid: X={centroid[0]:.4f}, Y={centroid[1]:.4f}, Z={centroid[2]:.4f}")

# Calculate distances from centroid
distances_from_centroid = np.linalg.norm(frame_0 - centroid, axis=1)
print(f"\nDistances from centroid:")
print(f"  Min: {distances_from_centroid.min():.4f}")
print(f"  Max: {distances_from_centroid.max():.4f}")
print(f"  Mean: {distances_from_centroid.mean():.4f}")
print(f"  Std: {distances_from_centroid.std():.4f}")

# Check if points are actually separated or clustered
print(f"\n{'='*80}")
print("CLUSTERING ANALYSIS")
print(f"{'='*80}\n")

# Calculate pairwise distances
pairwise_distances = np.zeros((frame_0.shape[0], frame_0.shape[0]))
for i in range(frame_0.shape[0]):
    for j in range(i+1, frame_0.shape[0]):
        dist = np.linalg.norm(frame_0[i] - frame_0[j])
        pairwise_distances[i, j] = dist
        pairwise_distances[j, i] = dist

# Get statistics
unique_distances = pairwise_distances[pairwise_distances > 0]
print(f"Pairwise distances between control points:")
print(f"  Min: {unique_distances.min():.4f}")
print(f"  Max: {unique_distances.max():.4f}")
print(f"  Mean: {unique_distances.mean():.4f}")
print(f"  Median: {np.median(unique_distances):.4f}")
print(f"  Std: {unique_distances.std():.4f}")

# Count how many pairs are very close (< 0.01)
very_close_pairs = np.sum(unique_distances < 0.01)
close_pairs = np.sum(unique_distances < 0.05)
print(f"\nClose point pairs:")
print(f"  Very close (< 0.01m): {very_close_pairs}")
print(f"  Close (< 0.05m): {close_pairs}")

# Analyze motion across frames
print(f"\n{'='*80}")
print("MOTION ANALYSIS - Across All Frames")
print(f"{'='*80}\n")

# Calculate how much each control point moves between frames
frame_displacements = np.zeros((controller_points.shape[0] - 1, controller_points.shape[1]))
for frame_idx in range(controller_points.shape[0] - 1):
    frame_1 = controller_points[frame_idx]
    frame_2 = controller_points[frame_idx + 1]
    displacements = np.linalg.norm(frame_2 - frame_1, axis=1)
    frame_displacements[frame_idx] = displacements

# Statistics per control point
print("Per-control-point motion statistics:")
for i in range(controller_points.shape[1]):
    motion = frame_displacements[:, i]
    print(f"  Control point {i:2d}: "
          f"mean={motion.mean():.4f}, "
          f"max={motion.max():.4f}, "
          f"std={motion.std():.4f}")

# Overall motion statistics
all_motions = frame_displacements.flatten()
print(f"\nOverall motion statistics:")
print(f"  Mean: {all_motions.mean():.4f}")
print(f"  Max: {all_motions.max():.4f}")
print(f"  Std: {all_motions.std():.4f}")

# Check for static/barely moving points
static_threshold = 0.001
for i in range(controller_points.shape[1]):
    motion = frame_displacements[:, i]
    static_frames = np.sum(motion < static_threshold)
    if static_frames > controller_points.shape[0] * 0.5:  # More than 50% static
        print(f"\n⚠️  Control point {i} is mostly static ({static_frames}/{controller_points.shape[0]-1} frames)")

# Create a simple visualization
print(f"\n{'='*80}")
print("GENERATING VISUALIZATION")
print(f"{'='*80}\n")

fig = plt.figure(figsize=(15, 5))

# Plot 1: 3D scatter of control points
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(frame_0[:, 0], frame_0[:, 1], frame_0[:, 2], c=range(frame_0.shape[0]), cmap='hsv', s=100)
ax1.scatter(*centroid, c='red', s=200, marker='*', label='Centroid')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Frame 0 - 3D Distribution')
ax1.legend()

# Plot 2: XY projection
ax2 = fig.add_subplot(132)
ax2.scatter(frame_0[:, 0], frame_0[:, 1], c=range(frame_0.shape[0]), cmap='hsv', s=100)
ax2.scatter(centroid[0], centroid[1], c='red', s=200, marker='*', label='Centroid')
for i, point in enumerate(frame_0):
    ax2.annotate(str(i), (point[0], point[1]), fontsize=8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Frame 0 - XY Projection')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Motion over time (first 5 control points)
ax3 = fig.add_subplot(133)
for i in range(min(5, controller_points.shape[1])):
    motion = frame_displacements[:, i]
    ax3.plot(motion, label=f'CP {i}', linewidth=2)
ax3.set_xlabel('Frame')
ax3.set_ylabel('Displacement (m)')
ax3.set_title('Motion Over Time (first 5 control points)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_path = f"{cfg.base_dir}/control_point_analysis.png"
import os
os.makedirs(cfg.base_dir, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")
plt.close()

print(f"\n{'='*80}")
print("ANALYSIS SUMMARY")
print(f"{'='*80}\n")

if distances_from_centroid.std() < 0.01:
    print("⚠️  FINDING: Control points are very tightly clustered around centroid")
    print(f"    Standard deviation from centroid: {distances_from_centroid.std():.4f}m")
    print("    This suggests poor spatial distribution of tracking markers.")
    
if unique_distances.mean() < 0.05:
    print(f"\n⚠️  FINDING: Average pairwise distance is very small: {unique_distances.mean():.4f}m")
    print("    Points are too close together to effectively control different parts.")
    
if (frame_displacements < static_threshold).sum() / frame_displacements.size > 0.3:
    print(f"\n⚠️  FINDING: Many control points are mostly static.")
    print("    This could indicate tracking failure or marker occlusion.")
