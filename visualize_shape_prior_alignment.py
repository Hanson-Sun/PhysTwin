#!/usr/bin/env python3
"""
Visualization script to compare observation.ply, shape_prior.glb, and final_mesh.glb alignment.
This helps verify if the shape prior mesh is shifted/misaligned.

Usage:
    python visualize_shape_prior_alignment.py --gaussian_data_dir path/to/gaussian_data --source_data_dir path/to/source_data
    python visualize_shape_prior_alignment.py --gaussian_data_dir data/gaussian_data/cloth_double_bend --source_data_dir data/different_types/cloth_double_bend
"""

import argparse
import os
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path


def as_mesh(scene_or_mesh):
    """Convert a possible scene to a mesh."""
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = []
        for name, geometry in scene_or_mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        if len(meshes) > 1:
            combined_mesh = trimesh.util.concatenate(meshes)
        elif len(meshes) == 1:
            combined_mesh = meshes[0]
        else:
            raise ValueError("No valid meshes found in the GLB file")
        mesh = combined_mesh
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def load_observation_pcd(pcd_path):
    """Load observation point cloud from PLY file."""
    if not os.path.exists(pcd_path):
        print(f"Error: observation.ply not found at {pcd_path}")
        return None
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"✓ Loaded observation.ply: {len(pcd.points)} points")
    print(f"  Bounds: {np.asarray(pcd.points).min(axis=0)} to {np.asarray(pcd.points).max(axis=0)}")
    print(f"  Center: {np.asarray(pcd.points).mean(axis=0)}")
    return pcd


def load_shape_prior_mesh(mesh_path):
    """Load shape prior mesh from GLB file."""
    if not os.path.exists(mesh_path):
        print(f"Error: shape_prior.glb not found at {mesh_path}")
        return None
    
    try:
        scene = trimesh.load(mesh_path)
        mesh = as_mesh(scene)
        print(f"✓ Loaded shape_prior.glb: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"  Bounds: {mesh.bounds[0]} to {mesh.bounds[1]}")
        print(f"  Center of mass: {mesh.center_mass}")
        return mesh
    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None


def visualize_alignment(pcd, shape_prior_mesh, final_mesh):
    """Visualize point cloud and meshes in Open3D."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Shape Prior Alignment Check")
    
    # Add point cloud (colored in blue)
    if pcd is not None:
        pcd.paint_uniform_color([0.0, 0.5, 1.0])  # Light blue
        vis.add_geometry(pcd)
    
    # Add shape_prior mesh (colored in red)
    if shape_prior_mesh is not None:
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(shape_prior_mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(shape_prior_mesh.faces)
        mesh_o3d.paint_uniform_color([1.0, 0.0, 0.0])  # Red
        vis.add_geometry(mesh_o3d)
    
    # Add final_mesh (colored in green)
    if final_mesh is not None:
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(final_mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(final_mesh.faces)
        mesh_o3d.paint_uniform_color([0.0, 1.0, 0.0])  # Green
        vis.add_geometry(mesh_o3d)
    
    # Create a coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0]
    )
    vis.add_geometry(coord_frame)
    
    print("\n" + "="*60)
    print("VISUALIZATION GUIDE:")
    print("="*60)
    print("Blue points: observation.ply (tracked point cloud)")
    print("Red mesh: shape_prior.glb (shape prior from gaussian_data)")
    print("Green mesh: final_mesh.glb (original mesh from source_data)")
    print("RGB frame: World coordinate system origin (0, 0, 0)")
    print("\nControls:")
    print("  - Scroll: Zoom in/out")
    print("  - Right-click + drag: Rotate view")
    print("  - Left-click + drag: Pan view")
    print("  - Press 'Q' to exit")
    print("="*60 + "\n")
    
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize alignment between observation.ply, shape_prior.glb, and final_mesh.glb"
    )
    parser.add_argument(
        "--gaussian_data_dir",
        type=str,
        required=True,
        help="Path to gaussian_data directory containing observation.ply and shape_prior.glb"
    )
    parser.add_argument(
        "--source_data_dir",
        type=str,
        required=True,
        help="Path to source data directory containing shape/matching/final_mesh.glb"
    )
    
    args = parser.parse_args()
    gaussian_data_dir = args.gaussian_data_dir
    source_data_dir = args.source_data_dir
    
    if not os.path.isdir(gaussian_data_dir):
        print(f"Error: Gaussian data directory not found: {gaussian_data_dir}")
        return
    
    if not os.path.isdir(source_data_dir):
        print(f"Error: Source data directory not found: {source_data_dir}")
        return
    
    print("="*60)
    print("SHAPE PRIOR ALIGNMENT VISUALIZATION")
    print("="*60 + "\n")
    
    # Load files from both directories
    pcd_path = os.path.join(gaussian_data_dir, "observation.ply")
    shape_prior_path = os.path.join(gaussian_data_dir, "shape_prior.glb")
    final_mesh_path = os.path.join(source_data_dir, "shape/matching/final_mesh.glb")
    
    pcd = load_observation_pcd(pcd_path)
    shape_prior_mesh = load_shape_prior_mesh(shape_prior_path)
    final_mesh = load_shape_prior_mesh(final_mesh_path)
    
    if pcd and (shape_prior_mesh or final_mesh):
        print("\nDifference Analysis:")
        pcd_points = np.asarray(pcd.points)
        pcd_center = pcd_points.mean(axis=0)
        
        print(f"  PCD center: {pcd_center}")
        
        if shape_prior_mesh is not None:
            shape_prior_center = shape_prior_mesh.center_mass
            center_diff_sp = shape_prior_center - pcd_center
            print(f"  shape_prior.glb center: {shape_prior_center}")
            print(f"  Distance (PCD to shape_prior): {np.linalg.norm(center_diff_sp):.4f}")
            
            if np.linalg.norm(center_diff_sp) > 0.01:
                print(f"  ⚠️  shape_prior.glb is MISALIGNED by {center_diff_sp}")
            else:
                print(f"  ✓ shape_prior.glb is well-aligned")
        
        if final_mesh is not None:
            final_mesh_center = final_mesh.center_mass
            center_diff_fm = final_mesh_center - pcd_center
            print(f"  final_mesh.glb center: {final_mesh_center}")
            print(f"  Distance (PCD to final_mesh): {np.linalg.norm(center_diff_fm):.4f}")
            
            if np.linalg.norm(center_diff_fm) > 0.01:
                print(f"  ⚠️  final_mesh.glb is MISALIGNED by {center_diff_fm}")
            else:
                print(f"  ✓ final_mesh.glb is well-aligned")
        
        if shape_prior_mesh is not None and final_mesh is not None:
            mesh_diff = final_mesh.center_mass - shape_prior_mesh.center_mass
            print(f"\n  Difference between final_mesh and shape_prior: {np.linalg.norm(mesh_diff):.4f}")
            if np.linalg.norm(mesh_diff) > 0.001:
                print(f"  ⚠️  Meshes are NOT identical: {mesh_diff}")
            else:
                print(f"  ✓ Meshes are identical")
        
        print()
        visualize_alignment(pcd, shape_prior_mesh, final_mesh)
    else:
        print("Error: Could not load required data (need observation.ply and at least one mesh)")
        if pcd:
            print("\nOnly observation.ply loaded, visualizing point cloud...")
            vis = o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
