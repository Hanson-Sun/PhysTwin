import os
import pickle
import numpy as np
import open3d as o3d
import argparse

def visualize_final_object(case_name):
    """
    Visualize the final processed object from final_data.pkl using Open3D,
    including surface points, interior points, object points, and controller points.

    Args:
        case_name: The name of the case (e.g., 'cloth_double_bend')
    """
    path = f"data/different_types/{case_name}/final_data.pkl"

    with open(path, "rb") as f:
        data = pickle.load(f)

    object_points = data["object_points"]
    object_colors = data["object_colors"]
    surface_points = data["surface_points"]
    interior_points = data["interior_points"]
    controller_points = data["controller_points"]

    geometries = []

    # Use the first frame for static visualization
    points = object_points[0]
    colors = object_colors[0]

    # Normalize colors to 0-1 range if they appear to be 0-255
    if colors.max() > 1.0:
        colors = colors / 255.0

    # Create point cloud for object points
    if len(points) > 0:
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(points)
        obj_pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(obj_pcd)

    # Create point cloud for surface points (green)
    if len(surface_points) > 0:
        surf_pcd = o3d.geometry.PointCloud()
        surf_pcd.points = o3d.utility.Vector3dVector(surface_points)
        surf_pcd.paint_uniform_color([0, 1, 0])  # Green
        geometries.append(surf_pcd)

    # Create point cloud for interior points (blue)
    if len(interior_points) > 0:
        int_pcd = o3d.geometry.PointCloud()
        int_pcd.points = o3d.utility.Vector3dVector(interior_points)
        int_pcd.paint_uniform_color([0, 0, 1])  # Blue
        geometries.append(int_pcd)

    # Create spheres for controller points (red, bigger)
    for ctrl_point in controller_points[0]:
        ctrl_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        ctrl_sphere.translate(ctrl_point)
        ctrl_sphere.paint_uniform_color([1, 0, 0])  # Red
        geometries.append(ctrl_sphere)

    # Add coordinate frame for reference at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geometries.append(coord_frame)

    # Visualize
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize final object")
    parser.add_argument("--case_name", type=str, help="Name of the case")
    args = parser.parse_args()

    visualize_final_object(args.case_name)
