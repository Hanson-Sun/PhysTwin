"""
Plane and object visualization
"""

import json
import numpy as np
import open3d as o3d
import os
import pickle
from argparse import ArgumentParser


def visualize_planes(case_name):
    """
    Visualize object and detected planes.
    """
    geometries = []
    
    # Load object data
    data_path = f"./data/different_types/{case_name}/final_data.pkl"
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        obj_pts = data["object_points"][0]
        obj_colors = data["object_colors"][0]
        
        obj_pcd = o3d.geometry.PointCloud()
        obj_pcd.points = o3d.utility.Vector3dVector(obj_pts)
        if obj_colors.max() > 1.0:
            obj_colors = obj_colors / 255.0
        obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)
        geometries.append(obj_pcd)
        print(f"[VIS] Loaded object with {len(obj_pts)} points")
    
    # Load planes
    planes_path = f"./data/different_types/{case_name}/environment_planes.json"
    with open(planes_path, "r") as f:
        planes_data = json.load(f)
    
    plane_list = planes_data.get("planes", [])
    print(f"[VIS] Loaded {len(plane_list)} planes")
    
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    
    for i, plane in enumerate(plane_list):
        plane_eq = np.array(plane["equation"], dtype=np.float32)
        mesh = create_plane_quad(plane_eq, size=1.0)
        if mesh is not None:
            mesh.paint_uniform_color(colors[i % len(colors)])
            geometries.append(mesh)
    
    # Coordinate frame
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    geometries.append(coord)
    
    print(f"[VIS] Displaying {len(geometries)} objects...")
    o3d.visualization.draw_geometries(geometries)


def create_plane_quad(plane_eq, size=1.0):
    """
    Create dual-sided quad for plane.
    """
    a, b, c, d = plane_eq
    norm_sq = a*a + b*b + c*c
    
    if norm_sq < 1e-6:
        return None
    
    normal = np.array([a, b, c]) / np.sqrt(norm_sq)
    
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    
    plane_point = -d / norm_sq * normal
    
    # Four corners
    corners = [
        plane_point - u * size/2 - v * size/2,
        plane_point + u * size/2 - v * size/2,
        plane_point + u * size/2 + v * size/2,
        plane_point - u * size/2 + v * size/2
    ]
    vertices = np.array(corners + corners)  # Duplicate for both sides
    
    # Two triangles, each rendered twice (back and front)
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],  # Front
        [2, 1, 0], [3, 2, 0]   # Back (reversed)
    ])
    
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    return mesh

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--case_name", type=str, required=True, help="Case name")
    args = parser.parse_args()
    
    visualize_planes(args.case_name)
