import sys
import open3d as o3d

if len(sys.argv) < 2:
    print("Usage: python visualize_gaussians.py /path/to/point_cloud.ply")
    sys.exit(1)

path = sys.argv[1]
pcd = o3d.io.read_point_cloud(path)
o3d.visualization.draw_geometries([pcd])