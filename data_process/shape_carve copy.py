"""Generate a deterministic shape prior from object masks and depth maps.

This is a deterministic, non-generative alternative to the TRELLIS shape prior
in ``data_process/shape_prior.py``. The legacy module name and ``carve`` method
are retained for CLI compatibility. It writes the same contract that
``align.py`` consumes (``{base_path}/{case_name}/shape/object.glb``):

    {output_dir}/object.glb
    {output_dir}/object.ply
    {output_dir}/observed_points.ply
    {output_dir}/observed_points_filtered.ply
    {output_dir}/visualization.mp4   (only with --visualize)

Occupancy rule per voxel v:

    occupied(v) = inside the object mask in every camera
                  AND not in front of the observed surface in any camera
                  (plus every voxel containing an observed object point)

The visible boundary follows the observed depth (free space between a camera and
the object is carved away), while the hidden side is closed by the mask visual
hull, which makes the result a solid rather than an open shell.
"""

import json
import os
import pickle
import sys
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
import trimesh
from scipy import ndimage
from scipy.spatial import Delaunay, QhullError
from skimage import measure
from trimesh.intersections import slice_faces_plane
from trimesh.repair import fill_holes, fix_inversion, fix_normals
from trimesh.smoothing import filter_taubin

try:
    # shape_carve.py is launched directly by process_data.py, so data_process is
    # the script directory rather than an importable top-level package.
    from controller_labels import is_controller_label, parse_controller_names
except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_process.controller_labels import (
        is_controller_label,
        parse_controller_names,
    )


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to {base_path}/{case_name}/shape.",
    )
    parser.add_argument("--controller_names", type=str, default=None)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=["voxel", "poisson"],
        default="voxel",
        help="Surface reconstruction backend. Poisson uses the observed 3-D "
        "points and estimated normals; voxel uses space carving.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.004 / 3,
        help="Voxel edge length in metres (default: 1.33 mm); caps the finest "
        "representable detail.",
    )
    parser.add_argument(
        "--max_voxels",
        type=int,
        default=12_000_000,
        help="Safety cap on the occupancy grid size; voxel_size is grown to fit.",
    )
    parser.add_argument("--bbox_pad", type=float, default=0.02)
    parser.add_argument(
        "--outlier_voxel",
        type=float,
        default=None,
        help="Coarse voxel size for dropping sparse observed-point outliers; "
        "defaults to max(4 * --voxel_size, 0.016).",
    )
    parser.add_argument(
        "--mask_erode",
        type=int,
        default=3,
        help="Pixels to erode each object mask before carving. Trims the "
        "background/depth-discontinuity fringe that segmentation masks leave "
        "at the silhouette; raise it for a tighter mesh, lower it toward 0 if "
        "the masks are already exact.",
    )
    parser.add_argument(
        "--close_iters",
        type=int,
        default=0,
        help="26-connected morphological closing passes on the occupancy that "
        "seal voxel-thin tunnels and holes punched by depth noise. 0 disables.",
    )
    parser.add_argument(
        "--depth_margin",
        type=float,
        default=0.004,
        help="Voxels closer than this to the observed surface are not carved.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=None,
        help="Divide raw depth by this to reach metres. Integer depth defaults "
        "to 1000 (millimetres); float depth is assumed to be metres.",
    )
    parser.add_argument("--depth_max", type=float, default=2.0)
    parser.add_argument(
        "--poisson_depth",
        type=int,
        default=9,
        help="Octree depth for Poisson reconstruction; higher is finer and slower.",
    )
    parser.add_argument(
        "--poisson_density_percentile",
        type=float,
        default=0.0,
        help="Discard Poisson vertices below this density percentile before repair. "
        "0 preserves the closed Poisson surface (recommended).",
    )
    parser.add_argument(
        "--extract",
        choices=["isosurface", "blocky"],
        default="isosurface",
        help="isosurface: marching cubes on the occupancy's signed distance "
        "field (default; smooth sub-voxel surface with no staircase). blocky: "
        "exact voxel-boundary surface, Taubin-smoothed (visibly stepped).",
    )
    parser.add_argument(
        "--smooth_iters",
        type=int,
        default=10,
        help="Taubin smoothing iterations applied to the blocky surface.",
    )
    parser.add_argument(
        "--ground_z",
        type=float,
        default=0.0,
        help="Ground-plane height used when flattening the support side.",
    )
    parser.add_argument(
        "--flatten_ground",
        dest="flatten_ground",
        action="store_true",
        default=True,
        help="Flatten generated vertices above ground_z (default).",
    )
    parser.add_argument(
        "--no_flatten_ground",
        dest="flatten_ground",
        action="store_false",
        help="Disable default ground-plane flattening.",
    )
    parser.add_argument(
        "--allow_open",
        action="store_true",
        help="Write the mesh even if it is not a closed volume (default: fail).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write a turntable visualization.mp4 (best effort, needs OpenGL).",
    )
    return parser.parse_args()


def object_index(case_dir, cam, controller_names):
    info_path = f"{case_dir}/mask/mask_info_{cam}.json"
    with open(info_path, "r") as handle:
        data = json.load(handle)
    obj_idx = None
    for key, value in data.items():
        if not is_controller_label(value, controller_names):
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    if obj_idx is None:
        raise ValueError(f"No object detection found in {info_path}.")
    return obj_idx


def load_depth(path, depth_scale):
    raw = np.load(path)
    if np.issubdtype(raw.dtype, np.integer):
        scale = 1000.0 if depth_scale is None else depth_scale
        return raw.astype(np.float32) / scale
    return raw.astype(np.float32)


def backproject_masked(depth_m, mask, intrinsic, c2w, depth_max):
    ys, xs = np.nonzero(mask)
    z = depth_m[ys, xs]
    valid = (z > 0) & (z < depth_max)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    cam = np.stack(
        [
            (xs - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (ys - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ],
        axis=1,
    ).astype(np.float64)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def largest_cluster(points, coarse):
    """Drop sparse silhouette/depth outliers by keeping the largest
    6-connected component of a coarse voxelization."""
    if points.size == 0:
        return points
    low = points.min(axis=0)
    dims = np.floor((points.max(axis=0) - low) / coarse).astype(int) + 1
    indices = np.floor((points - low) / coarse).astype(int)
    grid = np.zeros(dims, dtype=bool)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(grid, structure=structure)
    if count <= 1:
        return points
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, count + 1))
    best = int(np.argmax(sizes)) + 1
    keep = labels[indices[:, 0], indices[:, 1], indices[:, 2]] == best
    return points[keep]


def carve_occupancy(dims, grid_origin, voxel, masks, depths, intrinsics, w2cs,
                    depth_margin):
    nx, ny, nz = dims
    inside_all = np.ones((nx, ny, nz), dtype=bool)
    free_any = np.zeros((nx, ny, nz), dtype=bool)

    # Bound peak memory by carving one slab of x-rows at a time.
    chunk = max(1, int(1_000_000 // max(1, ny * nz)))
    for i0 in range(0, nx, chunk):
        i1 = min(nx, i0 + chunk)
        x = grid_origin[0] + (np.arange(i0, i1) + 0.5) * voxel
        y = grid_origin[1] + (np.arange(ny) + 0.5) * voxel
        z = grid_origin[2] + (np.arange(nz) + 0.5) * voxel
        gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")
        centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        inside = np.ones(centers.shape[0], dtype=bool)
        free = np.zeros(centers.shape[0], dtype=bool)
        for cam in range(len(w2cs)):
            w2c = w2cs[cam]
            cam_pts = centers @ w2c[:3, :3].T + w2c[:3, 3]
            depth = cam_pts[:, 2]
            front = depth > 1e-6
            safe_depth = np.where(front, depth, 1.0)
            u = (intrinsics[cam][0, 0] * cam_pts[:, 0] / safe_depth
                 + intrinsics[cam][0, 2])
            v = (intrinsics[cam][1, 1] * cam_pts[:, 1] / safe_depth
                 + intrinsics[cam][1, 2])
            ui = np.floor(u).astype(np.int64)
            vi = np.floor(v).astype(np.int64)
            h, w = masks[cam].shape
            inb = front & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

            cam_inside = np.zeros(centers.shape[0], dtype=bool)
            cam_inside[inb] = masks[cam][vi[inb], ui[inb]]
            inside &= cam_inside

            observed = np.zeros(centers.shape[0], dtype=np.float32)
            observed[inb] = depths[cam][vi[inb], ui[inb]]
            free |= inb & (observed > 0) & (depth < observed - depth_margin)

        inside_all[i0:i1] = inside.reshape(i1 - i0, ny, nz)
        free_any[i0:i1] = free.reshape(i1 - i0, ny, nz)

    return inside_all & ~free_any


def manifoldize(occupied, max_iter=10):
    """Break in-plane diagonal-only voxel contacts so the extracted surface is
    an edge-manifold (every edge shared by exactly two faces), which the
    watertight check requires. An edge-only contact is a 2x2 block whose two
    diagonal cells are occupied and the other two are empty; filling either
    empty cell turns it into a face contact. Mutates and returns ``occupied``."""
    shape = occupied.shape

    def shift_forward(array, axis):
        out = np.zeros_like(array)
        destination = [slice(None)] * 3
        source = [slice(None)] * 3
        destination[axis] = slice(1, None)
        source[axis] = slice(None, -1)
        out[tuple(destination)] = array[tuple(source)]
        return out

    for _ in range(max_iter):
        changed = False
        for u, v in ((1, 2), (0, 2), (0, 1)):
            padded = np.pad(occupied, 1)

            def corner(du, dv):
                slices = [slice(1, 1 + shape[0]), slice(1, 1 + shape[1]),
                          slice(1, 1 + shape[2])]
                for axis, shift in ((u, du), (v, dv)):
                    slices[axis] = slice(shift + 1, shift + 1 + shape[axis])
                return padded[tuple(slices)]

            a, b = corner(0, 0), corner(0, 1)
            c, d = corner(1, 0), corner(1, 1)
            # Fill the empty neighbour for each diagonal-only arrangement.
            fill = shift_forward(a & d & ~b & ~c, v) | (b & c & ~a & ~d)
            if fill.any():
                occupied |= fill
                changed = True
        if not changed:
            break
    return occupied


def voxel_surface_mesh(occupied, grid_origin, voxel):
    """Exact voxel-boundary triangle mesh (outward, consistently wound)."""
    occp = np.pad(occupied, 1, constant_values=False)
    core = occp[1:-1, 1:-1, 1:-1]
    vertices = {}
    vertex_list = []
    faces = []

    def vid(corner):
        index = vertices.get(corner)
        if index is None:
            index = len(vertex_list)
            vertices[corner] = index
            vertex_list.append(corner)
        return index

    def add_quad(a, b, c, d):
        ia, ib, ic, id_ = vid(a), vid(b), vid(c), vid(d)
        faces.append((ia, ib, ic))
        faces.append((ia, ic, id_))

    neg_x = occp[:-2, 1:-1, 1:-1]
    for i, j, k in np.argwhere(core & ~neg_x).tolist():
        add_quad((i, j, k), (i, j, k + 1), (i, j + 1, k + 1), (i, j + 1, k))
    pos_x = occp[2:, 1:-1, 1:-1]
    for i, j, k in np.argwhere(core & ~pos_x).tolist():
        x = i + 1
        add_quad((x, j, k), (x, j + 1, k), (x, j + 1, k + 1), (x, j, k + 1))
    neg_y = occp[1:-1, :-2, 1:-1]
    for i, j, k in np.argwhere(core & ~neg_y).tolist():
        add_quad((i, j, k), (i + 1, j, k), (i + 1, j, k + 1), (i, j, k + 1))
    pos_y = occp[1:-1, 2:, 1:-1]
    for i, j, k in np.argwhere(core & ~pos_y).tolist():
        y = j + 1
        add_quad((i, y, k), (i, y, k + 1), (i + 1, y, k + 1), (i + 1, y, k))
    neg_z = occp[1:-1, 1:-1, :-2]
    for i, j, k in np.argwhere(core & ~neg_z).tolist():
        add_quad((i, j, k), (i, j + 1, k), (i + 1, j + 1, k), (i + 1, j, k))
    pos_z = occp[1:-1, 1:-1, 2:]
    for i, j, k in np.argwhere(core & ~pos_z).tolist():
        z = k + 1
        add_quad((i, j, z), (i + 1, j, z), (i + 1, j + 1, z), (i, j + 1, z))

    if not faces:
        raise RuntimeError("Carving produced an empty occupancy grid.")
    verts = grid_origin[None, :] + np.asarray(vertex_list, dtype=np.float64) * voxel
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )


def isosurface_mesh(occupied, grid_origin, voxel):
    """Smooth sub-voxel surface via marching cubes on the occupancy's signed
    distance field, avoiding the voxel staircase of a voxel-boundary mesh."""
    inside = ndimage.distance_transform_edt(occupied)
    outside = ndimage.distance_transform_edt(~occupied)
    sdf = (outside - inside) * voxel
    vertices, faces, _, _ = measure.marching_cubes(
        sdf, level=0.0, spacing=(voxel, voxel, voxel)
    )
    vertices = np.asarray(vertices, dtype=np.float64) + (grid_origin + 0.5 * voxel)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def poisson_mesh(points, voxel_size, depth, density_percentile):
    """Reconstruct a smooth closed surface from calibrated surface points."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud = point_cloud.voxel_down_sample(max(voxel_size, 1e-4))
    if len(point_cloud.points) < 32:
        raise RuntimeError("Poisson reconstruction needs at least 32 surface points.")

    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(4.0 * voxel_size, 0.01), max_nn=50
        )
    )
    point_cloud.orient_normals_consistent_tangent_plane(
        min(100, len(point_cloud.points) - 1)
    )
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=depth,
        scale=1.1,
        linear_fit=True,
    )

    densities = np.asarray(densities)
    if densities.size and density_percentile > 0:
        cutoff = np.percentile(densities, density_percentile)
        mesh.remove_vertices_by_mask(densities < cutoff)
    return mesh_to_trimesh(mesh)


def mesh_to_trimesh(mesh):
    triangles = np.asarray(mesh.triangles)
    if triangles.size == 0:
        raise RuntimeError("Surface extraction produced no triangles.")
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=triangles, process=False
    )


def flatten_mesh_to_ground(mesh, ground_z):
    """Clip above-ground geometry and cap the cut at the support plane.

    Vertex clamping collapses every intersecting triangle onto the plane and can
    remove its faces during repair, leaving an open mesh. This implementation
    uses trimesh's dependency-free face clipping and adds a planar cap from the
    resulting boundary, so the support side remains a closed volume.
    """
    plane_normal = np.array([0.0, 0.0, -1.0])
    mesh_max_z = float(np.max(mesh.vertices[:, 2]))
    if mesh_max_z <= ground_z + 1e-7:
        return mesh

    vertices, faces, _ = slice_faces_plane(
        mesh.vertices,
        mesh.faces,
        plane_normal=plane_normal,
        plane_origin=np.array([0.0, 0.0, ground_z]),
    )
    if len(faces) == 0:
        raise RuntimeError("Ground-plane clipping removed the reconstructed mesh.")

    # Clipping can create the same intersection vertex once per incident face;
    # merge those duplicates before extracting the cap boundary.
    clipped = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    clipped.merge_vertices()
    vertices, faces = clipped.vertices.copy(), clipped.faces.copy()

    # Find the boundary created by the cut. Every cap boundary edge is used by
    # exactly one retained face and lies on the horizontal support plane.
    edges = np.vstack(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    edges.sort(axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    on_ground = np.isclose(vertices[:, 2], ground_z, atol=1e-7)
    boundary_edges = boundary_edges[on_ground[boundary_edges].all(axis=1)]
    boundary_vertices = np.unique(boundary_edges)
    if len(boundary_vertices) < 3:
        raise RuntimeError("Ground-plane clipping produced no valid cap boundary.")

    # Triangulate a conservative support cap. Keeping all boundary vertices
    # ensures clipped boundary edges are represented in the cap.
    # Split the boundary graph into connected contact regions before
    # triangulating. This prevents one Delaunay hull from spanning the rope.
    adjacency = {int(index): set() for index in boundary_vertices}
    for a, b in boundary_edges:
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    components = []
    unvisited = set(adjacency)
    while unvisited:
        start = unvisited.pop()
        component = [start]
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbour in adjacency[current]:
                if neighbour in unvisited:
                    unvisited.remove(neighbour)
                    component.append(neighbour)
                    stack.append(neighbour)
        components.append(np.asarray(component, dtype=np.int64))

    cap_parts = []
    for component in components:
        if len(component) < 3:
            continue
        try:
            cap_faces = component[Delaunay(vertices[component, :2]).simplices]
        except QhullError:
            continue
        cap_points = vertices[cap_faces]
        normal_z = np.cross(
            cap_points[:, 1] - cap_points[:, 0],
            cap_points[:, 2] - cap_points[:, 0],
        )[:, 2]
        cap_faces[normal_z < 0] = cap_faces[normal_z < 0][:, [0, 2, 1]]
        cap_parts.append(cap_faces)
    if not cap_parts:
        raise RuntimeError("Ground-plane clipping produced no valid cap triangles.")
    faces = np.vstack([faces, *cap_parts])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def repair_and_check(mesh, allow_open):
    """Repair the generated prior and require a usable volume by default."""
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda item: len(item.faces))

    mesh.merge_vertices()
    # Use fan triangulation so larger support-plane boundaries are closed too.
    fill_holes(mesh, use_fan=True)
    fix_normals(mesh)
    fix_inversion(mesh)

    if not mesh.is_watertight:
        fill_holes(mesh, use_fan=True)
        fix_normals(mesh)

    # ``fill_holes`` can add the same cap triangle twice with opposite winding
    # when the reconstruction contains a nearly coplanar boundary.  Trimesh's
    # ``unique_faces`` only catches identical index order, so normalize each
    # face's vertex order before deduplicating.  Re-select the largest connected
    # component after hole filling as well; otherwise these tiny cap fragments
    # survive export and make an otherwise closed rope non-manifold.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    if len(mesh.faces):
        unoriented_faces = np.sort(np.asarray(mesh.faces), axis=1)
        _, keep = np.unique(unoriented_faces, axis=0, return_index=True)
        mesh.update_faces(np.sort(keep))
    mesh.remove_unreferenced_vertices()
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda item: len(item.faces))

    watertight = bool(mesh.is_watertight and mesh.is_winding_consistent)
    closed_volume = bool(
        mesh.is_volume and np.isfinite(mesh.volume) and abs(float(mesh.volume)) > 0
    )
    if not watertight or not closed_volume:
        if not allow_open:
            raise RuntimeError(
                "Reconstructed mesh is not a valid closed volume after repair "
                f"(watertight={mesh.is_watertight}, "
                f"winding_consistent={mesh.is_winding_consistent}, "
                f"is_volume={mesh.is_volume}, volume={mesh.volume}). "
                "Try different reconstruction settings or pass --allow_open."
            )
    return mesh, bool(watertight and closed_volume)


def render_turntable(mesh, path):
    try:
        import cv2

        render_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces),
        )
        render_mesh.compute_vertex_normals()
        render_mesh.paint_uniform_color([0.75, 0.3, 0.3])
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(render_mesh)
        first = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width = first.shape[:2]
        view_control = vis.get_view_control()
        frames = []
        for _ in range(180):
            view_control.rotate(10, 0)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frames.append(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        vis.destroy_window()
        video = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
        )
        for frame in frames:
            video.write(frame)
        video.release()
    except Exception as error:  # noqa: BLE001 - visualization is best effort
        print(f"[deterministic_shape_prior] Skipping visualization: {error}")


def main():
    args = parse_args()
    case_dir = f"{args.base_path}/{args.case_name}"
    output_dir = args.output_dir or f"{case_dir}/shape"
    os.makedirs(output_dir, exist_ok=True)
    controller_names = parse_controller_names(args.controller_names)

    with open(f"{case_dir}/metadata.json", "r") as handle:
        metadata = json.load(handle)
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float64)
    num_cam = len(intrinsics)
    with open(f"{case_dir}/calibrate.pkl", "rb") as handle:
        c2ws = pickle.load(handle)

    masks, depths, w2cs, observed = [], [], [], []
    for cam in range(num_cam):
        obj_idx = object_index(case_dir, cam, controller_names)
        mask = imageio.imread(f"{case_dir}/mask/{cam}/{obj_idx}/{args.frame}.png")
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask > 127
        if args.mask_erode > 0:
            mask = ndimage.binary_erosion(mask, iterations=args.mask_erode)
        depth = load_depth(
            f"{case_dir}/depth/{cam}/{args.frame}.npy", args.depth_scale
        )
        masks.append(mask)
        depths.append(depth)
        w2cs.append(np.linalg.inv(c2ws[cam]))
        observed.append(
            backproject_masked(depth, mask, intrinsics[cam], c2ws[cam], args.depth_max)
        )

    observed = np.concatenate([points for points in observed if points.size])
    if observed.size == 0:
        raise RuntimeError("No observed object points; check masks and depth.")

    coarse = args.outlier_voxel or max(4 * args.voxel_size, 0.016)
    observed_raw = observed
    observed = largest_cluster(observed, coarse)
    print(
        "[deterministic_shape_prior] observed object points after outlier "
        f"filtering: {observed.shape[0]}"
    )

    low = observed.min(axis=0) - args.bbox_pad
    high = observed.max(axis=0) + args.bbox_pad
    extent = high - low

    voxel = args.voxel_size
    dims = np.ceil(extent / voxel).astype(int) + 3
    while int(np.prod(dims)) > args.max_voxels:
        voxel *= 1.25
        dims = np.ceil(extent / voxel).astype(int) + 3
    if not np.isclose(voxel, args.voxel_size):
        print(
            f"[deterministic_shape_prior] voxel_size grown to {voxel:.4f} m to fit "
            f"--max_voxels={args.max_voxels}"
        )

    if args.method == "poisson":
        mesh = poisson_mesh(
            observed,
            voxel,
            args.poisson_depth,
            args.poisson_density_percentile,
        )
    else:
        grid_origin = low - voxel
        dims = tuple(int(d) for d in dims)
        print(
            f"[deterministic_shape_prior] grid dims={dims} voxel={voxel:.4f} m"
        )

        occupied = carve_occupancy(
            dims, grid_origin, voxel, masks, depths, intrinsics, w2cs, args.depth_margin
        )

        # Force-occupy every voxel containing an observed point so the mesh is
        # guaranteed to enclose the shell, then keep a one-voxel empty border.
        indices = np.floor((observed - grid_origin) / voxel).astype(int)
        valid = np.all(indices >= 0, axis=1) & np.all(
            indices < np.asarray(dims), axis=1
        )
        indices = indices[valid]
        occupied[indices[:, 0], indices[:, 1], indices[:, 2]] = True
        # Depth noise and mask edges punch voxel-thin tunnels through the solid;
        # closing is opt-in because it changes measured dimensions.
        if args.close_iters > 0:
            occupied = ndimage.binary_closing(
                occupied,
                structure=ndimage.generate_binary_structure(3, 3),
                iterations=args.close_iters,
            )
        occupied = ndimage.binary_fill_holes(occupied)
        occupied = manifoldize(occupied)
        for axis in range(3):
            occupied = occupied.swapaxes(0, axis)
            occupied[0] = False
            occupied[-1] = False
            occupied = occupied.swapaxes(0, axis)

        if not occupied.any():
            raise RuntimeError("Carving produced no occupied voxels.")

        if args.extract == "isosurface":
            mesh = isosurface_mesh(occupied, grid_origin, voxel)
        else:
            mesh = mesh_to_trimesh(voxel_surface_mesh(occupied, grid_origin, voxel))
            if args.smooth_iters > 0:
                filter_taubin(mesh, iterations=args.smooth_iters)

    if args.flatten_ground:
        # Clip and cap instead of clamping vertices, which preserves a closed
        # volume when the reconstructed surface crosses the support plane.
        mesh = flatten_mesh_to_ground(mesh, args.ground_z)

    mesh, watertight = repair_and_check(mesh, args.allow_open)

    mesh.export(f"{output_dir}/object.glb")
    mesh.export(f"{output_dir}/object.ply")
    observed_pcd = o3d.geometry.PointCloud()
    observed_pcd.points = o3d.utility.Vector3dVector(observed_raw)
    o3d.io.write_point_cloud(f"{output_dir}/observed_points.ply", observed_pcd)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(observed)
    o3d.io.write_point_cloud(
        f"{output_dir}/observed_points_filtered.ply", filtered_pcd
    )
    if args.visualize:
        render_turntable(mesh, f"{output_dir}/visualization.mp4")

    extents = mesh.bounds[1] - mesh.bounds[0]
    print(
        f"[deterministic_shape_prior] wrote {output_dir}/object.glb  "
        f"faces={len(mesh.faces)} watertight={watertight} "
        f"euler={mesh.euler_number} volume={abs(float(mesh.volume)):.6f} m^3 "
        f"extents={np.round(extents, 4).tolist()}"
    )


if __name__ == "__main__":
    main()
