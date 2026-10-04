# FIlter the tracking based on the object and controller mask, filter the track based on the neighbour motion
# Get the nearest controller points that are valid across all frames

import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import glob
import json
import pickle
import matplotlib.pyplot as plt
import sys
from argparse import ArgumentParser

try:
    from qqtt.utils.controller_collider import (
        count_visible_controller_frames,
        controller_depth_consistency,
        fill_occluded_controller_points,
    )
except ModuleNotFoundError:
    # data_process_track.py is launched directly by process_data.py, so the
    # repository root is not on sys.path.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qqtt.utils.controller_collider import (
        count_visible_controller_frames,
        controller_depth_consistency,
        fill_occluded_controller_points,
    )

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--no_visualize", action="store_true")
parser.add_argument(
    "--controller_point_count",
    type=int,
    default=30,
    help="Number of sparse controller points retained for legacy consumers.",
)
parser.add_argument(
    "--controller_visibility_threshold",
    type=float,
    default=0.7,
    help="Minimum fraction of frames a controller point must be visible in.",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


# Based on the valid mask, filter out the bad tracking data
def filter_track(track_path, pcd_path, mask_path, frame_num, num_cam):
    with open(f"{mask_path}/processed_masks.pkl", "rb") as f:
        processed_masks = pickle.load(f)

    # Filter out the points not valid in the first frame
    object_points = []
    object_colors = []
    object_visibilities = []
    controller_points = []
    controller_colors = []
    controller_visibilities = []
    for i in range(num_cam):
        current_track_data = np.load(f"{track_path}/{i}.npz")
        # Filter out the track data
        tracks = current_track_data["tracks"]
        tracks = np.round(tracks).astype(int)
        visibility = current_track_data["visibility"]
        if tracks.shape[0] != frame_num:
            raise ValueError(
                f"camera {i} was tracked over {tracks.shape[0]} frames but the "
                f"capture has {frame_num}; re-run the tracking stages for this "
                "case so the tracked video matches the current frames"
            )
        num_points = np.shape(tracks)[1]

        # Locate the track points in the object mask of the first frame
        object_mask = processed_masks[0][i]["object"]
        track_object_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                track_object_idx[j] = object_mask[tracks[0, j, 0], tracks[0, j, 1]]
        # Locate the controller points in the controller mask of the first frame
        controller_mask = processed_masks[0][i]["controller"]
        track_controller_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                track_controller_idx[j] = controller_mask[
                    tracks[0, j, 0], tracks[0, j, 1]
                ]

        # Filter out bad tracking in other frames
        for frame_idx in range(1, frame_num):
            # Filter based on object_mask
            object_mask = processed_masks[frame_idx][i]["object"]
            for j in range(num_points):
                try:
                    if track_object_idx[j] == 1 and visibility[frame_idx, j] == 1:
                        if not object_mask[
                            tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                        ]:
                            visibility[frame_idx, j] = 0
                except:
                    # Sometimes the track coordinate is out of image
                    visibility[frame_idx, j] = 0
            # Filter based on controller_mask
            controller_mask = processed_masks[frame_idx][i]["controller"]
            for j in range(num_points):
                if track_controller_idx[j] == 1 and visibility[frame_idx, j] == 1:
                    if not controller_mask[
                        tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                    ]:
                        visibility[frame_idx, j] = 0

        # Get the track point cloud
        track_points = np.zeros((frame_num, num_points, 3))
        track_colors = np.zeros((frame_num, num_points, 3))
        for frame_idx in range(frame_num):
            data = np.load(f"{pcd_path}/{frame_idx}.npz")
            points = data["points"]
            colors = data["colors"]

            track_points[frame_idx][np.where(visibility[frame_idx])] = points[i][
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 0],
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 1],
            ]
            track_colors[frame_idx][np.where(visibility[frame_idx])] = colors[i][
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 0],
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 1],
            ]

        object_points.append(track_points[:, np.where(track_object_idx)[0], :])
        object_colors.append(track_colors[:, np.where(track_object_idx)[0], :])
        object_visibilities.append(visibility[:, np.where(track_object_idx)[0]])
        controller_points.append(track_points[:, np.where(track_controller_idx)[0], :])
        controller_colors.append(track_colors[:, np.where(track_controller_idx)[0], :])
        controller_visibilities.append(visibility[:, np.where(track_controller_idx)[0]])

    object_points = np.concatenate(object_points, axis=1)
    object_colors = np.concatenate(object_colors, axis=1)
    object_visibilities = np.concatenate(object_visibilities, axis=1)
    controller_points = np.concatenate(controller_points, axis=1)
    controller_colors = np.concatenate(controller_colors, axis=1)
    controller_visibilities = np.concatenate(controller_visibilities, axis=1)

    track_data = {}
    track_data["object_points"] = object_points
    track_data["object_colors"] = object_colors
    track_data["object_visibilities"] = object_visibilities
    track_data["controller_points"] = controller_points
    track_data["controller_colors"] = controller_colors
    track_data["controller_visibilities"] = controller_visibilities

    return track_data


def filter_motion(track_data, neighbor_dist=0.01, visualize=True):
    # Calculate the motion of each point
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions = np.zeros_like(object_points)
    object_motions[:-1] = object_points[1:] - object_points[:-1]
    object_motions_valid = np.zeros_like(object_visibilities)
    object_motions_valid[:-1] = np.logical_and(
        object_visibilities[:-1], object_visibilities[1:]
    )

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    num_frames = object_points.shape[0]
    num_points = object_points.shape[1]

    vis = None
    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
    for i in tqdm(range(num_frames - 1)):
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points[i])
        pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        # modified_points = []
        # new_points = []
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):
            if object_motions_valid[i, j] == 0:
                continue
            # Get the neighbors within neighbor_dist
            [k, idx, _] = kdtree.search_radius_vector_3d(
                object_points[i, j], neighbor_dist
            )
            neighbors = [index for index in idx if object_motions_valid[i, index] == 1]
            if len(neighbors) < 5:
                object_motions_valid[i, j] = 0
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])
            motion_diff = np.linalg.norm(
                object_motions[i, j] - object_motions[i, neighbors], axis=1
            )
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                object_motions_valid[i, j] = 0
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])

        motion_pcd = o3d.geometry.PointCloud()
        motion_pcd.points = o3d.utility.Vector3dVector(
            object_points[i][np.where(object_motions_valid[i])]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            object_colors[i][np.where(object_motions_valid[i])]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_motions_valid[i])]
        )

        # modified_pcd = o3d.geometry.PointCloud()
        # modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
        # modified_pcd.colors = o3d.utility.Vector3dVector(
        #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
        # )

        # new_pcd = o3d.geometry.PointCloud()
        # new_pcd.points = o3d.utility.Vector3dVector(new_points)
        # new_pcd.colors = o3d.utility.Vector3dVector(
        #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
        # )
        if visualize and i == 0:
            render_motion_pcd = motion_pcd
            # render_modified_pcd = modified_pcd
            # render_new_pcd = new_pcd
            vis.add_geometry(render_motion_pcd)
            # vis.add_geometry(render_modified_pcd)
            # vis.add_geometry(render_new_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        elif visualize:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)
            # render_modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
            # render_modified_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
            # )
            # render_new_pcd.points = o3d.utility.Vector3dVector(new_points)
            # render_new_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
            # )
            vis.update_geometry(render_motion_pcd)
            # vis.update_geometry(render_modified_pcd)
            # vis.update_geometry(render_new_pcd)
            vis.poll_events()
            vis.update_renderer()
        # modified_num = len(modified_points)
        # print(f"Object Frame {i}: {modified_num} points are modified")

    track_data["object_motions_valid"] = object_motions_valid

    controller_points = track_data["controller_points"]
    controller_colors = track_data["controller_colors"]
    controller_visibilities = track_data["controller_visibilities"]
    controller_motions = np.zeros_like(controller_points)
    controller_motions[:-1] = controller_points[1:] - controller_points[:-1]
    controller_motions_valid = np.zeros_like(controller_visibilities)
    controller_motions_valid[:-1] = np.logical_and(
        controller_visibilities[:-1], controller_visibilities[1:]
    )
    num_points = controller_points.shape[1]
    # Filter all points that disappear in the sequence
    mask = np.prod(controller_visibilities, axis=0)

    y_min, y_max = np.min(controller_points[0, :, 1]), np.max(
        controller_points[0, :, 1]
    )
    y_normalized = (controller_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in tqdm(range(num_frames - 1)):
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points[i])
        pcd.colors = o3d.utility.Vector3dVector(controller_colors[i])
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):
            if mask[j] == 0:
                controller_motions_valid[i, j] = 0
            if controller_motions_valid[i, j] == 0:
                continue
            # Get the neighbors within neighbor_dist
            [k, idx, _] = kdtree.search_radius_vector_3d(
                controller_points[i, j], neighbor_dist
            )
            neighbors = [
                index for index in idx if controller_motions_valid[i, index] == 1
            ]
            if len(neighbors) < 5:
                controller_motions_valid[i, j] = 0
                mask[j] = 0

            motion_diff = np.linalg.norm(
                controller_motions[i, j] - controller_motions[i, neighbors], axis=1
            )
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                controller_motions_valid[i, j] = 0
                mask[j] = 0

        motion_pcd = o3d.geometry.PointCloud()
        motion_pcd.points = o3d.utility.Vector3dVector(
            controller_points[i][np.where(mask)]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            controller_colors[i][np.where(controller_motions_valid[i])]
        )

        if visualize and i == 0:
            render_motion_pcd = motion_pcd
            vis.add_geometry(render_motion_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        elif visualize:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)
            vis.update_geometry(render_motion_pcd)
            vis.poll_events()
            vis.update_renderer()

    if vis is not None:
        vis.destroy_window()
    track_data["controller_mask"] = mask
    return track_data


def get_final_track_data(
    track_data,
    controller_threhsold=0.01,
    controller_point_count=30,
    controller_visibility_threshold=0.7,
    masks=None,
    intrinsics=None,
    w2cs=None,
    depth_provider=None,
):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]
    controller_visibilities = track_data["controller_visibilities"]
    mask = track_data["controller_mask"]

    new_controller_points = controller_points[:, np.where(mask)[0], :]
    if controller_point_count < 1:
        raise ValueError("controller_point_count must be positive")
    if len(new_controller_points[0]) < controller_point_count:
        raise ValueError(
            "Not enough valid controller tracks for sparse sampling: "
            f"required {controller_point_count}, found {len(new_controller_points[0])}"
        )

    # Dense collider: points visible in most frames, with occluded frames filled.
    keep, min_visible = count_visible_controller_frames(
        controller_visibilities, controller_visibility_threshold
    )
    dense_visibilities = controller_visibilities[:, np.where(keep)[0]]
    dense_controller_points, dropped = fill_occluded_controller_points(
        controller_points[:, np.where(keep)[0], :], dense_visibilities
    )

    # Preserve a fixed point count by pruning whole trajectories. Depth checks
    # distinguish direct surface support from occlusion and free-space conflict.
    depth_rejected = 0
    if all(value is not None for value in (masks, intrinsics, w2cs, depth_provider)):
        keep_depth, support_fraction, conflict_fraction = controller_depth_consistency(
            dense_controller_points,
            masks,
            intrinsics,
            w2cs,
            depth_provider,
            depth_tolerance=0.004,
            min_support_fraction=0.7,
            max_conflict_fraction=0.05,
        )
        depth_rejected = int((~keep_depth).sum())
        dense_controller_points = dense_controller_points[:, keep_depth, :]
        print(
            f"Depth consistency: kept {int(keep_depth.sum())}/"
            f"{keep_depth.size} tracks; median support="
            f"{np.median(support_fraction):.1%}, median conflict="
            f"{np.median(conflict_fraction):.1%}"
        )

    dense_controller_points = dense_controller_points[~np.isnan(
        dense_controller_points).any(axis=2)].reshape(
        dense_controller_points.shape[0], -1, 3
    )
    print(
        f"Dense Controller Point Number: {dense_controller_points.shape[1]} "
        f"(visible in >= {min_visible}/{controller_visibilities.shape[0]} frames, "
        f"{depth_rejected} tracks rejected by depth consistency)"
    )

    # Do farthest point sampling on the valid controller points to select the
    # sparse controller representation used by existing PhysTwin consumers.
    valid_indices = np.arange(len(new_controller_points[0]))
    points_map = {}
    sample_points = []
    for i in valid_indices:
        points_map[tuple(new_controller_points[0, i])] = i
        sample_points.append(new_controller_points[0, i])
    sample_points = np.array(sample_points)
    sample_pcd = o3d.geometry.PointCloud()
    sample_pcd.points = o3d.utility.Vector3dVector(sample_points)
    fps_pcd = sample_pcd.farthest_point_down_sample(controller_point_count)
    final_indices = []
    for point in fps_pcd.points:
        final_indices.append(points_map[tuple(point)])

    print(f"Controller Point Number: {len(final_indices)}")
    print(f"Dense Controller Point Number: {dense_controller_points.shape[1]}")

    # Get the sparse controller points and preserve the dense points separately.
    nearest_controller_points = new_controller_points[:, final_indices]

    # object_pcd = o3d.geometry.PointCloud()
    # object_pcd.points = o3d.utility.Vector3dVector(valid_object_points)
    # object_pcd.colors = o3d.utility.Vector3dVector(
    #     object_colors[0][np.where(object_motions_valid[0])]
    # )
    # controller_meshes = []
    # for j in range(nearest_controller_points.shape[1]):
    #     origin = nearest_controller_points[0, j]
    #     origin_color = [1, 0, 0]
    #     controller_meshes.append(
    #         getSphereMesh(origin, color=origin_color, radius=0.005)
    #     )
    # o3d.visualization.draw_geometries([object_pcd])
    # o3d.visualization.draw_geometries([object_pcd] + controller_meshes)

    track_data.pop("controller_points")
    track_data.pop("controller_colors")
    track_data.pop("controller_visibilities")
    track_data["controller_points"] = nearest_controller_points
    track_data["controller_points_dense"] = dense_controller_points

    return track_data


def visualize_track(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_motions_valid[i])[0], :]
        )
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_motions_valid[i])[0]]
        )

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"
    track_path = f"{base_path}/{case_name}/cotracker"

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))
    # Take the capture length from the exported metadata instead of counting the
    # files in pcd: a shorter regeneration leaves stale frames behind, and
    # counting them would silently mix two captures into one case.
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        metadata = json.load(f)
        frame_num = metadata["frame_num"]

    with open(f"{mask_path}/processed_masks.pkl", "rb") as f:
        masks = pickle.load(f)
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        w2cs = np.array([np.linalg.inv(c2w) for c2w in pickle.load(f)])
    intrinsics = np.array(metadata["intrinsics"])

    # Filter the track data using the semantic mask of object and controller
    track_data = filter_track(track_path, pcd_path, mask_path, frame_num, num_cam)
    # Filter motion
    track_data = filter_motion(track_data, visualize=not args.no_visualize)
    # # Save the filtered track data
    # with open(f"test2.pkl", "wb") as f:
    #     pickle.dump(track_data, f)

    # with open(f"test2.pkl", "rb") as f:
    #     track_data = pickle.load(f)

    depth_cache = {}

    def load_depth_m(frame, camera):
        key = (frame, camera)
        if key not in depth_cache:
            depth = np.load(f"{base_path}/{case_name}/depth/{camera}/{frame}.npy")
            depth_cache[key] = depth.astype(np.float32) / 1000.0
        return depth_cache[key]

    track_data = get_final_track_data(
        track_data,
        controller_point_count=args.controller_point_count,
        controller_visibility_threshold=args.controller_visibility_threshold,
        masks=masks,
        intrinsics=intrinsics,
        w2cs=w2cs,
        depth_provider=load_depth_m,
    )

    with open(f"{base_path}/{case_name}/track_process_data.pkl", "wb") as f:
        pickle.dump(track_data, f)

    if not args.no_visualize:
        visualize_track(track_data)
