# FIlter the tracking based on the object and controller mask, filter the track based on the neighbour motion
# Get the nearest controller points that are valid across all frames

import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import glob
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
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
        print(processed_masks[0].keys())

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
        
        # Handle case where tracks have different number of frames than pcd data
        actual_frame_num = min(tracks.shape[0], frame_num)
        if tracks.shape[0] != frame_num:
            print(f"Warning: tracks have {tracks.shape[0]} frames but pcd has {frame_num} frames. Using {actual_frame_num} frames.")
            tracks = tracks[:actual_frame_num]
            visibility = visibility[:actual_frame_num]
        
        num_points = np.shape(tracks)[1]

        # Locate the track points in the object mask of the first frame
        object_mask = processed_masks[0][i]["object"]
        track_object_idx = np.zeros((num_points), dtype=int)
        print(object_mask.shape)
        print(tracks.shape)
        print(tracks[0])
        
        for j in range(num_points):
            if visibility[0, j] == 1:
                y, x = tracks[0, j, 0], tracks[0, j, 1]
                if 0 <= y < object_mask.shape[0] and 0 <= x < object_mask.shape[1]:
                    track_object_idx[j] = object_mask[y, x]
                else:
                    visibility[0, j] = 0
        # Locate the controller points in the controller mask of the first frame
        controller_mask = processed_masks[0][i]["controller"]
        track_controller_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                y, x = tracks[0, j, 0], tracks[0, j, 1]
                if 0 <= y < controller_mask.shape[0] and 0 <= x < controller_mask.shape[1]:
                    track_controller_idx[j] = controller_mask[y, x]
                else:
                    visibility[0, j] = 0

        # Filter out bad tracking in other frames
        for frame_idx in range(1, actual_frame_num):
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
        track_points = np.zeros((actual_frame_num, num_points, 3))
        track_colors = np.zeros((actual_frame_num, num_points, 3))
        for frame_idx in range(actual_frame_num):
            data = np.load(f"{pcd_path}/{frame_idx}.npz")
            points = data["points"]
            colors = data["colors"]

            visible_indices = np.where(visibility[frame_idx])[0]
            for idx in visible_indices:
                y, x = tracks[frame_idx, idx, 0], tracks[frame_idx, idx, 1]
                # Check bounds before indexing
                if 0 <= y < points[i].shape[0] and 0 <= x < points[i].shape[1]:
                    track_points[frame_idx, idx] = points[i][y, x]
                    track_colors[frame_idx, idx] = colors[i][y, x]
                else:
                    # Mark as not visible if out of bounds
                    visibility[frame_idx, idx] = 0

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


def filter_motion(track_data, neighbor_dist=0.01, min_neighbors=None):
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

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in tqdm(range(num_frames - 1)):
        # Check for NaNs or invalid shapes before creating point clouds
        if np.isnan(object_points[i]).any() or np.isnan(object_colors[i]).any():
            print(f"[ERROR] NaN detected in object_points or object_colors at frame {i}")
            print(f"object_points[{i}]:", object_points[i])
            print(f"object_colors[{i}]:", object_colors[i])
            raise ValueError(f"NaN detected in object_points or object_colors at frame {i}")
        if object_points[i].ndim != 2 or object_colors[i].ndim != 2:
            print(f"[ERROR] Invalid shape for object_points or object_colors at frame {i}")
            print(f"object_points[{i}].shape: {object_points[i].shape}")
            print(f"object_colors[{i}].shape: {object_colors[i].shape}")
            raise ValueError(f"Invalid shape for object_points or object_colors at frame {i}")
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
        if i == 0:
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
        else:
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

    vis.destroy_window()
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

    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=False)

    for i in tqdm(range(num_frames - 1)):
        # Check for NaNs or invalid shapes before creating point clouds
        if np.isnan(controller_points[i]).any() or np.isnan(controller_colors[i]).any():
            print(f"[ERROR] NaN detected in controller_points or controller_colors at frame {i}")
            print(f"controller_points[{i}]:", controller_points[i])
            print(f"controller_colors[{i}]:", controller_colors[i])
            raise ValueError(f"NaN detected in controller_points or controller_colors at frame {i}")
        if controller_points[i].ndim != 2 or controller_colors[i].ndim != 2:
            print(f"[ERROR] Invalid shape for controller_points or controller_colors at frame {i}")
            print(f"controller_points[{i}].shape: {controller_points[i].shape}")
            print(f"controller_colors[{i}].shape: {controller_colors[i].shape}")
            raise ValueError(f"Invalid shape for controller_points or controller_colors at frame {i}")
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
            # For sparse controller points (like in cloth lift), use fewer neighbor requirement
            # Use adaptive neighbor threshold: if min_neighbors specified, use it; otherwise compute adaptively
            if min_neighbors is None:
                min_neighbors_required = max(2, min(5, num_points // 2))
            else:
                min_neighbors_required = min_neighbors
            
            if len(neighbors) < min_neighbors_required:
                controller_motions_valid[i, j] = 0
                mask[j] = 0
                continue

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

        if i == 0:
            render_motion_pcd = motion_pcd
            # vis.add_geometry(render_motion_pcd)
            # # Adjust the viewpoint
            # view_control = vis.get_view_control()
            # view_control.set_front([1, 0, -2])
            # view_control.set_up([0, 0, -1])
            # view_control.set_zoom(1)
        else:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)
            # vis.update_geometry(render_motion_pcd)
            # vis.poll_events()
            # vis.update_renderer()

    track_data["controller_mask"] = mask
    return track_data


def get_final_track_data(track_data, controller_threhsold=0.01):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]
    mask = track_data["controller_mask"]

    new_controller_points = controller_points[:, np.where(mask)[0], :]
    print(f"Controller points after motion filtering: {len(new_controller_points[0])}")
    
    # If too few controller points remain, use all of them or relax the filtering
    if len(new_controller_points[0]) < 5:
        print(f"WARNING: Only {len(new_controller_points[0])} controller points after filtering!")
        print("Using all original controller points instead of motion-filtered ones")
        new_controller_points = controller_points
        print(f"Total controller points available: {len(new_controller_points[0])}")
    
    if len(new_controller_points[0]) < 30:
        # If we have fewer than 30, just use all of them without downsampling
        print(f"INFO: Only {len(new_controller_points[0])} controller points available, using all without downsampling")
        nearest_controller_points = new_controller_points
    else:
        # Do farthest point sampling on the valid controller points to select the final controller points
        valid_indices = np.arange(len(new_controller_points[0]))
        points_map = {}
        sample_points = []
        for i in valid_indices:
            points_map[tuple(new_controller_points[0, i])] = i
            sample_points.append(new_controller_points[0, i])
        sample_points = np.array(sample_points)
        sample_pcd = o3d.geometry.PointCloud()
        sample_pcd.points = o3d.utility.Vector3dVector(sample_points)
        fps_pcd = sample_pcd.farthest_point_down_sample(30)
        final_indices = []
        for point in fps_pcd.points:
            final_indices.append(points_map[tuple(point)])

        print(f"Final controller point count after FPS: {len(final_indices)}")

        # Get the nearest controller points and their colors
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
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))

    print(f"\n{'='*80}")
    print(f"Processing tracking data for case: {case_name}")
    print(f"Number of cameras: {num_cam}")
    print(f"Number of frames: {frame_num}")
    print(f"{'='*80}\n")

    # Filter the track data using the semantic mask of object and controller
    track_data = filter_track(track_path, pcd_path, mask_path, frame_num, num_cam)
    print(f"After initial filtering: {track_data['controller_points'].shape[1]} controller points\n")
    
    # Filter motion - with adaptive min_neighbors for sparse controller points
    track_data = filter_motion(track_data, neighbor_dist=0.01, min_neighbors=None)
    print(f"After motion filtering: {np.sum(track_data['controller_mask'])} controller points remain\n")
    
    # # Save the filtered track data
    # with open(f"test2.pkl", "wb") as f:
    #     pickle.dump(track_data, f)

    # with open(f"test2.pkl", "rb") as f:
    #     track_data = pickle.load(f)

    track_data = get_final_track_data(track_data)
    print(f"Final controller points: {track_data['controller_points'].shape[1]}\n")

    with open(f"{base_path}/{case_name}/track_process_data.pkl", "wb") as f:
        pickle.dump(track_data, f)
    
    print(f"Saved to {base_path}/{case_name}/track_process_data.pkl\n")

    visualize_track(track_data)
