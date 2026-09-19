"""Overlay simulated controller points on the corresponding inference video.

The inference renderer already produces the useful object video. This script keeps
that video as the main view and adds a synchronized 3D inset showing the simulated
object and controller points, including controller motion trails.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qqtt.utils.controller_collider import hollow_controller_points


def _load_array(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_camera_config(data_dir: Path) -> tuple[float, int, int, list[tuple[np.ndarray, np.ndarray]] | None]:
    """Load capture settings and calibrated world-to-camera views."""
    metadata_path = data_dir / "metadata.json"
    fps = 30.0
    width, height = 848, 480
    camera_views = None
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if "WH" in metadata:
            width, height = map(int, metadata["WH"])
        fps = float(metadata.get("fps", metadata.get("FPS", fps)))
        intrinsics = metadata.get("intrinsics")
        calibration_path = data_dir / "calibrate.pkl"
        if intrinsics is not None and calibration_path.is_file():
            with calibration_path.open("rb") as handle:
                c2ws = pickle.load(handle)
            camera_views = [
                (np.asarray(intrinsic), np.linalg.inv(np.asarray(c2w)))
                for intrinsic, c2w in zip(intrinsics, c2ws)
            ]
    return fps, width, height, camera_views


def _render_inset(
    object_frame: np.ndarray,
    controller_frame: np.ndarray,
    object_trajectory: np.ndarray,
    controller_trajectory: np.ndarray,
    frame_index: int,
    width: int,
    height: int,
    max_object_points: int,
    max_controller_points: int,
    trail_length: int,
) -> np.ndarray:
    """Render a compact 3D state view as an RGB image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = plt.Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    canvas = FigureCanvasAgg(figure)

    object_points = object_frame[:: max(1, len(object_frame) // max_object_points)]
    controller_points = controller_frame[
        :: max(1, len(controller_frame) // max_controller_points)
    ]
    axis.scatter(
        object_points[:, 0],
        object_points[:, 1],
        object_points[:, 2],
        s=2,
        c="#d9e2ec",
        alpha=0.7,
        depthshade=False,
    )
    axis.scatter(
        controller_points[:, 0],
        controller_points[:, 1],
        controller_points[:, 2],
        s=9,
        c="#ff3b30",
        alpha=0.95,
        depthshade=False,
        label="controller",
    )

    trail_start = max(0, frame_index - trail_length + 1)
    trail = controller_trajectory[trail_start : frame_index + 1]
    if trail.shape[0] > 1:
        for point_index in range(0, trail.shape[1], max(1, trail.shape[1] // 80)):
            axis.plot(
                trail[:, point_index, 0],
                trail[:, point_index, 1],
                trail[:, point_index, 2],
                color="#ff9500",
                linewidth=0.7,
                alpha=0.45,
            )

    all_points = np.concatenate((object_trajectory, controller_trajectory), axis=1)
    bounds_min = np.nanmin(all_points.reshape(-1, 3), axis=0)
    bounds_max = np.nanmax(all_points.reshape(-1, 3), axis=0)
    center = (bounds_min + bounds_max) / 2.0
    radius = max(float(np.max(bounds_max - bounds_min)) / 2.0, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.view_init(elev=20, azim=-60)
    axis.set_title(f"simulation state  |  frame {frame_index}", fontsize=9, pad=2)
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=0.91)
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return image


def _create_open3d_video(
    output_path: Path,
    object_trajectory: np.ndarray,
    controller_trajectory: np.ndarray,
    fps: float,
    width: int,
    height: int,
    trail_length: int,
    camera_views: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Render a full-frame headless Open3D animation.

    This mirrors the repository's existing ``visualize_pc`` rendering path, but
    also renders dense controller samples as a separate red point cloud.
    """
    import open3d as o3d

    frame_count = min(object_trajectory.shape[0], controller_trajectory.shape[0])
    object_stride = max(1, object_trajectory.shape[1] // 6000)
    controller_stride = max(1, controller_trajectory.shape[1] // 6000)
    object_points = object_trajectory[:, ::object_stride, :]
    controller_points = controller_trajectory[:, ::controller_stride, :]

    if not camera_views:
        camera_views = [None]

    visualizers = []
    object_clouds = []
    controller_clouds = []
    for camera_view in camera_views:
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(visible=False, width=width, height=height)
        render_option = visualizer.get_render_option()
        render_option.background_color = np.asarray([0.025, 0.025, 0.025])
        render_option.point_size = 3.0

        object_cloud = o3d.geometry.PointCloud()
        object_cloud.points = o3d.utility.Vector3dVector(object_points[0])
        object_cloud.colors = o3d.utility.Vector3dVector(
            np.tile([0.62, 0.72, 0.86], (object_points.shape[1], 1))
        )
        controller_cloud = o3d.geometry.PointCloud()
        controller_cloud.points = o3d.utility.Vector3dVector(controller_points[0])
        controller_cloud.colors = o3d.utility.Vector3dVector(
            np.tile([1.0, 0.12, 0.08], (controller_points.shape[1], 1))
        )
        visualizer.add_geometry(object_cloud)
        visualizer.add_geometry(controller_cloud)

        view_control = visualizer.get_view_control()
        if camera_view is None:
            view_control.set_lookat(np.mean(object_points[0], axis=0))
            view_control.set_front([1.0, 0.0, -2.0])
            view_control.set_up([0.0, 0.0, -1.0])
            view_control.set_zoom(0.8)
        else:
            intrinsic, w2c = camera_view
            camera_parameters = o3d.camera.PinholeCameraParameters()
            camera_parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width, height, intrinsic
            )
            camera_parameters.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_parameters, allow_arbitrary=True
            )
        visualizers.append(visualizer)
        object_clouds.append(object_cloud)
        controller_clouds.append(controller_cloud)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * len(visualizers), height),
    )
    if not writer.isOpened():
        for visualizer in visualizers:
            visualizer.destroy_window()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    try:
        for frame_index in range(frame_count):
            panel_frames = []
            for visualizer, object_cloud, controller_cloud in zip(
                visualizers, object_clouds, controller_clouds
            ):
                object_cloud.points = o3d.utility.Vector3dVector(
                    object_points[frame_index]
                )
                controller_cloud.points = o3d.utility.Vector3dVector(
                    controller_points[frame_index]
                )
                visualizer.update_geometry(object_cloud)
                visualizer.update_geometry(controller_cloud)
                visualizer.poll_events()
                visualizer.update_renderer()
                panel_frames.append(
                    np.asarray(
                        visualizer.capture_screen_float_buffer(do_render=True)
                    )
                )
            frame = np.concatenate(
                [(panel * 255).astype(np.uint8) for panel in panel_frames], axis=1
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        for visualizer in visualizers:
            visualizer.destroy_window()

    if frame_count == 0:
        raise RuntimeError("No frames were written to the Open3D output video")
    print(f"Saved {frame_count} full-frame Open3D frames to {output_path}")


def create_overlay_video(
    data_path: Path,
    output_path: Path,
    inference_video: Path,
    trajectory_path: Path,
    key: str,
    hollow: bool,
    voxel_size: float,
    inset_scale: float,
    trail_length: int,
) -> None:
    with data_path.open("rb") as handle:
        data = pickle.load(handle)
    if key not in data:
        raise KeyError(f"Dataset does not contain {key}")

    controller_trajectory = np.asarray(data[key])
    if hollow:
        controller_trajectory = hollow_controller_points(controller_trajectory, voxel_size)
    if controller_trajectory.ndim != 3 or controller_trajectory.shape[-1] != 3:
        raise ValueError(
            f"{key} must have shape (frames, points, 3), got {controller_trajectory.shape}"
        )

    trajectory_includes_controllers = trajectory_path.exists()
    if trajectory_includes_controllers:
        simulated_vertices = _load_array(trajectory_path)
    elif "object_points" in data:
        # Older inference runs may have written inference.mp4 without
        # save_trajectory=True. final_data still gives us a synchronized object
        # view for the inset, while the inference video remains the main view.
        print(
            f"Warning: {trajectory_path} was not found; using final_data.pkl "
            "object_points for the inset."
        )
        simulated_vertices = np.asarray(data["object_points"])
    else:
        raise FileNotFoundError(
            f"Missing trajectory {trajectory_path}. Re-run inference with "
            "save_trajectory=True or provide --trajectory PATH."
        )
    if simulated_vertices.ndim != 3 or simulated_vertices.shape[-1] != 3:
        raise ValueError(
            f"Simulation trajectory must have shape (frames, vertices, 3), "
            f"got {simulated_vertices.shape}"
        )
    # The simulation state contains the sparse controller vertices. Dense points
    # are collider samples and are visualized from final_data independently.
    sparse_controller_points = np.asarray(data.get("controller_points", controller_trajectory))
    if sparse_controller_points.ndim != 3 or sparse_controller_points.shape[-1] != 3:
        raise ValueError("Dataset controller_points must have shape (frames, points, 3)")
    controller_count = sparse_controller_points.shape[1]
    if trajectory_includes_controllers:
        if simulated_vertices.shape[1] <= controller_count:
            raise ValueError("Simulation trajectory does not contain object vertices")
        object_trajectory = simulated_vertices[:, :-controller_count, :]
    else:
        # final_data.object_points contains object vertices only; unlike
        # inference.pkl it does not have sparse controller vertices appended.
        object_trajectory = simulated_vertices
    frame_count = min(simulated_vertices.shape[0], controller_trajectory.shape[0])

    fps, video_width, video_height, camera_views = _load_camera_config(
        data_path.parent
    )
    print(
        f"Generating {len(camera_views) if camera_views else 1} calibrated Open3D "
        f"camera panel(s), each {video_width}x{video_height}."
    )
    _create_open3d_video(
        output_path,
        object_trajectory[:frame_count],
        controller_trajectory[:frame_count],
        fps,
        video_width,
        video_height,
        trail_length,
        camera_views=camera_views,
    )
    return

    capture = None
    if inference_video.is_file():
        capture = cv2.VideoCapture(str(inference_video))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open inference video: {inference_video}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if video_width <= 0 or video_height <= 0:
            capture.release()
            raise RuntimeError(f"Inference video has invalid dimensions: {inference_video}")
    else:
        # When no rendered inference video exists, use the same full-frame
        # headless Open3D style used elsewhere in this repository instead of
        # placing a tiny visualization over a mostly black canvas.
        metadata_path = data_path.parent / "metadata.json"
        fps = 30.0
        video_width, video_height = 848, 480
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if "WH" in metadata:
                video_width, video_height = map(int, metadata["WH"])
            fps = float(metadata.get("fps", metadata.get("FPS", fps)))
        print(
            f"Warning: {inference_video} was not found; generating a full-frame "
            f"Open3D video at {video_width}x{video_height}."
        )
        _create_open3d_video(
            output_path,
            object_trajectory[:frame_count],
            controller_trajectory[:frame_count],
            fps,
            video_width,
            video_height,
            trail_length,
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (video_width, video_height),
    )
    if not writer.isOpened():
        if capture is not None:
            capture.release()
        raise RuntimeError(f"Unable to create output video: {output_path}")

    inset_width = min(video_width, max(160, int(video_width * inset_scale)))
    inset_height = min(video_height, max(120, int(video_height * inset_scale)))
    inset_margin = max(8, video_width // 100)
    frames_written = 0
    try:
        while frames_written < frame_count:
            if capture is None:
                success = True
                frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            else:
                success, frame = capture.read()
                if not success:
                    break
            inset_rgb = _render_inset(
                object_trajectory[frames_written],
                controller_trajectory[frames_written],
                object_trajectory[:frame_count],
                controller_trajectory[:frame_count],
                frames_written,
                inset_width,
                inset_height,
                max_object_points=2500,
                max_controller_points=2500,
                trail_length=trail_length,
            )
            inset = cv2.cvtColor(inset_rgb, cv2.COLOR_RGB2BGR)
            x0 = video_width - inset_width - inset_margin
            y0 = inset_margin
            roi = frame[y0 : y0 + inset_height, x0 : x0 + inset_width]
            if roi.shape[:2] != inset.shape[:2]:
                inset = cv2.resize(inset, (roi.shape[1], roi.shape[0]))
            cv2.addWeighted(inset, 0.88, roi, 0.12, 0, roi)
            cv2.rectangle(
                frame,
                (x0 - 1, y0 - 1),
                (x0 + inset_width, y0 + inset_height),
                (255, 255, 255),
                2,
            )
            writer.write(frame)
            frames_written += 1
    finally:
        if capture is not None:
            capture.release()
        writer.release()

    if frames_written == 0:
        raise RuntimeError("No frames were written to the output video")
    print(f"Saved {frames_written} overlay frames to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render synchronized controller points from inference.pkl in calibrated Open3D camera panels"
    )
    parser.add_argument("data_path", type=Path, help="Path to final_data.pkl")
    parser.add_argument("output_path", type=Path, help="Output MP4 path")
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Simulated vertex trajectory (default: <data dir>/inference.pkl)",
    )
    parser.add_argument("--dense", action="store_true", help="Use dense controller points")
    parser.add_argument("--hollow", action="store_true", help="Keep only the dense shell")
    parser.add_argument("--voxel-size", type=float, default=0.003)
    parser.add_argument(
        "--trail-length",
        type=int,
        default=20,
        help="Number of previous frames shown as controller trails",
    )
    args = parser.parse_args()

    if args.trail_length < 0:
        parser.error("--trail-length must be non-negative")

    data_dir = args.data_path.parent
    # Kept as an internal compatibility value; rendering now uses Open3D
    # directly and does not depend on an inference.mp4 background.
    inference_video = data_dir / "inference.mp4"
    trajectory = args.trajectory or data_dir / "inference.pkl"
    key = "controller_points_dense" if args.dense else "controller_points"
    create_overlay_video(
        args.data_path,
        args.output_path,
        inference_video,
        trajectory,
        key,
        args.hollow,
        args.voxel_size,
        0.34,
        args.trail_length,
    )


if __name__ == "__main__":
    main()
