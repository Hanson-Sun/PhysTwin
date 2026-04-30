#!/usr/bin/env python3
"""
Automate a single-case real-world visualization pipeline:

1) Convert an NPZ trajectory to inference.pkl
2) Render dynamic Gaussian outputs with the converted trajectory
3) Build overlay visualization videos on top of source video frames

This script is intended for data under scripts/real_world and similar NPZ inputs.
"""

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


DEFAULT_EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
DEFAULT_GS_INIT_OPT = "hybrid"
DEFAULT_LAMBDA_DEPTH = 0.001
DEFAULT_LAMBDA_NORMAL = 0.0
DEFAULT_LAMBDA_ANISO = 0.0
DEFAULT_LAMBDA_SEG = 1.0


def detect_workspace_root(script_file: Path) -> Path:
    """Find repository root by searching upward for core pipeline entry files."""
    markers = ["split_video.py", "gs_render_dynamics.py", "visualize_render_results.py"]

    script_file = script_file.resolve()
    search_roots = [script_file.parent, *script_file.parents]
    for candidate in search_roots:
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    # Best-effort fallback for common layout: <repo>/scripts/npz_to_inference_render.py
    return script_file.parents[1]


def run_cmd(cmd, cwd):
    printable = " ".join(str(x) for x in cmd)
    print(f"[Run] {printable}")
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def resolve_path(workspace_root: Path, path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (workspace_root / path).resolve()


def resolve_existing_path(
    workspace_root: Path,
    path_str: str,
    extra_roots=None,
) -> Path:
    """Resolve a path and prefer existing candidates for convenient short inputs."""
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [(workspace_root / path).resolve(), (Path.cwd() / path).resolve()]
    if extra_roots is not None:
        for root in extra_roots:
            if root is None:
                continue
            candidates.append((Path(root).resolve() / path).resolve())

    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(cand)

    for cand in unique_candidates:
        if cand.exists():
            return cand

    return unique_candidates[0]


def get_numeric_pngs(directory: Path):
    if not directory.exists():
        return []
    files = [p for p in directory.glob("*.png") if p.stem.isdigit()]
    files.sort(key=lambda p: int(p.stem))
    return files


def has_trained_gaussian_checkpoint(model_path: Path) -> bool:
    point_cloud_dir = model_path / "point_cloud"
    if not point_cloud_dir.exists() or not point_cloud_dir.is_dir():
        return False

    for child in point_cloud_dir.iterdir():
        if child.is_dir() and child.name.startswith("iteration_"):
            return True
    return False


def ensure_case_video(source_video: Path, target_video: Path, overwrite: bool):
    target_video.parent.mkdir(parents=True, exist_ok=True)

    if target_video.exists() and not overwrite:
        if source_video.resolve() != target_video.resolve():
            print(
                f"[Info] Target video exists, keeping it (use --overwrite-source-video to replace): {target_video}"
            )
        else:
            print(f"[Info] Source video already in place: {target_video}")
        return

    if source_video.resolve() != target_video.resolve():
        shutil.copy2(source_video, target_video)
        print(f"[Info] Copied source video to {target_video}")
    else:
        print(f"[Info] Source video already at target path: {target_video}")


def write_split_json(split_path: Path, frame_len: int):
    split = {
        "frame_len": int(frame_len),
        "train": [int(frame_len * 0.3), int(frame_len)],
        "test": [0, int(frame_len * 0.3)],
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f)
    print(f"[Info] Wrote split.json to {split_path}")


def select_object_id(mask_info_path: Path, controller_name: str):
    with open(mask_info_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    object_ids = []
    for k, v in data.items():
        if str(v).strip().lower() != controller_name.lower():
            object_ids.append(int(k))

    if len(object_ids) == 0:
        raise ValueError(
            f"No object ID found in {mask_info_path}. Controller name={controller_name}"
        )
    if len(object_ids) > 1:
        raise ValueError(
            f"Multiple object IDs found in {mask_info_path}: {object_ids}."
        )
    return object_ids[0]


def copy_object_masks_if_available(
    case_dir: Path,
    object_mask_case_dir: Path,
    num_views: int,
    controller_name: str,
):
    for view_idx in range(num_views):
        target_view_dir = object_mask_case_dir / "mask" / str(view_idx)
        target_view_dir.mkdir(parents=True, exist_ok=True)

        existing = list(target_view_dir.glob("*.png"))
        if existing:
            continue

        mask_info_path = case_dir / "mask" / f"mask_info_{view_idx}.json"
        if not mask_info_path.exists():
            print(f"[Warn] Missing mask info: {mask_info_path}")
            continue

        try:
            object_id = select_object_id(mask_info_path, controller_name)
        except Exception as exc:
            print(f"[Warn] Could not parse object ID for view {view_idx}: {exc}")
            continue

        source_object_dir = case_dir / "mask" / str(view_idx) / str(object_id)
        if not source_object_dir.exists():
            print(f"[Warn] Missing source object mask dir: {source_object_dir}")
            continue

        copied = 0
        for src in source_object_dir.glob("*.png"):
            shutil.copy2(src, target_view_dir / src.name)
            copied += 1

        print(
            f"[Info] Copied {copied} object masks for view {view_idx} -> {target_view_dir}"
        )


def ensure_empty_masks(
    case_dir: Path,
    object_mask_case_dir: Path,
    human_mask_case_dir: Path,
    num_views: int,
    frame_len: int,
):
    for view_idx in range(num_views):
        color_frame_dir = case_dir / "color" / str(view_idx)
        frame_files = get_numeric_pngs(color_frame_dir)
        if not frame_files:
            raise FileNotFoundError(
                f"No color frames found for view {view_idx}: {color_frame_dir}"
            )

        ref = cv2.imread(str(frame_files[0]), cv2.IMREAD_UNCHANGED)
        if ref is None:
            raise RuntimeError(f"Failed to read frame: {frame_files[0]}")
        h, w = ref.shape[0], ref.shape[1]
        zeros = np.zeros((h, w), dtype=np.uint8)

        object_view_dir = object_mask_case_dir / "mask" / str(view_idx)
        human_view_dir = human_mask_case_dir / "mask" / str(view_idx) / "0"
        object_view_dir.mkdir(parents=True, exist_ok=True)
        human_view_dir.mkdir(parents=True, exist_ok=True)

        created_object = 0
        created_human = 0

        for frame_idx in range(frame_len):
            obj_path = object_view_dir / f"{frame_idx}.png"
            hum_path = human_view_dir / f"{frame_idx}.png"

            if not obj_path.exists():
                cv2.imwrite(str(obj_path), zeros)
                created_object += 1

            if not hum_path.exists():
                cv2.imwrite(str(hum_path), zeros)
                created_human += 1

        print(
            f"[Info] Empty mask fallback for view {view_idx}: "
            f"object={created_object}, human={created_human}"
        )


def detect_fps(source_video: Path, fallback_fps: int = 30) -> int:
    try:
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            return fallback_fps
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps is None or fps <= 0:
            return fallback_fps
        fps_int = int(round(float(fps)))
        return fps_int if fps_int > 0 else fallback_fps
    except Exception:
        return fallback_fps


def normalize_shape_prior(value):
    if isinstance(value, bool):
        return "True" if value else "False"

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return "True"
    if text in {"false", "0", "no", "n", "f"}:
        return "False"
    raise ValueError(
        "shape_prior must be one of true/false/1/0/yes/no. "
        f"Got: {value}"
    )


def read_scene_metadata_from_config(data_config_path: Path, scene_name: str):
    with open(data_config_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip() != scene_name:
                continue
            if len(row) < 3:
                raise ValueError(
                    f"Malformed row for scene '{scene_name}' in {data_config_path}: {row}"
                )
            category = row[1].strip()
            shape_prior = normalize_shape_prior(row[2])
            return category, shape_prior

    raise KeyError(
        f"Scene '{scene_name}' was not found in {data_config_path}."
    )


def write_single_scene_config(scene_name: str, category: str, shape_prior: str) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="gaussian_case_", delete=False
    ) as f:
        writer = csv.writer(f)
        writer.writerow([scene_name, category, shape_prior])
        return Path(f.name)


def export_gaussian_data_for_scene(
    workspace_root: Path,
    python_exec: Path,
    data_config_path: Path,
):
    run_cmd(
        [
            python_exec,
            workspace_root / "export_gaussian_data.py",
            "--data-config",
            data_config_path,
        ],
        cwd=workspace_root,
    )


def train_gaussian_scene(
    workspace_root: Path,
    python_exec: Path,
    data_config_path: Path,
    gaussian_data_path: Path,
    gaussian_model_path: Path,
    iterations: int,
    lambda_depth: float,
    lambda_normal: float,
    lambda_aniso: float,
    lambda_seg: float,
    gs_init_opt: str,
):
    run_cmd(
        [
            python_exec,
            workspace_root / "gaussian_splatting" / "generate_interp_poses.py",
            "--data-config",
            data_config_path,
        ],
        cwd=workspace_root,
    )

    run_cmd(
        [
            python_exec,
            workspace_root / "gs_train.py",
            "-s",
            gaussian_data_path,
            "-m",
            gaussian_model_path,
            "--iterations",
            str(iterations),
            "--lambda_depth",
            str(lambda_depth),
            "--lambda_normal",
            str(lambda_normal),
            "--lambda_anisotropic",
            str(lambda_aniso),
            "--lambda_seg",
            str(lambda_seg),
            "--use_masks",
            "--isotropic",
            "--gs_init_opt",
            gs_init_opt,
        ],
        cwd=workspace_root,
    )


def load_pickle_trajectory(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        if "object_points" in data:
            arr = np.asarray(data["object_points"], dtype=np.float32)
        elif "controller_points" in data:
            arr = np.asarray(data["controller_points"], dtype=np.float32)
        else:
            keys = ", ".join(sorted(str(k) for k in data.keys()))
            raise KeyError(
                f"Unsupported dict payload in {path}. Expected object_points/controller_points. Keys: {keys}"
            )
    else:
        arr = np.asarray(data, dtype=np.float32)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"Trajectory in {path} must have shape [T, N, 3], got {arr.shape}."
        )
    return arr


def load_reference_trajectory(path: Path, npz_position_key: str):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if npz_position_key not in data.files:
                available = ", ".join(sorted(data.files))
                raise KeyError(
                    f"Key '{npz_position_key}' not in {path}. Available keys: {available}"
                )
            arr = np.asarray(data[npz_position_key], dtype=np.float32)
    else:
        arr = load_pickle_trajectory(path)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"Reference trajectory in {path} must have shape [T, N, 3], got {arr.shape}."
        )
    return arr


def register_scale_shift(source_traj: np.ndarray, ref_traj: np.ndarray):
    t = min(source_traj.shape[0], ref_traj.shape[0])
    n = min(source_traj.shape[1], ref_traj.shape[1])
    if t < 1 or n < 1:
        raise ValueError(
            f"Cannot register empty overlap: source={source_traj.shape}, ref={ref_traj.shape}"
        )

    src = source_traj[:t, :n].reshape(-1, 3)
    dst = ref_traj[:t, :n].reshape(-1, 3)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    denom = float(np.sum(src_c * src_c))
    if denom < 1e-12:
        raise ValueError("Degenerate source trajectory overlap; cannot estimate scale.")

    scale = float(np.sum(src_c * dst_c) / denom)
    shift = mu_dst - scale * mu_src

    aligned = source_traj * scale + shift[None, None, :]

    before_rmse = float(np.sqrt(np.mean((src - dst) ** 2)))
    after_rmse = float(
        np.sqrt(
            np.mean(
                (
                    aligned[:t, :n].reshape(-1, 3)
                    - ref_traj[:t, :n].reshape(-1, 3)
                )
                ** 2
            )
        )
    )

    return aligned.astype(np.float32), scale, shift.astype(np.float32), before_rmse, after_rmse


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert NPZ to inference.pkl, run dynamic rendering, and generate overlay video for one case."
        )
    )
    parser.add_argument("--source-video", required=True, help="Path to source video (e.g., 0.mp4)")
    parser.add_argument("--source-npz", required=True, help="Path to source NPZ trajectory")
    parser.add_argument("--case-name", required=True, help="Output case name")
    parser.add_argument(
        "--gaussian-scene",
        default=None,
        help="Scene name used to resolve gaussian_data/model defaults (defaults to case-name).",
    )

    parser.add_argument("--source-view-index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=1)

    parser.add_argument("--base-path", default="data/different_types")
    parser.add_argument("--human-mask-path", default="data/different_types_human_mask")
    parser.add_argument("--object-mask-path", default="data/render_eval_data")
    parser.add_argument("--prediction-dir", default="gaussian_output_dynamic_white")
    parser.add_argument("--inference-dir", default="experiments")

    parser.add_argument("--gaussian-data-path", default=None)
    parser.add_argument("--gaussian-model-path", default=None)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument(
        "--auto-prepare-gaussian",
        action="store_true",
        help=(
            "Automatically export Gaussian data and train Gaussian model for this scene "
            "when required files are missing."
        ),
    )
    parser.add_argument(
        "--force-prepare-gaussian",
        action="store_true",
        help="Force Gaussian export and training even if outputs already exist.",
    )
    parser.add_argument(
        "--prepare-data-config",
        default="data_config.csv",
        help="CSV used to resolve scene category/shape_prior for auto Gaussian export.",
    )
    parser.add_argument(
        "--scene-category",
        default=None,
        help="Override category for auto Gaussian export (e.g. toy, cloth).",
    )
    parser.add_argument(
        "--scene-shape-prior",
        default=None,
        help="Override shape prior flag for auto export (true/false).",
    )
    parser.add_argument("--gaussian-train-iterations", type=int, default=10000)
    parser.add_argument("--lambda-depth", type=float, default=DEFAULT_LAMBDA_DEPTH)
    parser.add_argument("--lambda-normal", type=float, default=DEFAULT_LAMBDA_NORMAL)
    parser.add_argument("--lambda-anisotropic", type=float, default=DEFAULT_LAMBDA_ANISO)
    parser.add_argument("--lambda-seg", type=float, default=DEFAULT_LAMBDA_SEG)
    parser.add_argument("--gs-init-opt", default=DEFAULT_GS_INIT_OPT)

    parser.add_argument(
        "--reconstruct-endpoints",
        choices=["none", "prepend", "append", "both"],
        default="both",
        help="How to recover dropped boundary frames when converting NPZ back to PKL.",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--no-unflip-z", action="store_true")

    parser.add_argument(
        "--register-to",
        default=None,
        help=(
            "Optional reference trajectory path (.npz/.pkl/.kpl) for scale+shift registration "
            "after NPZ->inference conversion."
        ),
    )
    parser.add_argument(
        "--register-npz-position-key",
        default="position",
        help="Position key used when --register-to points to an NPZ file.",
    )

    parser.add_argument("--controller-name", default="hand")
    parser.add_argument(
        "--mask-fallback",
        choices=["empty", "error"],
        default="empty",
        help="If masks are missing, either create empty masks or stop with an error.",
    )

    parser.add_argument("--overwrite-source-video", action="store_true")
    parser.add_argument("--overwrite-split-json", action="store_true")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--workspace-root", default=None)

    args = parser.parse_args()

    workspace_root = (
        Path(args.workspace_root).resolve()
        if args.workspace_root is not None
        else detect_workspace_root(Path(__file__))
    )

    python_exec = Path(sys.executable).resolve()

    source_video = resolve_existing_path(workspace_root, args.source_video)
    source_npz = resolve_existing_path(
        workspace_root,
        args.source_npz,
        extra_roots=[workspace_root / "scripts" / "real_world"],
    )
    base_path = resolve_path(workspace_root, args.base_path)
    human_mask_path = resolve_path(workspace_root, args.human_mask_path)
    object_mask_path = resolve_path(workspace_root, args.object_mask_path)
    prediction_dir = resolve_path(workspace_root, args.prediction_dir)
    inference_dir = resolve_path(workspace_root, args.inference_dir)

    if not source_video.exists():
        raise FileNotFoundError(f"Source video not found: {source_video}")
    if not source_npz.exists():
        raise FileNotFoundError(f"Source NPZ not found: {source_npz}")

    register_to_path = (
        resolve_existing_path(
            workspace_root,
            args.register_to,
            extra_roots=[source_npz.parent, workspace_root / "scripts" / "real_world"],
        )
        if args.register_to is not None
        else None
    )
    if register_to_path is not None and not register_to_path.exists():
        raise FileNotFoundError(
            "Registration reference not found: "
            f"{register_to_path}. "
            "Try --register-to scripts/real_world/<file>.npz or an absolute path."
        )

    gaussian_scene = args.gaussian_scene or args.case_name

    if args.gaussian_data_path is None:
        gaussian_data_path = resolve_path(
            workspace_root, f"data/gaussian_data/{gaussian_scene}"
        )
    else:
        gaussian_data_path = resolve_path(workspace_root, args.gaussian_data_path)

    if args.gaussian_model_path is None:
        gaussian_model_path = resolve_path(
            workspace_root,
            f"gaussian_output/{gaussian_scene}/{args.exp_name}",
        )
    else:
        gaussian_model_path = resolve_path(workspace_root, args.gaussian_model_path)

    if args.auto_prepare_gaussian or args.force_prepare_gaussian:
        prepare_data_config = resolve_existing_path(
            workspace_root,
            args.prepare_data_config,
        )
        missing_data = not gaussian_data_path.exists()
        missing_model = not has_trained_gaussian_checkpoint(gaussian_model_path)
        needs_prepare = args.force_prepare_gaussian or missing_data or missing_model

        if needs_prepare:
            scene_category = args.scene_category
            scene_shape_prior = (
                normalize_shape_prior(args.scene_shape_prior)
                if args.scene_shape_prior is not None
                else None
            )

            if scene_category is None or scene_shape_prior is None:
                cfg_category, cfg_shape_prior = read_scene_metadata_from_config(
                    prepare_data_config,
                    gaussian_scene,
                )
                if scene_category is None:
                    scene_category = cfg_category
                if scene_shape_prior is None:
                    scene_shape_prior = cfg_shape_prior

            print(
                "[Info] Auto Gaussian prepare enabled for scene "
                f"{gaussian_scene} with category={scene_category}, shape_prior={scene_shape_prior}"
            )

            single_scene_config = write_single_scene_config(
                scene_name=gaussian_scene,
                category=scene_category,
                shape_prior=scene_shape_prior,
            )

            try:
                export_gaussian_data_for_scene(
                    workspace_root=workspace_root,
                    python_exec=python_exec,
                    data_config_path=single_scene_config,
                )

                if not gaussian_data_path.exists():
                    raise FileNotFoundError(
                        "Gaussian data export did not produce expected path: "
                        f"{gaussian_data_path}"
                    )

                train_gaussian_scene(
                    workspace_root=workspace_root,
                    python_exec=python_exec,
                    data_config_path=single_scene_config,
                    gaussian_data_path=gaussian_data_path,
                    gaussian_model_path=gaussian_model_path,
                    iterations=args.gaussian_train_iterations,
                    lambda_depth=args.lambda_depth,
                    lambda_normal=args.lambda_normal,
                    lambda_aniso=args.lambda_anisotropic,
                    lambda_seg=args.lambda_seg,
                    gs_init_opt=args.gs_init_opt,
                )
            finally:
                if single_scene_config.exists():
                    single_scene_config.unlink()

    if not gaussian_data_path.exists():
        raise FileNotFoundError(f"Gaussian data path not found: {gaussian_data_path}")
    if not has_trained_gaussian_checkpoint(gaussian_model_path):
        if gaussian_model_path.exists():
            raise FileNotFoundError(
                "Gaussian model is incomplete (missing point_cloud checkpoints): "
                f"{gaussian_model_path}. "
                "Use --auto-prepare-gaussian --force-prepare-gaussian to retrain."
            )
        raise FileNotFoundError(f"Gaussian model path not found: {gaussian_model_path}")

    case_dir = base_path / args.case_name
    color_dir = case_dir / "color"
    color_dir.mkdir(parents=True, exist_ok=True)

    target_video = color_dir / f"{args.source_view_index}.mp4"
    ensure_case_video(source_video, target_video, overwrite=args.overwrite_source_video)

    run_cmd(
        [python_exec, workspace_root / "split_video.py", "--input_dir", color_dir],
        cwd=workspace_root,
    )

    source_view_frame_dir = color_dir / str(args.source_view_index)
    frame_files = get_numeric_pngs(source_view_frame_dir)
    if not frame_files:
        raise RuntimeError(
            f"No extracted frames found for source view {args.source_view_index}: {source_view_frame_dir}"
        )

    frame_len = len(frame_files)
    split_path = case_dir / "split.json"
    if (not split_path.exists()) or args.overwrite_split_json:
        write_split_json(split_path, frame_len)
    else:
        print(f"[Info] Keeping existing split.json: {split_path}")

    inference_path = inference_dir / args.case_name / "inference.pkl"
    inference_path.parent.mkdir(parents=True, exist_ok=True)

    convert_cmd = [
        python_exec,
        workspace_root / "scripts" / "final_npz_to_pkl.py",
        "--input",
        source_npz,
        "--output",
        inference_path,
        "--output-format",
        "array",
        "--reconstruct-endpoints",
        args.reconstruct_endpoints,
        "--dt",
        str(args.dt),
        "--max-frames",
        str(frame_len),
    ]
    if args.no_unflip_z:
        convert_cmd.append("--no-unflip-z")
    run_cmd(convert_cmd, cwd=workspace_root)

    if register_to_path is not None:
        converted_traj = load_pickle_trajectory(inference_path)
        reference_traj = load_reference_trajectory(
            register_to_path,
            npz_position_key=args.register_npz_position_key,
        )
        (
            aligned_traj,
            scale,
            shift,
            rmse_before,
            rmse_after,
        ) = register_scale_shift(converted_traj, reference_traj)

        with open(inference_path, "wb") as f:
            pickle.dump(aligned_traj, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(
            "[Info] Applied scale+shift registration "
            f"using {register_to_path}. overlap_frames={min(converted_traj.shape[0], reference_traj.shape[0])}, "
            f"overlap_points={min(converted_traj.shape[1], reference_traj.shape[1])}, "
            f"scale={scale:.6f}, shift={shift.tolist()}, rmse_before={rmse_before:.6f}, rmse_after={rmse_after:.6f}"
        )

    # Cap frame_len to the trajectory length if the trajectory is shorter than the video.
    # This ensures the render, mask creation, and overlay all operate on the same frame range.
    converted_traj = load_pickle_trajectory(inference_path)
    traj_frame_len = int(converted_traj.shape[0])

    if traj_frame_len == 0:
        raise RuntimeError(
            f"Trajectory at {inference_path} has 0 frames; aborting rollout."
        )

    effective_len = min(traj_frame_len, frame_len)

    if traj_frame_len != effective_len:
        print(
            f"[Info] PKL has {traj_frame_len} frames but video has {frame_len} frames; "
            f"truncating PKL to {effective_len} frames."
        )
        with open(inference_path, "wb") as f:
            pickle.dump(converted_traj[:effective_len], f, protocol=pickle.HIGHEST_PROTOCOL)

    if frame_len != effective_len:
        print(
            f"[Info] Capping frame_len from {frame_len} to {effective_len} to match trajectory."
        )
        frame_len = effective_len
        write_split_json(split_path, frame_len)

    prediction_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            python_exec,
            workspace_root / "gs_render_dynamics.py",
            "-s",
            gaussian_data_path,
            "-m",
            gaussian_model_path,
            "--name",
            args.case_name,
            "--white_background",
            "--output_dir",
            prediction_dir,
            "--ctrl_pts_path",
            inference_path,
            "--max_steps", str(frame_len)
        ],
        cwd=workspace_root,
    )

    output_case_dir = prediction_dir / args.case_name
    render_fps = args.fps if args.fps is not None else detect_fps(source_video, 30)
    for view_idx in range(args.num_views):
        view_dir = output_case_dir / str(view_idx)
        if not view_dir.exists():
            print(f"[Warn] Missing rendered frames for view {view_idx}: {view_dir}")
            continue
        run_cmd(
            [
                python_exec,
                workspace_root / "gaussian_splatting" / "img2video.py",
                "--image_folder",
                view_dir,
                "--video_path",
                output_case_dir / f"{view_idx}.mp4",
                "--fps",
                str(render_fps),
            ],
            cwd=workspace_root,
        )

    if args.skip_overlay:
        print("[Done] Rendering completed. Overlay step skipped.")
        print(f"[Output] Inference PKL: {inference_path}")
        print(f"[Output] Render directory: {output_case_dir}")
        return

    object_mask_case_dir = object_mask_path / args.case_name
    human_mask_case_dir = human_mask_path / args.case_name
    object_mask_case_dir.mkdir(parents=True, exist_ok=True)
    human_mask_case_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(split_path, object_mask_case_dir / "split.json")

    copy_object_masks_if_available(
        case_dir=case_dir,
        object_mask_case_dir=object_mask_case_dir,
        num_views=args.num_views,
        controller_name=args.controller_name,
    )

    if args.mask_fallback == "empty":
        with open(split_path, "r", encoding="utf-8") as f:
            split_data = json.load(f)
        frame_len_from_split = int(split_data["frame_len"])
        ensure_empty_masks(
            case_dir=case_dir,
            object_mask_case_dir=object_mask_case_dir,
            human_mask_case_dir=human_mask_case_dir,
            num_views=args.num_views,
            frame_len=frame_len_from_split,
        )
    else:
        for view_idx in range(args.num_views):
            obj_dir = object_mask_case_dir / "mask" / str(view_idx)
            hum_dir = human_mask_case_dir / "mask" / str(view_idx) / "0"
            if not list(obj_dir.glob("*.png")):
                raise FileNotFoundError(
                    f"Missing object masks for view {view_idx}: {obj_dir}"
                )
            if not list(hum_dir.glob("*.png")):
                raise FileNotFoundError(
                    f"Missing human masks for view {view_idx}: {hum_dir}"
                )

    first_frame = frame_files[0]
    ref_img = cv2.imread(str(first_frame), cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        raise RuntimeError(f"Failed to read first frame for overlay sizing: {first_frame}")
    height, width = int(ref_img.shape[0]), int(ref_img.shape[1])

    temp_csv_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="single_case_", delete=False
        ) as f:
            writer = csv.writer(f)
            writer.writerow([args.case_name, "auto", "False"])
            temp_csv_path = Path(f.name)

        run_cmd(
            [
                python_exec,
                workspace_root / "visualize_render_results.py",
                "--base_path",
                base_path,
                "--prediction_dir",
                prediction_dir,
                "--human_mask_path",
                human_mask_path,
                "--object_mask_path",
                object_mask_path,
                "--data_config",
                temp_csv_path,
                "--height",
                str(height),
                "--width",
                str(width),
                "--fps",
                str(render_fps),
                "--num_views",
                str(args.num_views),
            ],
            cwd=workspace_root,
        )
    finally:
        if temp_csv_path is not None and temp_csv_path.exists():
            temp_csv_path.unlink()

    print("[Done] Full pipeline completed successfully.")
    print(f"[Output] Inference PKL: {inference_path}")
    print(f"[Output] Render directory: {output_case_dir}")
    print(f"[Output] Overlay video: {output_case_dir / '0_integrate.mp4'}")


if __name__ == "__main__":
    main()