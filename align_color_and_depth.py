#!/usr/bin/env python3
"""
align_color_and_depth.py

Resizes color camera frames to match depth camera resolution using bilinear interpolation.

Usage:
  python align_color_and_depth.py --depth_dir /path/to/depth --color_dir /path/to/color --output_dir /path/to/output

This expects matching subfolders (e.g., `0`, `2`) inside both `depth_dir` and `color_dir`.
Depth frames: <depth_dir>/<cam>/<i>.npy
Color frames: <color_dir>/<cam>/<i>.png
Outputs:     <output_dir>/<cam>/<i>.png (resized to depth resolution)
"""

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import sys
import cv2
import os
import shutil


def find_subdirs(path: Path):
    return [p for p in sorted(path.iterdir()) if p.is_dir()]


def resize_color_to_depth(depth_path: Path, color_path: Path, out_path: Path, overwrite: bool = False):
    if out_path.exists() and not overwrite:
        return 'skipped'

    # Load depth to get size
    try:
        depth = np.load(depth_path)
    except Exception as e:
        return f'depth_read_error: {e}'

    if depth.ndim >= 2:
        h, w = depth.shape[0], depth.shape[1]
    else:
        return 'invalid_depth_shape'

    try:
        # If color_path is an image file path (string or Path), open with PIL
        if isinstance(color_path, (str, Path)):
            img = Image.open(color_path)
        else:
            # color_path may be an ndarray (frame from cv2)
            img = Image.fromarray(color_path)
    except Exception as e:
        return f'color_read_error: {e}'

    img_resized = img.resize((w, h), resample=Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_resized.save(out_path)
    return 'written'


def build_depth_index(depth_files):
    # depth_files: list[Path] sorted
    idx_map = {}
    ordered = list(depth_files)
    for i, p in enumerate(ordered):
        try:
            k = int(p.stem)
        except Exception:
            k = i
        idx_map[k] = p
    return idx_map, ordered


def process_video_for_cam(depth_sub: Path, video_path: Path, out_root: Path, args):
    depth_files = sorted(depth_sub.glob('*.npy'))
    if not depth_files:
        print(f'No depth frames in {depth_sub}, skipping video {video_path}')
        return 0, 0, 0

    idx_map, ordered = build_depth_index(depth_files)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print('Failed to open video', video_path)
        return 0, 0, 1

    n_total = 0
    n_written = 0
    n_errors = 0

    # prepare video writer to write resized video to temp file
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 30.0

    resized_video = video_path.with_suffix('.resized.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # find matching depth file by frame index
        dfile = idx_map.get(frame_idx)
        if dfile is None and frame_idx < len(ordered):
            dfile = ordered[frame_idx]

        if dfile is None:
            frame_idx += 1
            n_total += 1
            continue

        # target size from depth
        try:
            depth = np.load(dfile)
            th, tw = depth.shape[0], depth.shape[1]
        except Exception as e:
            print(f'Failed loading depth {dfile}: {e}')
            n_errors += 1
            frame_idx += 1
            continue

        # initialize writer when we know target size
        if writer is None:
            writer = cv2.VideoWriter(str(resized_video), fourcc, fps, (tw, th))

        # resize BGR frame to target (cv2 uses (w,h))
        try:
            resized_bgr = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f'Failed to resize frame {frame_idx}: {e}')
            n_errors += 1
            frame_idx += 1
            continue

        # write to temp video
        writer.write(resized_bgr)

        # also save per-frame png aligned (convert to RGB for PIL)
        try:
            frame_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            frame_rgb = resized_bgr

        out_file = out_root / depth_sub.name / (dfile.stem + args.ext)
        res = resize_color_to_depth(dfile, frame_rgb, out_file, overwrite=args.overwrite)
        n_total += 1
        if res == 'written':
            n_written += 1
        else:
            n_errors += 1
            print(f'Error processing frame {frame_idx} of {video_path}: {res}')

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    return n_total, n_written, n_errors


def main():
    p = argparse.ArgumentParser(description='Align color frames to depth resolution using bilinear scaling')
    p.add_argument('--depth_dir', required=True, help='Path to depth folder (contains subfolders like 0,2 with .npy files)')
    p.add_argument('--color_dir', required=True, help='Path to color folder (same subfolder structure with .png files)')
    p.add_argument('--output_dir', required=True, help='Where to write aligned color frames')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs')
    p.add_argument('--ext', default='.png', help='Color file extension (default: .png)')
    args = p.parse_args()

    depth_root = Path(args.depth_dir)
    color_root = Path(args.color_dir)
    out_root = Path(args.output_dir)

    if not depth_root.exists():
        print('Depth directory does not exist:', depth_root, file=sys.stderr)
        sys.exit(2)
    if not color_root.exists():
        print('Color directory does not exist:', color_root, file=sys.stderr)
        sys.exit(2)

    # Iterate subfolders in depth directory
    subdirs = find_subdirs(depth_root)
    if not subdirs:
        print('No subdirectories found in depth directory', depth_root, file=sys.stderr)

    n_total = 0
    n_written = 0
    n_skipped = 0
    n_errors = 0

    for sd in subdirs:
        rel = sd.name
        depth_files = sorted(sd.glob('*.npy'))
        if not depth_files:
            continue

        # Prioritize folder of images: color/<cam>/*.png
        color_sub = color_root / rel
        video_file = color_root / (rel + '.mp4')
        # also check for common video extensions
        if not video_file.exists():
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                candidate = color_root / (rel + ext)
                if candidate.exists():
                    video_file = candidate
                    break

        if color_sub.exists() and color_sub.is_dir():
            # process per-frame color images
            for dfile in depth_files:
                name = dfile.stem
                cfile = color_sub / (name + args.ext)
                out_file = out_root / rel / (name + args.ext)
                n_total += 1

                if not cfile.exists():
                    print(f'Warning: color file missing: {cfile} (depth: {dfile})')
                    n_errors += 1
                    continue

                res = resize_color_to_depth(dfile, cfile, out_file, overwrite=args.overwrite)
                if res == 'written':
                    n_written += 1
                elif res == 'skipped':
                    n_skipped += 1
                else:
                    n_errors += 1
                    print(f'Error processing {dfile} / {cfile}: {res}')

        if video_file.exists():
            # process video frames
            t_total, t_written, t_errors = process_video_for_cam(sd, video_file, out_root, args)
            n_total += t_total
            n_written += t_written
            n_errors += t_errors

        if not video_file.exists() and not color_sub.exists():
            print(f'Warning: color data missing for camera {rel} (neither folder nor video found)')
            n_errors += 1

    print('Done. Total frames:', n_total, 'written:', n_written, 'skipped:', n_skipped, 'errors:', n_errors)


if __name__ == '__main__':
    main()
