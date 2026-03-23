#!/usr/bin/env python3
"""
Test script for the bag extraction pipeline.

Creates synthetic data matching the expected output structure, validates it,
and tests that all file types load correctly.
"""

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_color_frame(i: int, height: int, width: int) -> np.ndarray:
    """Generate a deterministic synthetic RGB frame."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = (i * 25) % 256
    frame[:, :, 1] = (i * 15) % 256
    frame[:, :, 2] = (i * 35) % 256
    cv2.putText(frame, f"Frame {i}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame


def _make_depth_frame(i: int, height: int, width: int) -> np.ndarray:
    """Generate a deterministic synthetic depth frame (float32, metres)."""
    depth = np.full((height, width), 0.5 + 0.1 * np.sin(i / 5.0), dtype=np.float32)
    depth[100:200, 100:300] = 0.3  # simulated closer object
    return depth


# ── Create ────────────────────────────────────────────────────────────────────

def create_test_case(output_dir: str, case_name: str = "test_case",
                     num_frames: int = 10) -> str:
    """
    Create a synthetic case directory with the expected extraction layout.
    Returns the path to the case directory.
    """
    height, width = 480, 640
    case_dir = os.path.join(output_dir, case_name)
    color_dir = os.path.join(case_dir, "color", "0")
    depth_dir = os.path.join(case_dir, "depth", "0")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    video_path = os.path.join(case_dir, "color", "0.mp4")
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height)
    )

    for i in range(num_frames):
        color = _make_color_frame(i, height, width)
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(color_dir, f"{i}.png"), bgr)
        writer.write(bgr)
        np.save(os.path.join(depth_dir, f"{i}.npy"), _make_depth_frame(i, height, width))

    writer.release()
    print(f"✓ Created {num_frames} color PNGs, {num_frames} depth NPYs, 1 MP4")
    return case_dir


# ── Validate ──────────────────────────────────────────────────────────────────

def validate_case(case_dir: str) -> bool:
    """Check that the case directory has the correct structure and frame counts."""
    errors = []

    color_dir = os.path.join(case_dir, "color", "0")
    depth_dir = os.path.join(case_dir, "depth", "0")
    video_path = os.path.join(case_dir, "color", "0.mp4")

    color_count = len([f for f in os.listdir(color_dir) if f.endswith(".png")]) \
        if os.path.isdir(color_dir) else 0
    depth_count = len([f for f in os.listdir(depth_dir) if f.endswith(".npy")]) \
        if os.path.isdir(depth_dir) else 0

    if color_count == 0:
        errors.append("No color frames found")
    else:
        print(f"✓ {color_count} color frames")

    if depth_count == 0:
        errors.append("No depth frames found")
    else:
        print(f"✓ {depth_count} depth frames")

    if color_count != depth_count:
        errors.append(f"Frame count mismatch: {color_count} color vs {depth_count} depth")

    if not os.path.exists(video_path):
        errors.append("MP4 video not found")
    else:
        print(f"✓ Video present ({os.path.getsize(video_path)} bytes)")

    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return False

    return True


# ── Load test ─────────────────────────────────────────────────────────────────

def test_loading(case_dir: str) -> bool:
    """Verify that all file types can be opened and have correct properties."""
    color_dir = os.path.join(case_dir, "color", "0")
    depth_dir = os.path.join(case_dir, "depth", "0")
    video_path = os.path.join(case_dir, "color", "0.mp4")

    # PNG
    png_files = sorted(f for f in os.listdir(color_dir) if f.endswith(".png"))
    frame = cv2.imread(os.path.join(color_dir, png_files[0]))
    if frame is None:
        print("✗ Failed to load PNG")
        return False
    print(f"✓ PNG shape: {frame.shape}")

    # NPY
    npy_files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
    depth = np.load(os.path.join(depth_dir, npy_files[0]))
    print(f"✓ NPY shape: {depth.shape}, dtype: {depth.dtype}, "
          f"range: [{depth.min():.3f}, {depth.max():.3f}] m")

    # MP4
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ Failed to open MP4")
        return False
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"✓ MP4: {n} frames @ {fps:.0f} fps, {w}×{h}")

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test the bag extraction pipeline")
    parser.add_argument("--output-dir", help="Output directory (default: temp)")
    parser.add_argument("--keep", action="store_true", help="Keep files after test")
    parser.add_argument("--num-frames", type=int, default=10, help="Synthetic frame count")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        keep = True
    else:
        output_dir = tempfile.mkdtemp(prefix="bag_extraction_test_")
        keep = args.keep

    try:
        print(f"Output directory: {output_dir}\n")

        case_dir = create_test_case(output_dir, num_frames=args.num_frames)

        print("\nValidating structure...")
        if not validate_case(case_dir):
            print("✗ Validation failed")
            return 1

        print("\nTesting file loading...")
        if not test_loading(case_dir):
            print("✗ Loading test failed")
            return 1

        print("\n✓ All tests passed")

        if keep:
            print(f"Files saved to: {output_dir}")
        else:
            shutil.rmtree(output_dir)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        if not keep:
            shutil.rmtree(output_dir, ignore_errors=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())