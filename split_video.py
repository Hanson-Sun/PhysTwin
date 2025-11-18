"""
Extract individual frames from MP4 videos into PNG files.

This script splits all MP4 videos in a specified directory into individual frames,
creating one folder per video with frames named sequentially (0.png, 1.png, etc.).

Usage:
    python split_video.py --input_dir ./data/different_types/cloth_double_bend/color
    python split_video.py --input_dir ./data/different_types/cloth_double_bend/color --fps 30
"""

import imageio
import os
import argparse
from pathlib import Path


def extract_frames_from_video(video_path: str, output_dir: str) -> int:
    """
    Extract all frames from a video file into PNG images.

    Args:
        video_path: Path to the input MP4 video file
        output_dir: Directory to save PNG frames

    Returns:
        Number of frames extracted
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Read video file
        reader = imageio.get_reader(video_path)
    except Exception as e:
        print(f"Error: Could not open video file: {video_path}")
        print(f"  Details: {e}")
        return 0

    # Get video properties
    num_frames = len(reader)
    meta = reader.get_meta_data()
    fps = meta.get("fps", "unknown")

    print(f"Processing: {os.path.basename(video_path)}")
    print(f"  Total frames: {num_frames}, FPS: {fps}")

    frame_count = 0

    for frame_idx, frame in enumerate(reader):
        # Save frame as PNG
        frame_path = os.path.join(output_dir, f"{frame_count}.png")
        imageio.imwrite(frame_path, frame)
        frame_count += 1

    reader.close()

    print(f"  Saved {frame_count} frames to {output_dir}")
    return frame_count


def split_videos(input_dir: str) -> None:
    """
    Split all MP4 videos in a directory into individual frames.

    For each video file (e.g., '0.mp4'), creates a folder (e.g., '0/')
    and extracts all frames as PNG files.

    Args:
        input_dir: Directory containing MP4 videos
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Find all MP4 files
    mp4_files = sorted(input_path.glob("*.mp4"))

    if not mp4_files:
        print(f"Error: No MP4 files found in {input_dir}")
        return

    print(f"Found {len(mp4_files)} MP4 file(s)")

    total_frames_extracted = 0

    for video_file in mp4_files:
        # Create output folder with same name as video (without extension)
        folder_name = video_file.stem  # e.g., "0" from "0.mp4"
        output_folder = input_path / folder_name

        # Extract frames
        num_frames = extract_frames_from_video(str(video_file), str(output_folder))
        total_frames_extracted += num_frames

    print(f"\nTotal frames extracted: {total_frames_extracted}")
    print(f"Frame extraction complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual frames from MP4 videos into PNG files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing MP4 video files",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional: Extract frames at specific FPS (if not specified, extracts all frames)",
    )

    args = parser.parse_args()

    if args.fps is not None:
        print(
            f"Note: --fps parameter specified but not implemented yet. Extracting all frames."
        )

    split_videos(args.input_dir)


if __name__ == "__main__":
    main()
