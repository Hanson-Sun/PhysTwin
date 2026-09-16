"""Visualize a depth map as a colorized PNG.

Examples:
    python -m mujoco_sim.visualize_depth \
        --input data/different_types/case/depth/0/0.npy \
        --output depth_preview.png

    python -m mujoco_sim.visualize_depth \
        --input depth.npy --output depth_preview.png \
        --unit millimeters --min_depth 0.2 --max_depth 1.5

PhysTwin case depth files are normally uint16 millimetres. Floating-point
inputs are interpreted as metres unless ``--unit`` says otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image


def load_depth(path: Path, unit: str = "auto") -> np.ndarray:
    """Load a 2-D depth map and return it in metres.

    ``.npy`` files and image files are supported. In automatic mode, integer
    data is treated as millimetres (PhysTwin's convention), while floating
    point data is treated as metres (MuJoCo's renderer convention).
    """
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        depth = np.asarray(Image.open(path))

    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth map, got shape {depth.shape}")

    was_integer = np.issubdtype(depth.dtype, np.integer)
    depth = depth.astype(np.float32, copy=False)
    if unit == "auto":
        unit = "millimeters" if was_integer else "meters"
    if unit == "millimeters":
        depth = depth / 1000.0
    elif unit != "meters":
        raise ValueError(f"Unsupported depth unit: {unit}")
    return depth


def colorize_depth(
    depth_m: np.ndarray,
    min_depth: float | None = None,
    max_depth: float | None = None,
    percentile: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    """Convert a depth map in metres to an RGB uint8 visualization."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        raise ValueError("Depth map contains no positive finite values")

    valid_depth = depth_m[valid]
    low = float(min_depth) if min_depth is not None else float(np.percentile(valid_depth, percentile[0]))
    high = float(max_depth) if max_depth is not None else float(np.percentile(valid_depth, percentile[1]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid visualization range: min={low}, max={high}")

    normalized = np.clip((depth_m - low) / (high - low), 0.0, 1.0)
    # Nearer points are brighter/yellow and farther points darker/blue.
    rgb = (cm.turbo(normalized)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def save_depth_visualization(
    input_path: str | Path,
    output_path: str | Path,
    unit: str = "auto",
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> None:
    """Load, colorize, and save a depth map visualization."""
    depth = load_depth(Path(input_path), unit=unit)
    image = colorize_depth(depth, min_depth=min_depth, max_depth=max_depth)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Depth .npy or image file")
    parser.add_argument("--output", type=Path, required=True, help="Output RGB PNG path")
    parser.add_argument(
        "--unit",
        choices=("auto", "meters", "millimeters"),
        default="auto",
        help="Input units; auto uses millimetres for integer data and metres for floating-point data.",
    )
    parser.add_argument("--min_depth", type=float, help="Color scale minimum in metres")
    parser.add_argument("--max_depth", type=float, help="Color scale maximum in metres")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    save_depth_visualization(
        args.input,
        args.output,
        unit=args.unit,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    print(f"Saved depth visualization to {args.output}")


if __name__ == "__main__":
    main()
