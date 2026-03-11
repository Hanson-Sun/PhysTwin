from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def visualize_depth(
    depth: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentiles: Tuple[float, float] = (2, 98),
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Convert depth map to color visualization."""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)

    if not np.any(valid):
        h, w = d.shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    valid_vals = d[valid]
    if vmin is None or vmax is None:
        lo, hi = np.percentile(valid_vals, percentiles)
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    gray = (norm * 255).astype(np.uint8)

    colored = cv2.applyColorMap(gray, colormap)
    colored[~valid] = 0

    return colored


class VideoWriterManager:
    """Manages video writers for multiple cameras."""

    def __init__(self, fps: int = 30, codec: str = "mp4v"):
        self.fps = fps
        self.codec = cv2.VideoWriter_fourcc(*codec)
        self.writers: Dict[int, cv2.VideoWriter] = {}

    def get_writer(
        self, camera_id: int, output_path: Path, frame_shape: Tuple[int, int]
    ) -> cv2.VideoWriter:
        """Get or create video writer for camera."""
        if camera_id not in self.writers:
            h, w = frame_shape
            output_path.parent.mkdir(parents=True, exist_ok=True)

            writer = cv2.VideoWriter(str(output_path), self.codec, self.fps, (w, h))

            if not writer.isOpened():
                raise RuntimeError(f"Failed to create video writer: {output_path}")

            self.writers[camera_id] = writer

        return self.writers[camera_id]

    def release_all(self):
        """Release all video writers."""
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
