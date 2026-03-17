"""
Convert PhysTwin data to temporal depth smoother format.
Loads color images from multiple cameras and saves as .npy files.
"""
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def load_frame(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to load {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_camera_frames(color_dir, camera_id) -> np.ndarray | None:
    cam_path = Path(color_dir) / str(camera_id)
    if not cam_path.exists():
        return None

    png_files = sorted(cam_path.glob('*.png'), key=lambda p: int(p.stem))
    if not png_files:
        return None

    frames = [load_frame(f) for f in png_files]
    print(f"  cam{camera_id}: loaded {len(frames)} frames", flush=True)
    return np.array(frames, dtype=np.uint8)  # [T, H, W, 3]


def process_case(case_dir, output_dir) -> bool:
    case_name = Path(case_dir).name
    color_dir = Path(case_dir) / 'color'
    if not color_dir.exists():
        print(f"  [{case_name}] no color dir, skipping", flush=True)
        return False

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for cam_id in [0, 1, 2]:
        frames = load_camera_frames(color_dir, cam_id)
        if frames is None:
            print(f"  [{case_name}] cam{cam_id} not found", flush=True)
            continue
        T, H, W, _ = frames.shape
        out = Path(output_dir) / f"{case_name}_cam{cam_id}_rgb.npy"
        np.save(out, frames)
        print(f"  [{case_name}] saved {out.name}  [{T}, {H}, {W}]", flush=True)

    return True


def main():
    source_dir = Path('/mnt/d/DATA/phystwin/data/different_types')
    output_dir = Path('/root/digital_clone_v2/temporal_depth_training_data')

    if not source_dir.exists():
        print(f"Source not found: {source_dir}"); return

    cases = sorted(d for d in source_dir.iterdir() if d.is_dir())
    print(f"Found {len(cases)} cases", flush=True)

    def run(case):
        print(f"\n── {case.name} ──", flush=True)
        try:
            return process_case(case, output_dir)
        except Exception as e:
            print(f"  [{case.name}] ERROR: {e}", flush=True)
            return False

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(run, cases))

    ok = sum(results)
    print(f"\nDone: {ok}/{len(cases)} cases processed")


if __name__ == '__main__':
    main()