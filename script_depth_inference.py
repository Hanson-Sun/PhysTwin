"""
Batch depth estimation with DA3 streaming for all cases in data_config.csv.

For each case:
  1. Runs parse_depth_da3_streaming.py (outputs to a temp directory)
  2. Clears the case's existing depth/ camera subdirectories
  3. Moves the new depth maps into data/different_types/<case_name>/depth/
"""

import csv
import shutil
import sys
from pathlib import Path

BASE_PATH      = Path("./data/different_types")
TEMP_OUTPUT    = Path("./parsed_depth_output")
CHUNK_SIZE     = 3
OVERLAP        = 2
BLEND_MODE     = "linear"
MODEL          = "DA3"
POSE_CALIBRATION_MODEL = "DA3" #"DUSt3R"

def run(cmd: str) -> int:
    import os
    print(f"\n$ {cmd}")
    return os.system(cmd)


def clear_depth_cameras(depth_dir: Path):
    """Remove and recreate all numeric camera subdirectories in depth/."""
    if not depth_dir.exists():
        depth_dir.mkdir(parents=True)
        return
    for sub in sorted(depth_dir.iterdir()):
        if sub.is_dir() and sub.name.isdigit():
            shutil.rmtree(sub)
            print(f"  Cleared {sub}")


def move_depth_results(temp_case_dir: Path, depth_dir: Path):
    """Move numeric camera subdirs from temp output into depth/."""
    depth_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for sub in sorted(temp_case_dir.iterdir()):
        if sub.is_dir() and sub.name.isdigit():
            dest = depth_dir / sub.name
            shutil.move(str(sub), str(dest))
            print(f"  Moved {sub.name}/ → {dest}")
            moved += 1
    if moved == 0:
        print(f"  WARNING: No camera subdirectories found in {temp_case_dir}")


def main():
    with open("data_config.csv", newline="", encoding="utf-8") as f:
        rows = [(r[0].strip(), r[1].strip(), r[2].strip()) for r in csv.reader(f)]

    total = len(rows)
    for idx, (case_name, category, shape_prior) in enumerate(rows, 1):
        case_dir = BASE_PATH / case_name

        print(f"\n{'='*70}")
        print(f"[{idx}/{total}] {case_name}")
        print(f"{'='*70}")

        if not case_dir.exists():
            print(f"  WARNING: {case_dir} not found — skipping")
            continue

        depth_dir = case_dir / "depth"
        temp_case_dir = TEMP_OUTPUT / case_name

        # Clean up any leftover temp output for this case
        if temp_case_dir.exists():
            shutil.rmtree(temp_case_dir)

        # Run DA3 streaming into temp directory
        ret = run(
            f"python depth_inference/infer_depth.py"
            f" --case_dir {case_dir}"
            f" --output_root {TEMP_OUTPUT}"
            f" --model {MODEL}"
            f" --chunk_size {CHUNK_SIZE}"
            f" --overlap {OVERLAP}"
            f" --blend_mode {BLEND_MODE}"
            f" --pose_calibration_model {POSE_CALIBRATION_MODEL}"
            f" --verbose"
            f" --visualize"
        )

        if ret != 0:
            print(f"  ERROR: parsing depth failed for {case_name} (exit {ret})")
            sys.exit(ret)

        # Swap old depth camera dirs for new ones
        print(f"\n  Installing depth maps into {depth_dir}...")
        clear_depth_cameras(depth_dir)
        move_depth_results(temp_case_dir, depth_dir)

        # Clean up temp dir for this case
        shutil.rmtree(temp_case_dir, ignore_errors=True)

    # Remove temp root if empty
    if TEMP_OUTPUT.exists() and not any(TEMP_OUTPUT.iterdir()):
        TEMP_OUTPUT.rmdir()

    print(f"\n{'='*70}")
    print(f"Done. Processed {total} case(s).")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
