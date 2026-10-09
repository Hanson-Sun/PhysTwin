import os
import csv
import sys
import shutil
from pathlib import Path

base_path = "./data/different_types"
checkpoint = "temporal_depth_output/training/best_model.pt"

# Check for --force flag
force = "--force" in sys.argv

os.system("rm -f timer.log")

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 1:
            continue
        case_name = row[0]

        case_path = Path(base_path) / case_name
        depth_old_path = case_path / "depth_old"
        depth_path = case_path / "depth"
        smoothed_path = case_path / "smoothed"

        if not case_path.exists():
            print(f"Warning: Case '{case_name}' not found in {base_path}. Skipping.")
            continue

        # Check if already smoothed
        if depth_old_path.exists() and not force:
            print(f"Skip: {case_name} (already smoothed, depth_old exists. Use --force to re-smooth.)")
            continue

        # Backup original depth if not already done
        if not depth_old_path.exists():
            if depth_path.exists():
                print(f"  Backing up: {case_name}/depth → depth_old")
                depth_path.rename(depth_old_path)
            else:
                print(f"  Warning: {case_name}/depth not found. Skipping backup.")
                continue

        # Run smoothing
        print(f"  Smoothing: {case_name}")
        ret = os.system(
            f"python -m temporal_depth_smoother.smooth_phystwin_data "
            f"--base-path {base_path} --case-name {case_name} --checkpoint {checkpoint}"
        )

        if ret == 0:
            # Move smoothed frames to depth
            smoothed_frames_path = smoothed_path / "frames"
            if smoothed_frames_path.exists():
                print(f"  Installing: smoothed/frames → depth")
                # Create depth directory and populate it with camera subdirs
                depth_path.mkdir(parents=True, exist_ok=True)
                for cam_dir in sorted(smoothed_frames_path.iterdir()):
                    if cam_dir.is_dir():
                        dest_cam = depth_path / cam_dir.name
                        if dest_cam.exists():
                            shutil.rmtree(dest_cam)
                        shutil.move(str(cam_dir), str(dest_cam))
                # Remove empty frames directory
                if smoothed_frames_path.exists():
                    smoothed_frames_path.rmdir()
                print(f"  ✓ {case_name} completed")
            else:
                print(f"  ✗ {case_name}: smoothed/frames not found!")
        else:
            print(f"  ✗ {case_name}: smoothing failed!")

