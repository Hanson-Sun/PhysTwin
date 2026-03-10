#!/usr/bin/env python3
"""
Align extrinsics to detected planes for all cases in data_config.csv.
Run after script_detect_environment_planes.py.
"""

import os
import csv
import sys

base_path = "./data/different_types"

# Read config and align each case
with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 1:
            print(f"Warning: Skipping empty row")
            continue
        
        case_name = row[0].strip()
        
        # Check if case exists
        if not os.path.exists(f"{base_path}/{case_name}"):
            print(f"Warning: Case '{case_name}' not found in {base_path}. Skipping.")
            continue
        
        # Check if planes were detected
        planes_path = f"{base_path}/{case_name}/environment_planes.json"
        if not os.path.exists(planes_path):
            print(f"Warning: Planes not found for '{case_name}' at {planes_path}. Skipping.")
            continue
        
        print(f"\n{'='*70}")
        print(f"Aligning extrinsics to plane: {case_name}")
        print(f"{'='*70}")
        
        # Run alignment
        cmd = f"python -u camera_alignment/align_to_plane.py --base_path {base_path} {case_name}"
        ret = os.system(cmd)
        
        if ret != 0:
            print(f"ERROR: Alignment failed for {case_name} with return code {ret}")
            sys.exit(1)

print(f"\n{'='*70}")
print("✓ All alignments completed successfully!")
print(f"{'='*70}")
