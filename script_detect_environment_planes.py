"""
Batch environment plane detection for all cases in data_config.csv
"""

import os
import csv
import sys
from pathlib import Path

base_path = "./data/different_types"

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) < 1:
            continue
        
        case_name = row[0].strip()
        data_pkl = f"{base_path}/{case_name}/final_data.pkl"
        
        # Check if data exists
        if not os.path.exists(data_pkl):
            print(f"[SKIP] {case_name}: final_data.pkl not found")
            continue
        
        # Create output directory
        output_dir = f"experiments_optimization/{case_name}"
        output_path = f"{output_dir}/environment_planes.json"
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[PROCESS] {case_name}")
        
        # Run detection
        cmd = f"python detect_environment_planes.py --base_path {base_path} --case_name {case_name} --output_path {output_path}"
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print(f"[ERROR] {case_name}: Detection failed with exit code {exit_code}. Exiting.")
            sys.exit(1)
        
        print(f"[OK] {case_name}")

print("[DONE] Environment plane detection complete")
