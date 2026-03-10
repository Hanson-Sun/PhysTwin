import os
import json
import csv
import sys

base_path = "./data/different_types"

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0].strip()
        
        if not os.path.exists(f"{base_path}/{case_name}"):
            continue
    
        # Read the train test split
        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)

        train_frame = split["train"][1]

        os.system(
            f"python optimize_cma.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}"
        )