import os
import json
import csv

base_path = "./data/different_types"

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        
        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        os.system(
            f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
        )
