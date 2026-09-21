import os
import csv
import shlex
import subprocess

base_path = "./data/different_types"

os.system("rm -f timer.log")

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]
        shape_generator = row[3].strip().lower() if len(row) >= 4 else ""

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        if shape_prior.lower() == "true":
            generator_arg = (
                f" --shape_generator {shlex.quote(shape_generator)}"
                if shape_generator
                else ""
            )
            subprocess.run(
                shlex.split(
                    f"python process_data.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)} --category {shlex.quote(category)} --shape_prior{generator_arg}"
                ),
                check=True,
            )
        else:
            subprocess.run(
                shlex.split(
                    f"python process_data.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)} --category {shlex.quote(category)}"
                ),
                check=True,
            )
