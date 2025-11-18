import os
import glob
import csv

base_path = "./data/different_types"
output_path = "./data/different_types_human_mask"

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Use data_config.csv to be consistent with other scripts
with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        
        if not os.path.exists(f"{base_path}/{case_name}"):
            continue
            
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
        existDir(f"{output_path}/{case_name}")
        # Process to get the whole human mask for the video

        TEXT_PROMPT = "human"
        camera_num = 3
        assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num

        for camera_idx in range(camera_num):
            print(f"Processing {case_name} camera {camera_idx}")
            os.system(
                f"python ./data_process/segment_util_video.py --output_path {output_path}/{case_name} --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}"
            )
            os.system(f"rm -rf {base_path}/{case_name}/tmp_data")