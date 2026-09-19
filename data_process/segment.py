# Process to get the masks of the controller and the object
import glob
import shutil
import sys
import subprocess
import shlex
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, required=True)
parser.add_argument(
    "--controller_names",
    type=str,
    default="hand,claw,gripper,robot gripper,robotic gripper,robot",
    help="Comma-separated labels that should remain as controller detections.",
)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_num = 3
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num
print(f"Processing {case_name}")

for camera_idx in range(camera_num):
    print(f"Processing {case_name} camera {camera_idx}")
    result = subprocess.run(
        shlex.split(
        f"{sys.executable} ./data_process/segment_util_video.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)} --TEXT_PROMPT {shlex.quote(TEXT_PROMPT)} --camera_idx {camera_idx} --controller_names {shlex.quote(args.controller_names)}"
        ),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Video segmentation failed for camera {camera_idx}")
    shutil.rmtree(f"{base_path}/{case_name}/tmp_data", ignore_errors=True)
