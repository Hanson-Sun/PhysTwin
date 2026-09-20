import os
from argparse import ArgumentParser
import time
import logging
import json
import glob
import sys
import shlex
import subprocess
import importlib.util
from data_process.controller_labels import is_controller_label, parse_controller_names

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="./data/different_types",
)
parser.add_argument("--case_name", type=str, required=True)
# The category of the object used for segmentation
parser.add_argument("--category", type=str, required=True)
parser.add_argument(
    "--controller_prompt",
    type=str,
    default="robot gripper.",
    help="GroundingDINO controller prompt; multiple phrases are dot-separated.",
)
parser.add_argument(
    "--controller_names",
    type=str,
    default="hand,claw,gripper,robot gripper,robotic gripper,robot",
    help="Comma-separated labels treated as the controller mask.",
)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument("--skip_segmentation", action="store_true")
parser.add_argument(
    "--visualize",
    action="store_true",
    help="Show preprocessing visualizations; disabled by default.",
)
args = parser.parse_args()

# Grounded-SAM2 is intentionally isolated in the data-processing environment.
if not args.skip_segmentation and importlib.util.find_spec("sam2") is None:
    raise RuntimeError(
        "SAM2 is unavailable in this Python environment. Run process_data.py "
        "with the phystwin-data interpreter or use scripts/run_case_pipeline.py."
    )

# Set the debug flags
PROCESS_SEG = not args.skip_segmentation
PROCESS_SHAPE_PRIOR = True
PROCESS_TRACK = True
PROCESS_3D = True
PROCESS_ALIGN = True
PROCESS_FINAL = True

base_path = args.base_path
case_name = args.case_name
category = args.category
TEXT_PROMPT = f"{category}.{args.controller_prompt}"
CONTROLLER_NAMES = parse_controller_names(args.controller_names)
SHAPE_PRIOR = args.shape_prior

logger = None


def setup_logger(log_file="timer.log"):
    global logger 

    if logger is None:
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


setup_logger()


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def run_stage(command):
    result = subprocess.run(shlex.split(command), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Processing stage failed with exit code {result.returncode}: {command}")


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


if PROCESS_SEG:
    # Get the masks of the controller and the object using GroundedSAM2
    with Timer("Video Segmentation"):
        run_stage(
            f"{sys.executable} ./data_process/segment.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)} --TEXT_PROMPT {shlex.quote(TEXT_PROMPT)} --controller_names {shlex.quote(','.join(CONTROLLER_NAMES))}"
        )


if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR:
    # Get the mask path for the image
    with open(f"{base_path}/{case_name}/mask/mask_info_{0}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if not is_controller_label(value, CONTROLLER_NAMES):
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_path = f"{base_path}/{case_name}/mask/0/{obj_idx}/0.png"

    existDir(f"{base_path}/{case_name}/shape")
    # Get the high-resolution of the image to prepare for the trellis generation
    with Timer("Image Upscale"):
        if not os.path.isfile(f"{base_path}/{case_name}/shape/high_resolution.png"):
            run_stage(
                f"{shlex.quote(sys.executable)} ./data_process/image_upscale.py --img_path {shlex.quote(f'{base_path}/{case_name}/color/0/0.png')} --mask_path {shlex.quote(mask_path)} --output_path {shlex.quote(f'{base_path}/{case_name}/shape/high_resolution.png')} --category {shlex.quote(category)}"
            )

    # Get the masked image of the object
    with Timer("Image Segmentation"):
        run_stage(
                f"{shlex.quote(sys.executable)} ./data_process/segment_util_image.py --img_path {shlex.quote(f'{base_path}/{case_name}/shape/high_resolution.png')} --TEXT_PROMPT {shlex.quote(category)} --output_path {shlex.quote(f'{base_path}/{case_name}/shape/masked_image.png')}"
        )

    with Timer("Shape Prior Generation"):
        run_stage(
                f"{shlex.quote(sys.executable)} ./data_process/shape_prior.py --img_path {shlex.quote(f'{base_path}/{case_name}/shape/masked_image.png')} --output_dir {shlex.quote(f'{base_path}/{case_name}/shape')}"
        )

if PROCESS_TRACK:
    # Get the dense tracking of the object using Co-tracker
    with Timer("Dense Tracking"):
        run_stage(
            f"{shlex.quote(sys.executable)} ./data_process/dense_track.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)}"
        )

if PROCESS_3D:
    # Get the pcd in the world coordinate from the raw observations
    with Timer("Lift to 3D"):
        run_stage(
            f"{shlex.quote(sys.executable)} ./data_process/data_process_pcd.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)}"
        )

    # Further process and filter the noise of object and controller masks
    with Timer("Mask Post-Processing"):
        run_stage(
            f"{shlex.quote(sys.executable)} ./data_process/data_process_mask.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)} --controller_names {shlex.quote(','.join(CONTROLLER_NAMES))}"
        )

    # Process the data tracking
    with Timer("Data Tracking"):
        run_stage(
            f"{shlex.quote(sys.executable)} ./data_process/data_process_track.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)}"
            + ("" if args.visualize else " --no_visualize")
        )

if PROCESS_ALIGN and SHAPE_PRIOR:
    # Align the shape prior with partial observation
    with Timer("Alignment"):
        run_stage(
            f"{shlex.quote(sys.executable)} ./data_process/align.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)} --controller_names {shlex.quote(','.join(sorted(CONTROLLER_NAMES)))}"
        )

if PROCESS_FINAL:
    # Get the final PCD used for the inverse physics with/without the shape prior
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:
            run_stage(
                f"{shlex.quote(sys.executable)} ./data_process/data_process_sample.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)} --shape_prior"
                + ("" if args.visualize else " --no_visualize")
            )
        else:
            run_stage(
                f"{shlex.quote(sys.executable)} ./data_process/data_process_sample.py --base_path {shlex.quote(base_path)} --case_name {shlex.quote(case_name)}"
                + ("" if args.visualize else " --no_visualize")
            )

    # Save the train test split
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
    split = {}
    split["frame_len"] = frame_len
    split["train"] = [0, int(frame_len * 0.7)]
    split["test"] = [int(frame_len * 0.7), frame_len]
    with open(f"{base_path}/{case_name}/split.json", "w") as f:
        json.dump(split, f)
