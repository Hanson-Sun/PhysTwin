# Use co-tracker to track the ibject and controller in the video (pick 5000 pixels in the masked area)

import torch
import imageio.v3 as iio
from utils.visualizer import Visualizer
import glob
import cv2
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

num_cam = 3
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == num_cam
device = "cuda"


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    exist_dir(f"{base_path}/{case_name}/cotracker")

    for i in range(num_cam):
        print(f"Processing {i}th camera")
        
        # Try to load high-res original video if available, otherwise use standard video
        orig_video_path = f"{base_path}/{case_name}/color/{i}.orig.mp4"
        std_video_path = f"{base_path}/{case_name}/color/{i}.mp4"
        
        # Get scale factor from video resolutions
        scale_factor = 1.0
        if os.path.exists(orig_video_path):
            # Load just first frame of each to get resolutions
            orig_frames = iio.imread(orig_video_path, plugin="FFMPEG", index=0)
            std_frames = iio.imread(std_video_path, plugin="FFMPEG", index=0)
            orig_width = orig_frames.shape[1]
            std_width = std_frames.shape[1]
            scale_factor = std_width / orig_width
            video_path = orig_video_path
            print(f"  Using high-res original video: {i}.orig.mp4 ({orig_width} → {std_width})")
        else:
            video_path = std_video_path
            print(f"  Using standard video: {i}.mp4")
        
        # Load the full video for tracking
        frames = iio.imread(video_path, plugin="FFMPEG")
        video_height, video_width = frames.shape[1:3]
        
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W
        
        # Load the first-frame mask and scale it to match video resolution
        mask_paths = glob.glob(f"{base_path}/{case_name}/mask/{i}/*/0.png")
        mask = None
        for mask_path in mask_paths:
            current_mask = read_mask(mask_path)
            # Scale mask to match the video resolution
            if (current_mask.shape[0], current_mask.shape[1]) != (video_height, video_width):
                current_mask = cv2.resize(
                    current_mask.astype(np.uint8),
                    (video_width, video_height),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            if mask is None:
                mask = current_mask
            else:
                mask = np.logical_or(mask, current_mask)

        # Draw the mask
        query_pixels = np.argwhere(mask)
        # Revert x and y
        query_pixels = query_pixels[:, ::-1]
        query_pixels = np.concatenate(
            [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
        )
        query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)
        # Randomly select 5000 query points
        query_pixels = query_pixels[torch.randperm(query_pixels.shape[0])[:5000]]

        # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        # pred_tracks, pred_visibility = cotracker(video, queries=query_pixels[None], backward_tracking=True)
        # pred_tracks, pred_visibility = cotracker(video, grid_query_frame=0)

        # # Run Online CoTracker:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)
        cotracker(video_chunk=video, is_first_step=True, queries=query_pixels[None])

        # Process the video
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )  # B T N 2,  B T N 1
        vis = Visualizer(
            save_dir=f"{base_path}/{case_name}/cotracker", pad_value=0, linewidth=3
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=f"{i}")
        
        # Scale tracked points back to original resolution if upscaled
        track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]
        visibility_to_save = pred_visibility[0].cpu().numpy()
        
        # Rescale tracked points to match standard video resolution
        if abs(scale_factor - 1.0) > 0.0001:
            print(f"  Rescaling tracked points by factor {scale_factor:.4f}")
            track_to_save = track_to_save * scale_factor
            track_to_save = np.round(track_to_save).astype(np.float32)
        
        np.savez(
            f"{base_path}/{case_name}/cotracker/{i}.npz",
            tracks=track_to_save,
            visibility=visibility_to_save,
        )

