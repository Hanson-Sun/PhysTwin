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


def bridge_mask_gaps(mask, base_path=None, case_name=None, camera_id=None):
    """
    Bridge gaps between disconnected mask regions while preserving edge quality.
    
    Args:
        mask: Binary numpy array
        base_path: Base path for saving visualization (optional)
        case_name: Case name for saving visualization (optional)
        camera_id: Camera ID for saving visualization (optional)
        
    Returns:
        Processed mask with gaps bridged if multiple components exist, otherwise original mask
    """
    # Find connected components
    num_components, labels = cv2.connectedComponents(mask.astype(np.uint8))
    
    # If only one connected component (plus background), no need to bridge
    if num_components <= 2:  # 0 is background, 1 is the single object
        return mask
    
    print(f"    Found {num_components - 1} disconnected regions, bridging gaps...")
    original_mask = mask.copy()
    
    # Use minimal morphological closing to connect nearby components
    # Use a small kernel to minimize edge quality loss
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Apply closing iteratively, checking if components are now connected
    bridged_mask = mask.copy()
    for iteration in range(1, 4):
        bridged_mask = cv2.morphologyEx(
            bridged_mask.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel, 
            iterations=1
        ).astype(bool)
        
        num_components_new, _ = cv2.connectedComponents(bridged_mask.astype(np.uint8))
        
        print(f"      Iteration {iteration}: {num_components_new - 1} components remaining")
        
        # Stop if all components are now connected
        if num_components_new <= 2:
            print(f"    Successfully bridged all gaps")
            
            # Save visualization if paths provided
            if base_path and case_name and camera_id is not None:
                vis_dir = f"{base_path}/{case_name}/mask_bridging_vis"
                exist_dir(vis_dir)
                
                # Create side-by-side visualization
                vis_image = np.zeros((original_mask.shape[0], original_mask.shape[1] * 2, 3), dtype=np.uint8)
                vis_image[:, :original_mask.shape[1]] = (original_mask.astype(np.uint8) * 255)[:, :, np.newaxis]
                vis_image[:, original_mask.shape[1]:] = (bridged_mask.astype(np.uint8) * 255)[:, :, np.newaxis]
                
                vis_path = f"{vis_dir}/camera_{camera_id}_bridged.png"
                cv2.imwrite(vis_path, vis_image)
                print(f"      Saved bridging visualization to {vis_path}")
            
            return bridged_mask
    
    # If still not fully connected after 3 iterations, return best effort
    print(f"    Warning: Could not fully bridge all gaps after 3 iterations")
    
    # Save visualization of partially bridged result
    if base_path and case_name and camera_id is not None:
        vis_dir = f"{base_path}/{case_name}/mask_bridging_vis"
        exist_dir(vis_dir)
        
        vis_image = np.zeros((original_mask.shape[0], original_mask.shape[1] * 2, 3), dtype=np.uint8)
        vis_image[:, :original_mask.shape[1]] = (original_mask.astype(np.uint8) * 255)[:, :, np.newaxis]
        vis_image[:, original_mask.shape[1]:] = (bridged_mask.astype(np.uint8) * 255)[:, :, np.newaxis]
        
        vis_path = f"{vis_dir}/camera_{camera_id}_bridged_partial.png"
        cv2.imwrite(vis_path, vis_image)
        print(f"      Saved partial bridging visualization to {vis_path}")
    
    return bridged_mask


if __name__ == "__main__":
    exist_dir(f"{base_path}/{case_name}/cotracker")

    for i in range(num_cam):
        print(f"Processing {i}th camera")

        video_path = f"{base_path}/{case_name}/color/{i}.mp4"

        # Determine target resolution from depth maps
        depth_files = sorted(glob.glob(f"{base_path}/{case_name}/depth/{i}/*.npy"))
        scale_factor = 1.0
        if depth_files:
            depth_sample = np.load(depth_files[0])
            depth_height, depth_width = depth_sample.shape[:2]
            first_frame = iio.imread(video_path, plugin="FFMPEG", index=0)
            orig_width = first_frame.shape[1]
            scale_factor = depth_width / orig_width
            print(f"  Using video {i}.mp4, scale to depth res {depth_width}×{depth_height} (factor {scale_factor:.4f})")
        else:
            print(f"  No depth files found for camera {i}; scale_factor=1.0")
        
        # Load the full video for tracking
        frames = iio.imread(video_path, plugin="FFMPEG")
        video_height, video_width = frames.shape[1:3]
        
        # Keep video on CPU to avoid memory issues, convert to tensor but don't move to GPU yet
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float()
        )  # B T C H W (on CPU)
        
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

        # Bridge any gaps in the mask caused by occlusion
        mask = bridge_mask_gaps(mask, base_path=base_path, case_name=case_name, camera_id=i)

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
        # pred_tracks, pred_visibility = cotracker(video.to(device), queries=query_pixels[None], backward_tracking=True)
        # pred_tracks, pred_visibility = cotracker(video, grid_query_frame=0)

        # Run Online CoTracker:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)
        cotracker(video_chunk=video[:, :].to(device), is_first_step=True, queries=query_pixels[None])

        # Process the video
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2].to(device)
            )  # B T N 2,  B T N 1

        torch.cuda.empty_cache()
        vis = Visualizer(
            save_dir=f"{base_path}/{case_name}/cotracker", pad_value=0, linewidth=3
        )
        vis.visualize(video.cpu(), pred_tracks.cpu(), pred_visibility.cpu(), filename=f"{i}")
        
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

