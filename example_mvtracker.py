#!/usr/bin/env python3
"""
Quick start examples for MVTracker pipeline.

This demonstrates:
1. Basic tracking with manual query points + Rerun visualization
2. How to integrate custom 3D segmentation model later
"""

from mvtracker_pipeline import MVTrackerPipeline
import numpy as np


def example_manual_tracking_with_rerun():
    """Track points and visualize with Rerun."""
    
    # Initialize pipeline
    pipeline = MVTrackerPipeline(
        data_dir="/root/digital_clone_v2/data/different_types/",
        output_dir="./results",
        device="cuda"
    )
    
    # Manually specify points to track (frame_idx, x, y, z)
    query_points = np.array([
        [0, 0.0, 0.0, 0.0],    # Origin
        [0, 0.1, 0.0, 0.0],    # +X
        [0, 0.0, 0.1, 0.0],    # +Y
    ], dtype=np.float32)
    
    # Track points
    traj, vis, metadata = pipeline.track(
        query_points=query_points,
        num_views=3,
        iters=6
    )
    
    # Save results
    pipeline.save_results(traj, vis, metadata)
    
    # Visualize with Rerun
    rgbs, _ = pipeline._load_video_frames()
    pipeline.visualize_3d_rerun(
        rgbs, traj, vis, metadata,
        output_path="./results/tracking.rrd",
        mode="save"
    )
    
    print(f"✓ Tracking complete: {traj.shape[0]} frames, {traj.shape[1]} points")
    print(f"✓ Results saved to ./results/")
    print(f"✓ View with: rerun ./results/tracking.rrd")


def example_with_segmentation():
    """
    Example showing how to integrate custom 3D segmentation model.
    This is the framework for future integration.
    """
    
    # Define custom segmentation function
    def custom_segmentation(rgbs, extrs, intrs):
        """
        Your custom 3D segmentation model.
        
        This should:
        1. Take RGB frames and camera parameters
        2. Return a 3D segmentation mask
        3. We'll sample points from the mask
        
        Args:
            rgbs: (V, T, H, W, 3) uint8
            extrs: (V, T, 3, 4) world-to-camera
            intrs: (V, T, 3, 3) camera intrinsics
            
        Returns:
            mask: (V, T, H, W) binary mask of object
        """
        # TODO: Implement your segmentation model here
        # Example: Use Mask3D, SAM3D, or other 3D segmentation
        V, T, H, W = rgbs.shape[0], rgbs.shape[1], rgbs.shape[2], rgbs.shape[3]
        return np.zeros((V, T, H, W), dtype=bool)
    
    # Initialize with segmentation function
    pipeline = MVTrackerPipeline(
        data_dir="/root/digital_clone_v2/data/different_types/bottle_lift_single",
        output_dir="./results_with_seg",
        device="cuda",
        segmentation_fn=custom_segmentation  # Pass your segmentation here
    )
    
    # Run without manual query_points - will use segmentation to select them
    traj, vis, metadata = pipeline.track(num_views=3, iters=6)
    pipeline.save_results(traj, vis, metadata)
    
    # Visualize
    rgbs, _ = pipeline._load_video_frames()
    pipeline.visualize_3d_rerun(rgbs, traj, vis, metadata, mode="save")
    
    print("✓ Tracking with segmentation complete")


if __name__ == "__main__":
    print("Running MVTracker example...")
    example_manual_tracking_with_rerun()
    
    # To use with segmentation later, uncomment:
    # example_with_segmentation()
