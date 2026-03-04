#!/usr/bin/env python3
"""
Temporal depth filtering to reduce jitter and improve consistency.
Applies several filtering strategies: median, bilateral, optical flow-guided.
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
import pickle
import os


class TemporalDepthFilter:
    """Filter depth maps temporally to reduce jitter."""
    
    def __init__(self, method: str = "bilateral", window_size: int = 3, strength: float = 0.5):
        """
        Args:
            method: "median", "bilateral", "gaussian", or "flow-guided"
            window_size: Temporal window size (should be odd)
            strength: Blending strength (0=original, 1=filtered)
        """
        self.method = method
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.strength = strength
    
    def apply_median_filter(self, depth_sequence: np.ndarray) -> np.ndarray:
        """
        Apply temporal median filtering across frames.
        Most robust to outliers.
        """
        T, H, W = depth_sequence.shape
        half_window = self.window_size // 2
        filtered = np.zeros_like(depth_sequence)
        
        for t in range(T):
            # Build temporal window
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            window = depth_sequence[start:end]
            
            # Apply median along temporal axis
            with np.errstate(invalid='ignore'):  # Ignore NaN warnings
                filtered[t] = np.nanmedian(window, axis=0)
        
        # Blend with original
        valid_orig = np.isfinite(depth_sequence)
        filtered[~valid_orig] = depth_sequence[~valid_orig]
        filtered = self.strength * filtered + (1 - self.strength) * depth_sequence
        
        return filtered
    
    def apply_bilateral_filter(self, depth_sequence: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering temporally (preserves edges).
        Better for edges between objects at different depths.
        """
        T, H, W = depth_sequence.shape
        half_window = self.window_size // 2
        filtered = np.zeros_like(depth_sequence)
        valid_mask = (depth_sequence > 0) & np.isfinite(depth_sequence)
        
        for t in range(T):
            # Build temporal window
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            window = depth_sequence[start:end]
            window_mask = valid_mask[start:end]
            
            center_depth = depth_sequence[t].copy()
            
            # For each pixel, compute weighted average in temporal neighborhood
            numerator = np.zeros((H, W), dtype=np.float64)
            denominator = np.zeros((H, W), dtype=np.float64)
            
            for i, frame_idx in enumerate(range(start, end)):
                frame_depth = window[i]
                frame_valid = window_mask[i]
                
                # Spatial distance-like weight (just temporal index difference)
                temporal_weight = np.exp(-((i - half_window) ** 2) / (2 * (half_window / 2) ** 2))
                
                # Range weight (depth similarity)
                depth_diff = np.abs(frame_depth - center_depth)
                depth_diff[~frame_valid] = np.inf
                range_weight = np.exp(-depth_diff / (2 * (np.nanstd(center_depth[center_depth > 0]) + 1e-6) ** 2))
                
                weight = temporal_weight * range_weight
                numerator += weight * frame_depth
                denominator += weight
            
            # Avoid division by zero
            denominator[denominator < 1e-6] = 1.0
            filtered[t] = numerator / denominator
            filtered[t, ~valid_mask[t]] = depth_sequence[t, ~valid_mask[t]]
        
        # Blend
        filtered = self.strength * filtered + (1 - self.strength) * depth_sequence
        return filtered
    
    def apply_gaussian_filter(self, depth_sequence: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian temporal filtering.
        Smooth but doesn't preserve edges as well.
        """
        T, H, W = depth_sequence.shape
        half_window = self.window_size // 2
        sigma_t = half_window / 2.355  # Convert to sigma
        
        # Create 1D Gaussian kernel
        kernel = np.exp(-np.arange(-half_window, half_window + 1) ** 2 / (2 * sigma_t ** 2))
        kernel = kernel / kernel.sum()
        
        filtered = np.zeros_like(depth_sequence)
        valid_mask = (depth_sequence > 0) & np.isfinite(depth_sequence)
        
        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            window = depth_sequence[start:end]
            
            # Pad kernel if needed
            actual_kernel = kernel[half_window - (t - start):half_window + (end - t)]
            actual_kernel = actual_kernel / actual_kernel.sum()  # Renormalize
            
            # Weighted average along time
            result = np.zeros((H, W))
            for i, frame_idx in enumerate(range(start, end)):
                result += actual_kernel[i] * window[i]
            
            filtered[t] = result
            filtered[t, ~valid_mask[t]] = depth_sequence[t, ~valid_mask[t]]
        
        # Blend
        filtered = self.strength * filtered + (1 - self.strength) * depth_sequence
        return filtered
    
    def apply_flow_guided_filter(self, depth_sequence: np.ndarray, rgb_sequence: np.ndarray = None) -> np.ndarray:
        """
        Apply optical flow-guided temporal filtering.
        Uses motion estimation to align depth before filtering.
        """
        if rgb_sequence is None:
            print("Warning: Flow-guided filtering requires RGB sequence, falling back to bilateral")
            return self.apply_bilateral_filter(depth_sequence)
        
        T, H, W = depth_sequence.shape
        half_window = self.window_size // 2
        filtered = np.zeros_like(depth_sequence)
        
        # Compute optical flow between frames
        flows = []
        for t in range(T - 1):
            curr_gray = cv2.cvtColor((rgb_sequence[t] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) if rgb_sequence[t].ndim == 3 else (rgb_sequence[t] * 255).astype(np.uint8)
            next_gray = cv2.cvtColor((rgb_sequence[t + 1] * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) if rgb_sequence[t + 1].ndim == 3 else (rgb_sequence[t + 1] * 255).astype(np.uint8)
            
            flow = cv2.calcOpticalFlowFarneback(curr_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(flow)
        
        # Filter using optical flow
        for t in range(T):
            start = max(0, t - half_window)
            end = min(T, t + half_window + 1)
            
            # Build aligned window using flows
            aligned_frames = [depth_sequence[t]]  # Center frame
            
            for offset in range(1, t - start + 1):
                # Warp forward using flows
                curr_depth = depth_sequence[t - offset].copy()
                for i in range(t - offset, t):
                    if i < len(flows):
                        h, w = flows[i].shape[:2]
                        x, y = np.meshgrid(np.arange(w), np.arange(h))
                        flow = flows[i]
                        x_warped = x + flow[..., 0]
                        y_warped = y + flow[..., 1]
                        curr_depth = cv2.remap(curr_depth, x_warped, y_warped, cv2.INTER_LINEAR)
                aligned_frames.append(curr_depth)
            
            for offset in range(1, end - t):
                curr_depth = depth_sequence[t + offset].copy()
                for i in range(t + offset - 1, t - 1, -1):
                    if i >= 0 and i < len(flows):
                        h, w = flows[i].shape[:2]
                        x, y = np.meshgrid(np.arange(w), np.arange(h))
                        flow = -flows[i]  # Reverse flow
                        x_warped = x + flow[..., 0]
                        y_warped = y + flow[..., 1]
                        curr_depth = cv2.remap(curr_depth, x_warped, y_warped, cv2.INTER_LINEAR)
                aligned_frames.append(curr_depth)
            
            # Median filter across aligned frames
            aligned_stack = np.stack(aligned_frames)
            filtered[t] = np.nanmedian(aligned_stack, axis=0)
        
        # Blend
        filtered = self.strength * filtered + (1 - self.strength) * depth_sequence
        return filtered
    
    def filter(self, depth_sequence: np.ndarray, rgb_sequence: np.ndarray = None) -> np.ndarray:
        """Apply selected filtering method."""
        print(f"Applying {self.method} temporal filtering (window={self.window_size}, strength={self.strength})...")
        
        if self.method == "median":
            return self.apply_median_filter(depth_sequence)
        elif self.method == "bilateral":
            return self.apply_bilateral_filter(depth_sequence)
        elif self.method == "gaussian":
            return self.apply_gaussian_filter(depth_sequence)
        elif self.method == "flow-guided":
            return self.apply_flow_guided_filter(depth_sequence, rgb_sequence)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def process_depth_data(
    input_path: str,
    output_path: str, 
    method: str = "median",
    window_size: int = 5,
    strength: float = 0.7,
) -> Dict[str, any]:
    """
    Process depth data and save filtered results.
    
    Args:
        input_path: Path to original depth data or directory
        output_path: Path to save filtered depth data
        method: Filtering method
        window_size: Temporal window size
        strength: Blending strength
    
    Returns:
        Statistics dictionary
    """
    
    # Load depth maps
    if input_path.endswith('.npy'):
        # Single .npy file
        depth_sequence = np.load(input_path)
    elif input_path.endswith('.pkl'):
        # Pickle file (possibly with dict)
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'depths' in data:
            depth_sequence = np.array(data['depths'])
        else:
            depth_sequence = np.array(data)
    else:
        # Directory with .npy files
        depth_files = sorted(Path(input_path).glob("*.npy"))
        depths = []
        for f in tqdm(depth_files, desc="Loading depths"):
            depths.append(np.load(f))
        depth_sequence = np.stack(depths)
    
    print(f"Loaded depth sequence: {depth_sequence.shape}")
    
    # Apply filtering
    filter_obj = TemporalDepthFilter(method=method, window_size=window_size, strength=strength)
    filtered_sequence = filter_obj.filter(depth_sequence)
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    
    if input_path.endswith('.npy'):
        np.save(os.path.join(output_path, os.path.basename(input_path)), filtered_sequence)
    elif isinstance(input_path, str) and not input_path.endswith('.pkl'):
        # Save individual frames
        for i in tqdm(range(len(filtered_sequence)), desc="Saving filtered depths"):
            np.save(os.path.join(output_path, f"{i:06d}_depth.npy"), filtered_sequence[i])
    
    # Compute statistics
    stats = {
        "original_shape": depth_sequence.shape,
        "filtered_shape": filtered_sequence.shape,
        "method": method,
        "window_size": window_size,
        "strength": strength,
        "mean_change": float(np.mean(np.abs(filtered_sequence - depth_sequence))),
        "max_change": float(np.max(np.abs(filtered_sequence - depth_sequence))),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Temporal depth filtering to reduce jitter")
    parser.add_argument("--input", type=str, required=True, help="Input depth path (file or directory)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for filtered depths")
    parser.add_argument("--method", type=str, default="median", 
                        choices=["median", "bilateral", "gaussian", "flow-guided"],
                        help="Filtering method")
    parser.add_argument("--window", type=int, default=5, help="Temporal window size")
    parser.add_argument("--strength", type=float, default=0.7, help="Filter strength (0-1)")
    args = parser.parse_args()
    
    stats = process_depth_data(
        args.input,
        args.output,
        method=args.method,
        window_size=args.window,
        strength=args.strength
    )
    
    print("\nFiltering complete!")
    print(f"Mean depth change: {stats['mean_change']:.6f}")
    print(f"Max depth change: {stats['max_change']:.6f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
