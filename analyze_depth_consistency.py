#!/usr/bin/env python3
"""
Analyze depth map consistency and jitter across frames.
Identifies problematic frames and temporal instability.
"""

import numpy as np
import cv2
import os
import json
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_depth_statistics(depth_maps: np.ndarray) -> dict:
    """
    Compute temporal consistency statistics for depth maps.
    
    Args:
        depth_maps: (T, H, W) array of depth maps
    
    Returns:
        Dictionary with metrics
    """
    T, H, W = depth_maps.shape
    
    # Remove invalid depths (0 or inf)
    valid_mask = (depth_maps > 0) & np.isfinite(depth_maps)
    
    # Temporal variance (frame-to-frame changes)
    temporal_diffs = []
    for t in range(T - 1):
        curr = depth_maps[t]
        next_frame = depth_maps[t + 1]
        
        # Pixelwise difference where both are valid
        mask = valid_mask[t] & valid_mask[t + 1]
        if mask.sum() > 0:
            diff = np.abs(curr[mask] - next_frame[mask])
            temporal_diffs.append(diff)
    
    if temporal_diffs:
        temporal_diffs = np.concatenate(temporal_diffs)
        temporal_variance = {
            "mean": float(np.mean(temporal_diffs)),
            "std": float(np.std(temporal_diffs)),
            "median": float(np.median(temporal_diffs)),
            "max": float(np.max(temporal_diffs)),
            "p95": float(np.percentile(temporal_diffs, 95)),
        }
    else:
        temporal_variance = {k: 0.0 for k in ["mean", "std", "median", "max", "p95"]}
    
    # Per-frame statistics
    per_frame_stats = []
    for t in range(T):
        depth = depth_maps[t]
        mask = valid_mask[t]
        
        if mask.sum() > 0:
            valid_depths = depth[mask]
            per_frame_stats.append({
                "frame": t,
                "valid_pixels": int(mask.sum()),
                "mean_depth": float(np.mean(valid_depths)),
                "std_depth": float(np.std(valid_depths)),
                "min_depth": float(np.min(valid_depths)),
                "max_depth": float(np.max(valid_depths)),
            })
    
    return {
        "num_frames": T,
        "temporal_variance": temporal_variance,
        "per_frame_stats": per_frame_stats,
    }


def detect_jittery_frames(depth_maps: np.ndarray, threshold_std: float = 0.05) -> dict:
    """
    Detect frames with unusually high variance (indicating jitter).
    
    Args:
        depth_maps: (T, H, W) array
        threshold_std: std of depth changes to flag as jittery
    
    Returns:
        Dictionary with jitter analysis
    """
    T, H, W = depth_maps.shape
    valid_mask = (depth_maps > 0) & np.isfinite(depth_maps)
    
    jittery_frames = []
    
    for t in range(T - 1):
        curr = depth_maps[t]
        next_frame = depth_maps[t + 1]
        
        mask = valid_mask[t] & valid_mask[t + 1]
        if mask.sum() > 0:
            diff = np.abs(curr[mask] - next_frame[mask])
            # Check local variance (high jitter = high local variance)
            local_std = np.std(diff)
            
            if local_std > threshold_std:
                jittery_frames.append({
                    "frame_pair": (t, t + 1),
                    "jitter_std": float(local_std),
                    "max_change": float(np.max(diff)),
                    "mean_change": float(np.mean(diff)),
                })
    
    return {
        "num_jittery_pairs": len(jittery_frames),
        "jitter_percent": 100.0 * len(jittery_frames) / (T - 1) if T > 1 else 0.0,
        "jittery_frames": jittery_frames,
    }


def analyze_depth_smoothness(depth_maps: np.ndarray, window_size: int = 3) -> dict:
    """
    Analyze local depth smoothness (gradient magnitude).
    High gradients with high variance indicate noisy regions.
    """
    T, H, W = depth_maps.shape
    valid_mask = (depth_maps > 0) & np.isfinite(depth_maps)
    
    smoothness_scores = []
    
    for t in range(T):
        depth = depth_maps[t]
        mask = valid_mask[t]
        
        if mask.sum() > window_size:
            # Compute spatial gradients
            gy, gx = np.gradient(depth)
            grad_mag = np.sqrt(gx**2 + gy**2)
            
            # Mask gradients by validity
            grad_mag[~mask] = 0
            
            smoothness_scores.append({
                "frame": t,
                "mean_grad": float(np.mean(grad_mag[mask])),
                "std_grad": float(np.std(grad_mag[mask])),
                "high_grad_ratio": float(np.sum(grad_mag[mask] > np.percentile(grad_mag[mask], 95)) / mask.sum()),
            })
    
    return {
        "per_frame_smoothness": smoothness_scores,
        "mean_smoothness": float(np.mean([s["mean_grad"] for s in smoothness_scores])),
    }


def load_depth_sequence(depth_dir: str, case_name: str, camera_id: int, max_frames: int = None) -> np.ndarray:
    """Load a sequence of depth maps."""
    depth_files = sorted(Path(depth_dir).glob(f"*_depth.npy"))
    
    if not depth_files:
        raise ValueError(f"No depth files found in {depth_dir}")
    
    depths = []
    for i, depth_file in enumerate(depth_files):
        if max_frames and i >= max_frames:
            break
        try:
            depth = np.load(str(depth_file))
            if isinstance(depth, dict):  # Some formats store in dict
                depth = depth.get('depth', depth.get('depth_map', depth))
            depths.append(depth)
        except Exception as e:
            print(f"Warning: Could not load {depth_file}: {e}")
            continue
    
    if not depths:
        raise ValueError("No valid depth maps loaded")
    
    return np.stack(depths)


def main():
    parser = ArgumentParser()
    parser.add_argument("--depth_dir", type=str, required=True, help="Directory containing depth maps")
    parser.add_argument("--case_name", type=str, help="Case name (for output naming)")
    parser.add_argument("--output_json", type=str, default="depth_consistency_analysis.json")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--max_frames", type=int, help="Limit number of frames to analyze")
    args = parser.parse_args()
    
    print(f"Loading depth maps from {args.depth_dir}...")
    try:
        depth_maps = load_depth_sequence(args.depth_dir, args.case_name or "unknown", 0, args.max_frames)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print(f"Loaded {depth_maps.shape[0]} depth maps of shape {depth_maps.shape[1:]}")
    
    # Analyze consistency
    print("\n=== Computing depth statistics ===")
    stats = compute_depth_statistics(depth_maps)
    
    print(f"Temporal variance stats:")
    for key, val in stats["temporal_variance"].items():
        print(f"  {key}: {val:.6f}")
    
    # Detect jitter
    print("\n=== Detecting jitter ===")
    jitter_analysis = detect_jittery_frames(depth_maps, threshold_std=0.05)
    print(f"Jittery frame pairs: {jitter_analysis['num_jittery_pairs']} ({jitter_analysis['jitter_percent']:.1f}%)")
    
    if jitter_analysis['jittery_frames']:
        print("  Top 5 most jittery transitions:")
        for item in sorted(jitter_analysis['jittery_frames'], key=lambda x: x['jitter_std'], reverse=True)[:5]:
            print(f"    Frames {item['frame_pair']}: jitter_std={item['jitter_std']:.6f}")
    
    # Smoothness analysis
    print("\n=== Analyzing depth smoothness ===")
    smoothness = analyze_depth_smoothness(depth_maps)
    print(f"Mean spatial gradient: {smoothness['mean_smoothness']:.6f}")
    
    # Save results
    results = {
        "case_name": args.case_name,
        "num_frames": depth_maps.shape[0],
        "depth_shape": list(depth_maps.shape),
        "statistics": stats,
        "jitter_analysis": jitter_analysis,
        "smoothness": smoothness,
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")
    
    # Visualize if requested
    if args.visualize:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Temporal variance over time
        per_frame = stats["per_frame_stats"]
        frames = [s["frame"] for s in per_frame]
        means = [s["mean_depth"] for s in per_frame]
        stds = [s["std_depth"] for s in per_frame]
        
        axes[0, 0].plot(frames, means, label="Mean depth")
        axes[0, 0].fill_between(frames, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.3)
        axes[0, 0].set_xlabel("Frame")
        axes[0, 0].set_ylabel("Depth (meters)")
        axes[0, 0].set_title("Temporal depth statistics")
        axes[0, 0].legend()
        
        # Jitter detection
        if jitter_analysis['jittery_frames']:
            jitter_frames = [j["frame_pair"][0] for j in jitter_analysis['jittery_frames']]
            jitter_stds = [j["jitter_std"] for j in jitter_analysis['jittery_frames']]
            axes[0, 1].scatter(jitter_frames, jitter_stds, alpha=0.5)
            axes[0, 1].set_xlabel("Frame")
            axes[0, 1].set_ylabel("Jitter (std)")
            axes[0, 1].set_title(f"Detected jitter ({jitter_analysis['jitter_percent']:.1f}%)")
        
        # Smoothness
        smoothness_frames = [s["frame"] for s in smoothness["per_frame_smoothness"]]
        smoothness_means = [s["mean_grad"] for s in smoothness["per_frame_smoothness"]]
        axes[1, 0].plot(smoothness_frames, smoothness_means)
        axes[1, 0].set_xlabel("Frame")
        axes[1, 0].set_ylabel("Mean gradient")
        axes[1, 0].set_title("Spatial smoothness over time")
        
        # Depth map range over time
        valid_depth_ranges = []
        for s in per_frame:
            depth_range = s["max_depth"] - s["min_depth"]
            valid_depth_ranges.append(depth_range)
        axes[1, 1].plot(frames, valid_depth_ranges)
        axes[1, 1].set_xlabel("Frame")
        axes[1, 1].set_ylabel("Depth range (meters)")
        axes[1, 1].set_title("Depth range per frame")
        
        plt.tight_layout()
        viz_path = args.output_json.replace(".json", "_viz.png")
        plt.savefig(viz_path, dpi=100)
        print(f"Visualization saved to {viz_path}")


if __name__ == "__main__":
    main()
