#!/usr/bin/env python3
"""
Quick depth quality checker - run this to validate depth maps before training.
"""

import numpy as np
import cv2
from pathlib import Path
import json
import sys
from argparse import ArgumentParser


def quick_check(depth_dir: str, num_samples: int = 10) -> dict:
    """
    Quick sanity check on depth maps.
    Returns: True if depths look good, False if problematic.
    """
    depth_files = sorted(Path(depth_dir).glob("*.npy"))[:num_samples]
    
    if not depth_files:
        return {"status": "ERROR", "message": "No depth files found"}
    
    results = {
        "num_files_checked": len(depth_files),
        "has_jitter": False,
        "has_outliers": False,
        "has_nan_inf": False,
        "issues": [],
    }
    
    jitter_scores = []
    
    for i in range(len(depth_files) - 1):
        d1 = np.load(depth_files[i])
        d2 = np.load(depth_files[i + 1])
        
        # Check for NaN/Inf
        if np.any(~np.isfinite(d1)) or np.any(~np.isfinite(d2)):
            results["has_nan_inf"] = True
            results["issues"].append(f"Frame {i}: Contains NaN or Inf values")
        
        # Check for jitter
        valid_mask = (d1 > 0) & (d2 > 0) & np.isfinite(d1) & np.isfinite(d2)
        if valid_mask.sum() > 100:
            frame_diff = np.abs(d1[valid_mask] - d2[valid_mask])
            jitter = np.std(frame_diff)
            jitter_scores.append(jitter)
            
            if jitter > 0.05:  # Threshold: 5cm temporal std
                results["has_jitter"] = True
                results["issues"].append(
                    f"Frame {i}-{i+1}: High jitter (std={jitter:.6f})"
                )
        
        # Check for outliers
        for depth in [d1, d2]:
            valid = depth[depth > 0]
            if len(valid) > 100:
                q1 = np.percentile(valid, 25)
                q3 = np.percentile(valid, 75)
                iqr = q3 - q1
                outliers = ((valid < q1 - 1.5*iqr) | (valid > q3 + 1.5*iqr)).sum()
                outlier_ratio = outliers / len(valid)
                
                if outlier_ratio > 0.1:  # >10% outliers
                    results["has_outliers"] = True
    
    # Summary
    if jitter_scores:
        results["jitter_stats"] = {
            "mean": float(np.mean(jitter_scores)),
            "max": float(np.max(jitter_scores)),
            "std": float(np.std(jitter_scores)),
        }
    
    # Overall status
    if results["has_nan_inf"]:
        results["status"] = "FAIL: Contains NaN/Inf"
    elif results["has_jitter"] and results["jitter_stats"]["mean"] > 0.1:
        results["status"] = "WARNING: Severe jitter detected"
    elif results["has_outliers"]:
        results["status"] = "WARNING: Many outliers"
    else:
        results["status"] = "OK"
    
    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()
    
    print(f"Checking depths in {args.depth_dir}...")
    results = quick_check(args.depth_dir, num_samples=args.samples)
    
    print(f"\n{'='*60}")
    print(f"Status: {results['status']}")
    print(f"Files checked: {results['num_files_checked']}")
    
    if "jitter_stats" in results:
        jst = results["jitter_stats"]
        print(f"Jitter (temporal std):")
        print(f"  Mean: {jst['mean']:.6f}m")
        print(f"  Max:  {jst['max']:.6f}m")
        print(f"  Std:  {jst['std']:.6f}m")
        
        if jst["mean"] > 0.05:
            print("\n⚠️  RECOMMENDATION: Apply temporal smoothing!")
            print("   python temporal_depth_filter.py \\")
            print(f"       --input {args.depth_dir} \\")
            print(f"       --output {args.depth_dir}_smoothed \\")
            print("       --method median --window 5 --strength 0.7")
    
    if results["issues"]:
        print(f"\nIssues found ({len(results['issues'])}):")
        for issue in results["issues"][:5]:
            print(f"  - {issue}")
        if len(results["issues"]) > 5:
            print(f"  ... and {len(results['issues']) - 5} more")
    
    print('='*60)
    
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.json_out}")
    
    # Exit code
    if "FAIL" in results["status"]:
        sys.exit(1)
    elif "WARNING" in results["status"]:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
