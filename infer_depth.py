#!/usr/bin/env python3
"""
Wrapper script to run Video-Depth-Anything and convert output to PhysTwin format.

This script:
1. Reads case names from data_config.csv
2. For each case, processes all camera videos (0.mp4, 1.mp4, 2.mp4)
3. Runs Video-Depth-Anything with metric depth estimation
4. Converts the output .npz to individual .npy files per frame
5. Saves to data/vda_depth/{case_name}/{camera_id}/

Usage:
    python infer_depth.py                           # Process all cases from data_config.csv
    python infer_depth.py --case_name my_case       # Process specific case only
    python infer_depth.py --encoder vitb            # Use smaller model
    python infer_depth.py --max_res 720             # Lower resolution for less VRAM
"""

import argparse
import subprocess
import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def read_case_names_from_csv(csv_path):
    """Read case names from data_config.csv"""
    case_names = []
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                case_names.append(parts[0])
    return case_names


def process_video(vda_dir, input_video, output_dir, camera_id, args):
    """Process a single video and convert to npy files"""
    
    temp_output_dir = output_dir / '_temp'
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    camera_output_dir = output_dir / str(camera_id)
    camera_output_dir.mkdir(parents=True, exist_ok=True)
    
    video_name = Path(input_video).stem
    
    # Build Video-Depth-Anything command
    cmd = [
        'python', str(vda_dir / 'run.py'),
        '--input_video', str(input_video),
        '--output_dir', str(temp_output_dir),
        '--encoder', args.encoder,
        '--max_res', str(args.max_res),
        '--input_size', str(args.input_size),
        # '--metric',  # Use metric depth
        '--save_npz',  # Save as npz for conversion
    ]
    
    if args.max_len > 0:
        cmd.extend(['--max_len', str(args.max_len)])
    if args.target_fps > 0:
        cmd.extend(['--target_fps', str(args.target_fps)])
    if args.fp32:
        cmd.append('--fp32')
    
    print(f"  Processing camera {camera_id}: {input_video}")
    print(f"  Command: {' '.join(cmd)}")
    
    # Run Video-Depth-Anything
    result = subprocess.run(cmd, cwd=str(vda_dir))
    if result.returncode != 0:
        print(f"  Error: Video-Depth-Anything failed with return code {result.returncode}")
        return False
    
    # Find the .npz file
    npz_path = temp_output_dir / f'{video_name}_depths.npz'
    if not npz_path.exists():
        print(f"  Error: Expected output file not found: {npz_path}")
        return False
    
    print(f"  Converting {npz_path} to individual .npy files...")
    
    # Load npz and save as individual npy files
    data = np.load(npz_path)
    depths = data['depths']
    
    # Invert depth if requested (VDA may output inverse depth)
    if args.invert:
        print(f"  Inverting depth values...")
        for i in range(len(depths)):
            valid_mask = (depths[i] > 0) & np.isfinite(depths[i])
            if valid_mask.any():
                max_val = depths[i][valid_mask].max()
                depths[i] = max_val - depths[i]
                depths[i][depths[i] < 0] = 0
    
    for i, depth in enumerate(depths):
        npy_path = camera_output_dir / f'{i}.npy'
        np.save(npy_path, depth)
    
    print(f"  Saved {len(depths)} depth frames to {camera_output_dir}")
    
    # Move visualization video to output directory
    vis_video = temp_output_dir / f'{video_name}_vis.mp4'
    if vis_video.exists():
        dest_video = output_dir / f'depth_vis_{camera_id}.mp4'
        shutil.move(str(vis_video), str(dest_video))
        print(f"  Saved depth visualization video to {dest_video}")
    
    # Cleanup temp files
    for f in temp_output_dir.glob(f'{video_name}*'):
        if f.exists():
            f.unlink()
    
    return True


def process_case(case_name, script_dir, vda_dir, args):
    """Process all cameras for a single case"""
    
    print(f"\n{'='*60}")
    print(f"Processing case: {case_name}")
    print(f"{'='*60}")
    
    # Input: data/different_types/{case_name}/color/{0,1,2}.mp4
    color_dir = script_dir / 'data' / 'different_types' / case_name / 'color'
    
    if not color_dir.exists():
        print(f"  Error: Color directory not found: {color_dir}")
        return False
    
    # Output: data/vda_depth/{case_name}/
    output_dir = script_dir / 'data' / 'vda_depth' / case_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each camera (0, 1, 2)
    success_count = 0
    for camera_id in [0, 1, 2]:
        video_path = color_dir / f'{camera_id}.mp4'
        if video_path.exists():
            if process_video(vda_dir, video_path, output_dir, camera_id, args):
                success_count += 1
        else:
            print(f"  Warning: Camera {camera_id} video not found: {video_path}")
    
    # Cleanup temp directory
    temp_dir = output_dir / '_temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print(f"\nCase {case_name}: Processed {success_count}/3 cameras successfully")
    print(f"Output saved to: {output_dir}")
    
    return success_count > 0


def main():
    parser = argparse.ArgumentParser(description='Infer depth using Video-Depth-Anything and convert to PhysTwin format')
    parser.add_argument('--case_name', type=str, default=None, help='Process specific case only (default: all from data_config.csv)')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help='Model encoder size (default: vitl)')
    parser.add_argument('--max_res', type=int, default=1280, help='Maximum resolution (default: 1280)')
    parser.add_argument('--input_size', type=int, default=518, help='Model input size (default: 518)')
    parser.add_argument('--max_len', type=int, default=-1, help='Maximum number of frames (-1 = all)')
    parser.add_argument('--target_fps', type=int, default=-1, help='Target FPS (-1 = original)')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')
    parser.add_argument('--invert', action='store_true', help='Invert depth values (use if VDA outputs inverse depth)')
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent.resolve()
    vda_dir = script_dir / 'Video-Depth-Anything'
    csv_path = script_dir / 'data_config.csv'
    
    # Determine which cases to process
    if args.case_name:
        case_names = [args.case_name]
    else:
        if not csv_path.exists():
            print(f"Error: data_config.csv not found at {csv_path}")
            print("Either create data_config.csv or specify --case_name")
            return 1
        case_names = read_case_names_from_csv(csv_path)
    
    print(f"Will process {len(case_names)} case(s): {case_names}")
    print(f"Encoder: {args.encoder}, Max res: {args.max_res}, Input size: {args.input_size}")
    
    # Process each case
    success_count = 0
    for case_name in tqdm(case_names):
        if process_case(case_name, script_dir, vda_dir, args):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Completed: {success_count}/{len(case_names)} cases processed successfully")
    print(f"{'='*60}")
    
    return 0 if success_count == len(case_names) else 1


if __name__ == '__main__':
    exit(main())

