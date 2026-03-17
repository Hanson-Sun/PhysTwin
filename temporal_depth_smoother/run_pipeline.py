#!/usr/bin/env python
"""Complete pipeline: preprocess → train → evaluate"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully\n")
        return True
    else:
        print(f"\n❌ {description} failed\n")
        return False


def main():
    parser = argparse.ArgumentParser(description='Complete training pipeline')
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./temporal_depth_output', help='Output directory')
    parser.add_argument('--bg-threshold', type=float, default=0.01, help='Background threshold')
    parser.add_argument('--visualize', action='store_true', help='Visualize preprocessing')
    parser.add_argument('--num-workers', type=int, default=1, help='Preprocessing workers')
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing')
    parser.add_argument('--skip-existing', action='store_true', help='Skip clips if already preprocessed')
    parser.add_argument('--skip-train', action='store_true', help='Skip training')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation')
    
    args = parser.parse_args()
    
    # Setup directories
    output_root = Path(args.output_dir)
    preprocess_dir = output_root / 'preprocessed'
    train_dir = output_root / 'training'
    eval_dir = output_root / 'evaluation'
    
    # Print configuration
    print(f"\n{'='*70}")
    print(f"  TEMPORAL DEPTH SMOOTHER - COMPLETE PIPELINE")
    print(f"{'='*70}")
    print(f"\n📁 Directories:")
    print(f"   Input data:      {args.data_dir}")
    print(f"   Output root:     {output_root}")
    print(f"   Preprocessed:    {preprocess_dir}")
    print(f"   Training:        {train_dir}")
    print(f"   Evaluation:      {eval_dir}")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Background threshold: {args.bg_threshold}")
    print(f"   Preprocessing workers: {args.num_workers}")
    print(f"   Visualize:            {args.visualize}")
    print(f"\n   ℹ️  Training hyperparameters are configured in config.py")
    
    print(f"\n🔄 Pipeline Steps:")
    print(f"   1. Preprocessing: {'SKIP' if args.skip_preprocess else 'RUN'}")
    print(f"   2. Training:      {'SKIP' if args.skip_train else 'RUN'}")
    print(f"   3. Evaluation:    {'SKIP' if args.skip_eval else 'RUN'}")
    print(f"\n{'='*70}\n")
    
    # Preprocess
    if not args.skip_preprocess:
        print("🔨 STEP 1: PREPROCESSING\n")
        print(f"   Scanning {args.data_dir}...")
        print(f"   Preprocessing with {args.num_workers} workers...")
        
        cmd = [
            sys.executable, '-m', 'temporal_depth_smoother.preprocess',
            '--batch-dir', args.data_dir,
            '--output-dir', str(preprocess_dir),
            '--bg-threshold', str(args.bg_threshold),
            '--num-workers', str(args.num_workers),
        ]
        if args.skip_existing:
            cmd.append('--skip-existing')
            print(f"   Skipping already preprocessed clips...")
        
        if not run_command(cmd, 'PREPROCESSING'):
            return False
    else:
        print("⊘ STEP 1: PREPROCESSING SKIPPED (using existing data)\n")
    
    # Train
    if not args.skip_train:
        print("🎓 STEP 2: TRAINING\n")
        print(f"   Loading preprocessed data from {preprocess_dir}...")
        print(f"   Starting training (using config.py for hyperparameters)...")
        print(f"   Monitor with: tensorboard --logdir {train_dir}/tensorboard")
        
        cmd = [
            sys.executable, '-m', 'temporal_depth_smoother.train',
            '--data-dir', str(preprocess_dir),
            '--output-dir', str(train_dir),
        ]
        if not run_command(cmd, 'TRAINING'):
            return False
    else:
        print("⊘ STEP 2: TRAINING SKIPPED\n")
    
    # Evaluate
    checkpoint = train_dir / 'best_model.pt'
    if not args.skip_eval:
        if checkpoint.exists():
            print("📊 STEP 3: EVALUATION\n")
            print(f"   Loading checkpoint: {checkpoint}")
            print(f"   Evaluating on preprocessed data...")
            
            cmd = [
                sys.executable, '-m', 'temporal_depth_smoother.evaluate',
                '--checkpoint', str(checkpoint),
                '--data-dir', str(preprocess_dir),
                '--output-dir', str(eval_dir),
                '--visualize',
            ]
            run_command(cmd, 'EVALUATION')
        else:
            print("⊘ STEP 3: EVALUATION SKIPPED (checkpoint not found)\n")
    else:
        print("⊘ STEP 3: EVALUATION SKIPPED\n")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  ✓ PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\n📦 Results:")
    print(f"   Preprocessed data: {preprocess_dir}")
    if checkpoint.exists():
        print(f"   Best model:        {checkpoint}")
    print(f"   Training logs:     {train_dir / 'tensorboard'}")
    if eval_dir.exists():
        print(f"   Evaluation results: {eval_dir}")
    
    if not args.skip_train and checkpoint.exists():
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review TensorBoard: tensorboard --logdir {train_dir}/tensorboard")
        print(f"   2. Use model for inference:")
        print(f"      python -m temporal_depth_smoother.inference")
        print(f"        --checkpoint {checkpoint}")
        print(f"        --depth-input <your_depth.npy>")
        print(f"        --rgb-input <your_rgb.npy>")
        print(f"        --output <output_depth.npy>")
    
    print(f"\n{'='*70}\n")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
