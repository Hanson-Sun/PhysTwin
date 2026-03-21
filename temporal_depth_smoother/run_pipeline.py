#!/usr/bin/env python
"""Complete pipeline: preprocess → train → evaluate"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, cwd=Path.cwd())
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully\n")
        return True
    else:
        print(f"\n❌ {description} failed\n")
        return False


def find_eval_clips(preprocess_dir: Path) -> list:
    """
    Find all complete clips in the preprocessed directory.
    Returns list of dicts with paths for depth_raw, rgb, depth_vda_aligned.
    """
    clips = []
    for depth_file in sorted(preprocess_dir.glob("*_depth_raw.npy")):
        clip_id = depth_file.stem.replace("_depth_raw", "")
        rgb_path = preprocess_dir / f"{clip_id}_rgb.npy"
        vda_path = preprocess_dir / f"{clip_id}_depth_vda_aligned.npy"

        if rgb_path.exists() and vda_path.exists():
            clips.append(
                {
                    "clip_id": clip_id,
                    "depth": depth_file,
                    "rgb": rgb_path,
                    "vda": vda_path,
                }
            )
        else:
            print(f"   Warning: incomplete files for {clip_id}, skipping eval")
    return clips


def main():
    parser = argparse.ArgumentParser(description="Complete training pipeline")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="./temporal_depth_output")
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.05,
        help="RGB motion detection threshold for evaluation",
    )
    parser.add_argument(
        "--depth-threshold",
        type=float,
        default=0.01,
        help="Normalised depth change threshold for flicker detection",
    )
    parser.add_argument(
        "--vda-weight",
        type=float,
        default=0.7,
        help="How much to trust VDA vs RGB in flicker mask (0-1)",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--eval-max-clips",
        type=int,
        default=None,
        help="Max clips to evaluate (useful for quick checks)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from best_model.pt checkpoint",
    )

    args = parser.parse_args()

    output_root = Path(args.output_dir)
    preprocess_dir = output_root / "preprocessed"
    train_dir = output_root / "training"
    eval_dir = output_root / "evaluation"

    print(f"\n{'=' * 70}")
    print(f"  TEMPORAL DEPTH SMOOTHER - COMPLETE PIPELINE")
    print(f"{'=' * 70}")
    print(f"\n📁 Directories:")
    print(f"   Input data:      {args.data_dir}")
    print(f"   Output root:     {output_root}")
    print(f"   Preprocessed:    {preprocess_dir}")
    print(f"   Training:        {train_dir}")
    print(f"   Evaluation:      {eval_dir}")
    print(f"\n⚙️  Configuration:")
    print(f"   Motion threshold:      {args.motion_threshold}")
    print(f"   Depth threshold:       {args.depth_threshold}")
    print(f"   VDA weight:            {args.vda_weight}")
    print(f"   Preprocessing workers: {args.num_workers}")
    print(f"   Visualize:             {args.visualize}")
    print(f"   Resume training:       {args.resume}")
    print(f"\n   ℹ️  Training hyperparameters are configured in config.py")
    print(f"\n🔄 Pipeline Steps:")
    print(f"   1. Preprocessing: {'SKIP' if args.skip_preprocess else 'RUN'}")
    print(f"   2. Training:      {'SKIP' if args.skip_train else 'RUN'}")
    print(f"   3. Evaluation:    {'SKIP' if args.skip_eval else 'RUN'}")
    print(f"\n{'=' * 70}\n")

    # ── Step 1: Preprocess ────────────────────────────────────────────────
    if not args.skip_preprocess:
        print("🔨 STEP 1: PREPROCESSING\n")
        cmd = [
            sys.executable,
            "-m",
            "temporal_depth_smoother.preprocess",
            "--batch-dir",
            args.data_dir,
            "--output-dir",
            str(preprocess_dir),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.skip_existing:
            cmd.append("--skip-existing")
        if not run_command(cmd, "PREPROCESSING"):
            return False
    else:
        print("⊘ STEP 1: PREPROCESSING SKIPPED (using existing data)\n")

    # ── Step 2: Train ─────────────────────────────────────────────────────
    if not args.skip_train:
        print("🎓 STEP 2: TRAINING\n")
        cmd = [
            sys.executable,
            "-m",
            "temporal_depth_smoother.train",
            "--data-dir",
            str(preprocess_dir),
            "--output-dir",
            str(train_dir),
        ]
        if args.resume:
            checkpoint = train_dir / "best_model.pt"
            if checkpoint.exists():
                cmd += ["--resume", str(checkpoint)]
                print(f"   Resuming from {checkpoint}")
            else:
                print(
                    f"   ⚠ --resume set but no checkpoint found at {checkpoint}, starting fresh"
                )
        if not run_command(cmd, "TRAINING"):
            return False
    else:
        print("⊘ STEP 2: TRAINING SKIPPED\n")

    # ── Step 3: Evaluate ──────────────────────────────────────────────────
    checkpoint = train_dir / "best_model.pt"
    if not args.skip_eval:
        if not checkpoint.exists():
            print("⊘ STEP 3: EVALUATION SKIPPED (checkpoint not found)\n")
        else:
            print("📊 STEP 3: EVALUATION\n")
            print(f"   Loading checkpoint: {checkpoint}")

            clips = find_eval_clips(preprocess_dir)
            if not clips:
                print("   ❌ No complete clips found for evaluation.")
                return False

            if args.eval_max_clips:
                clips = clips[: args.eval_max_clips]
                print(
                    f"   Evaluating on {len(clips)} clips (--eval-max-clips={args.eval_max_clips})"
                )
            else:
                print(f"   Evaluating on {len(clips)} clips")

            all_ok = True
            for clip in clips:
                clip_eval_dir = eval_dir / clip["clip_id"]
                print(f"\n   ── {clip['clip_id']} ──")

                cmd = [
                    sys.executable,
                    "-m",
                    "temporal_depth_smoother.evaluate",
                    "--checkpoint",
                    str(checkpoint),
                    "--depth-input",
                    str(clip["depth"]),
                    "--rgb-input",
                    str(clip["rgb"]),
                    "--vda-input",
                    str(clip["vda"]),
                    "--output-dir",
                    str(clip_eval_dir),
                    "--motion-threshold",
                    str(args.motion_threshold),
                    "--depth-threshold",
                    str(args.depth_threshold),
                    "--vda-weight",
                    str(args.vda_weight),
                    "--save-smoothed",
                ]
                if args.visualize:
                    cmd.append("--visualize")

                if not run_command(cmd, f"EVALUATION — {clip['clip_id']}"):
                    print(
                        f"   ⚠ Evaluation failed for {clip['clip_id']}, continuing..."
                    )
                    all_ok = False

            if not all_ok:
                print("\n⚠ Some clips failed evaluation — check logs above.")
    else:
        print("⊘ STEP 3: EVALUATION SKIPPED\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  ✓ PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n📦 Results:")
    print(f"   Preprocessed data:  {preprocess_dir}")
    if checkpoint.exists():
        print(f"   Best model:         {checkpoint}")
    print(f"   Training logs:      {train_dir / 'tensorboard'}")
    if eval_dir.exists():
        print(f"   Evaluation results: {eval_dir}/<clip_id>/")
        print(f"     metrics.json       — per-clip metrics")
        if args.visualize:
            print(f"     visualization.png  — before/after depth frames")
        print(f"     depth_smooth.npy   — smoothed depth output")

    if not args.skip_train and checkpoint.exists():
        print(f"\n🚀 Next Steps:")
        print(f"   1. Review TensorBoard: tensorboard --logdir {train_dir}/tensorboard")
        print(f"   2. Use model for inference:")
        print(f"      python -m temporal_depth_smoother.inference \\")
        print(f"        --checkpoint {checkpoint} \\")
        print(f"        --depth-input <your_depth.npy> \\")
        print(f"        --rgb-input <your_rgb.npy> \\")
        print(f"        --output <output_depth.npy>")

    print(f"\n{'=' * 70}\n")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
