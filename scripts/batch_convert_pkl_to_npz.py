#!/usr/bin/env python3
"""
Batch convert all inference.pkl files to npz format.
Reads cases from data_config_old.csv and processes each one.
"""

import os
import sys
import csv
import subprocess
from pathlib import Path


def main():
    # Paths
    workspace_root = Path(__file__).parent.parent
    config_file = workspace_root / "data_config_old.csv"
    source_base = Path("/mnt/d/DATA/phystwin/experiments")
    output_base = workspace_root / "inference_data"
    script_path = Path(__file__).parent / "final_pkl_to_npz.py"

    # Verify script exists
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    # Verify config file exists
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    # Create output base directory
    output_base.mkdir(parents=True, exist_ok=True)

    # Read cases from config
    cases = []
    with open(config_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Skip empty rows
                case_name = row[0].strip()  # Remove whitespace
                if case_name:
                    cases.append(case_name)

    print(f"Found {len(cases)} cases in config file")
    print(f"Source base: {source_base}")
    print(f"Output base: {output_base}")
    print()

    # Process each case
    successful = 0
    failed = 0
    skipped = 0

    for i, case_name in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] Processing: {case_name}")

        # Source file
        source_file = source_base / case_name / "inference.pkl"

        if not source_file.exists():
            print(f"  ⚠ Source file not found: {source_file}")
            skipped += 1
            print()
            continue

        # Output directory and file
        output_dir = output_base / case_name / "0"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "0.npz"

        # Run conversion
        try:
            # Controller data path
            controller_data_path = source_base.parent / "data" / "different_types" / case_name / "final_data.pkl"
            
            cmd = [
                sys.executable,
                str(script_path),
                "--input", str(source_file),
                "--output", str(output_file),
                "--controller-data", str(controller_data_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"  ✓ Successfully converted to: {output_file}")
                successful += 1
            else:
                print(f"  ✗ Conversion failed for {case_name}")
                print(f"    Error: {result.stderr}")
                failed += 1

        except subprocess.TimeoutExpired:
            print(f"  ✗ Conversion timed out for {case_name}")
            failed += 1
        except Exception as e:
            print(f"  ✗ Error processing {case_name}: {e}")
            failed += 1

        print()

    # Summary
    print("=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total cases: {len(cases)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (file not found): {skipped}")
    print()

    if failed == 0 and skipped == 0:
        print("✓ All cases processed successfully!")
        return 0
    elif failed > 0:
        print(f"⚠ {failed} case(s) failed to convert")
        return 1
    else:
        print(f"⚠ {skipped} case(s) were skipped (source files not found)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
