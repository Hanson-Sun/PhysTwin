#!/bin/bash

# INPUT_PATH="/mnt/d/DATA/phystwin/data/different_types"
INPUT_PATH="/mnt/d/DATA/phystwin/processed_raw_collected_data"
OUTPUT_PATH="/mnt/d/DATA/phystwin/temporal_depth_training_data_v2"

# python convert_phystwin_data.py --source_dir "$INPUT_PATH" --output_dir "$OUTPUT_PATH"

python infer_da3_simple.py --data_dir "$OUTPUT_PATH"

python infer_vda_simple.py --data_dir "$OUTPUT_PATH"