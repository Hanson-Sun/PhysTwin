output_dir="./gaussian_output"
output_video_dir="./gaussian_output_video"

# Read scenes from data_config.csv (first column is case_name)
scenes=()
while IFS=',' read -r case_name object_type double_hand || [ -n "$case_name" ]; do
    # Skip empty lines
    if [ -n "$case_name" ]; then
        scenes+=("$case_name")
    fi
done < data_config.csv

exp_name="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"

python ./gaussian_splatting/generate_interp_poses.py

# Iterate over each folder
for scene_name in "${scenes[@]}"; do
    echo "Processing: $scene_name"

    # Training
    python gs_train.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ${output_dir}/${scene_name}/${exp_name} \
        --iterations 10000 \
        --lambda_depth 0.001 \
        --lambda_normal 0.0 \
        --lambda_anisotropic 0.0 \
        --lambda_seg 1.0 \
        --use_masks \
        --isotropic \
        --gs_init_opt 'hybrid'

    # Rendering
    python gs_render.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ${output_dir}/${scene_name}/${exp_name} \

    # Convert images to video
    python gaussian_splatting/img2video.py \
        --image_folder ${output_dir}/${scene_name}/${exp_name}/test/ours_10000/renders \
        --video_path ${output_video_dir}/${scene_name}/${exp_name}.mp4
done
