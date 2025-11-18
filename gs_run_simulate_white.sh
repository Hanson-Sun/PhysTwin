output_dir="./gaussian_output_dynamic_white"

# views=("0" "1" "2")
views=("0")

# Read scenes from data_config.csv (first column is case_name)
scenes=()
while IFS=',' read -r case_name object_type double_hand || [ -n "$case_name" ]; do
    # Skip empty lines
    if [ -n "$case_name" ]; then
        scenes+=("$case_name")
    fi
done < data_config.csv

exp_name='init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0'

for scene_name in "${scenes[@]}"; do

    python gs_render_dynamics.py \
        -s ./data/gaussian_data/${scene_name} \
        -m ./gaussian_output/${scene_name}/${exp_name} \
        --name ${scene_name} \
        --white_background
        --output_dir ${output_dir}/${scene_name}

    for view_name in "${views[@]}"; do
        # Convert images to video
        python gaussian_splatting/img2video.py \
            --image_folder ${output_dir}/${scene_name}/${view_name} \
            --video_path ${output_dir}/${scene_name}/${view_name}.mp4
    done

done