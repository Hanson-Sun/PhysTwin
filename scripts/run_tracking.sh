#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/digital_clone_v2
RW=$ROOT/scripts/real_world
BASE_PATH=$ROOT/data/different_types
OUT_DIR=$ROOT/experiments

SCENE0=double_stretch_zebra
SCENE1=rope_double_hand
SCENE2=double_lift_cloth_1


run_case () {
  scene="$1"
  src_npz="$2"
  tag="$3"
  register_to="$4"

  case_dir="$OUT_DIR/$tag"
  obj_pkl="$case_dir/inference.pkl"
  out_mp4="$case_dir/tracking_overlay.mp4"

  # REQUIRED
  controller_pkl="$BASE_PATH/$scene/final_data.pkl"
  calibrate_pkl="$BASE_PATH/$scene/calibrate.pkl"
  metadata_json="$BASE_PATH/$scene/metadata.json"

  mkdir -p "$case_dir"

  python scripts/final_npz_to_pkl.py \
    --input "$src_npz" \
    --output "$obj_pkl" \
    --output-format array \
    --reconstruct-endpoints both \
    --dt 0.01


	python scripts/register_scale_shift.py \
		--source-pkl "$obj_pkl" \
		--reference "$controller_pkl" \


python scripts/render_tracking_overlay.py \
    --base_path "$BASE_PATH" \
    --object_pkl "$obj_pkl" \
    --case_name "$scene" \
    --view_index 0 \
    --controller_source_pkl "$controller_pkl" \
    --calibrate_pkl "$calibrate_pkl" \
    --metadata_json "$metadata_json" \
    --output "$out_mp4"
}



# run_case "$SCENE0" "$RW/0.npz" "${SCENE0}_refnpz" ""
# run_case "$SCENE1" "$RW/1.npz" "${SCENE1}_refnpz" ""
# run_case "$SCENE2" "$RW/2.npz" "${SCENE2}_refnpz" ""


# run_case "$SCENE0" "$RW/0_0_114.npz" "${SCENE0}_ours114_reg" "$RW/0.npz"
run_case "$SCENE1" "$RW/1_0_114.npz" "${SCENE1}_ours114_reg" "$RW/1.npz"
# run_case "$SCENE2" "$RW/2_0_114.npz" "${SCENE2}_ours114_reg" "$RW/2.npz"


echo "Done. Tracking videos:"
find "$OUT_DIR" -type f -name "tracking_overlay.mp4" | sort