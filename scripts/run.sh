cd /root/digital_clone_v2
set -e

ROOT=/root/digital_clone_v2
RW=$ROOT/scripts/real_world

SCENE0=double_stretch_zebra
SCENE1=rope_double_hand
SCENE2=double_lift_cloth_1

VIDEO0=$ROOT/data/different_types/$SCENE0/color/0.mp4
VIDEO1=$ROOT/data/different_types/$SCENE1/color/0.mp4
VIDEO2=$ROOT/data/different_types/$SCENE2/color/0.mp4

run_case () {
  scene="$1"
  src_video="$2"
  src_npz="$3"
  case_name="$4"
  register_to="$5"

  cmd=(
    python scripts/npz_to_inference_render.py
    --source-video "$src_video"
    --source-npz "$src_npz"
    --case-name "$case_name"
    --gaussian-scene "$scene"
    --source-view-index 0
    --num-views 1
    --prediction-dir gaussian_output_dynamic_white
    --inference-dir experiments
    --mask-fallback empty
    --reconstruct-endpoints both
    --dt 0.01
    # --auto-prepare-gaussian
    # --prepare-data-config data_config_test.csv
  )

  if [ -n "$register_to" ]; then
    cmd+=(--register-to "$register_to" --register-npz-position-key position)
  fi

  "${cmd[@]}"
}

# A) Reference renders (no registration)
# run_case "$SCENE0" "$VIDEO0" "$RW/0.npz" "${SCENE0}_refnpz" ""
# run_case "$SCENE1" "$VIDEO1" "$RW/1.npz" "${SCENE1}_refnpz" ""
# run_case "$SCENE2" "$VIDEO2" "$RW/2.npz" "${SCENE2}_refnpz" ""

# B) Ours renders (register to matching reference NPZ first)
run_case "$SCENE0" "$VIDEO0" "$RW/0_0_114.npz" "${SCENE0}_ours114_reg" "$RW/0.npz"
run_case "$SCENE1" "$VIDEO1" "$RW/1_0_114.npz" "${SCENE1}_ours114_reg" "$RW/1.npz"
run_case "$SCENE2" "$VIDEO2" "$RW/2_0_114.npz" "${SCENE2}_ours114_reg" "$RW/2.npz"

echo "Done. Final integrated videos:"
find gaussian_output_dynamic_white -type f -name "0_integrate.mp4" | sort