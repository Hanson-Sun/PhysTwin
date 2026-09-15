#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_dir="${repo_root}/data_process/groundedSAM_checkpoints"
weights_dir="${repo_root}/data_process/models/weights"
mkdir -p "${checkpoint_dir}" "${weights_dir}"

download() {
  local url="$1"
  local destination="$2"
  if [[ -s "${destination}" ]]; then
    echo "Already exists: ${destination}"
    return
  fi
  wget --continue --show-progress "${url}" -O "${destination}"
}

download \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" \
  "${checkpoint_dir}/sam2.1_hiera_large.pt"
download \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
  "${checkpoint_dir}/groundingdino_swint_ogc.pth"
download \
  "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_indoor.pth" \
  "${weights_dir}/superglue_indoor.pth"
download \
  "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superglue_outdoor.pth" \
  "${weights_dir}/superglue_outdoor.pth"
download \
  "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/refs/heads/master/models/weights/superpoint_v1.pth" \
  "${weights_dir}/superpoint_v1.pth"