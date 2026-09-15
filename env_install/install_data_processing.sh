#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "phystwin-data" ]]; then
  echo "Activate phystwin-data before running this script." >&2
  exit 1
fi

# Do not inherit Boba's CUDA 13.2 paths when installing or importing data tools.
unset CUDA_HOME
unset LD_LIBRARY_PATH
export CUDA_HOME="/usr/local/cuda-12.1"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64"

python -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/sam2.git' \
  'git+https://github.com/IDEA-Research/GroundingDINO.git'

python -m pip install \
  'transformers==4.46.3' \
  'tokenizers==0.20.3' \
  'huggingface-hub==0.26.2' \
  addict \
  pycocotools \
  timm \
  yapf

python - <<'PY'
import groundingdino
import sam2
import supervision

print("Data-processing environment imports passed.")
PY