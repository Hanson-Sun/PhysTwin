#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Create the pinned environment first:
# conda env create -f env_install/phystwin-cu132.yml
# conda activate phystwin-cu132
# The build also installs the RGB-D stack (SAM 2, GroundingDINO, PyTorch3D).
exec bash "${REPO_ROOT}/env_install/build_cuda13_extensions.sh" "$@"
