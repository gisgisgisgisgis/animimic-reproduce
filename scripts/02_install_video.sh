#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/02_install_video.sh: install CUDA 12.4 PyTorch 2.6 and video dependencies."
require_env
"$ANIMIMIC_PYTHON" -m pip install --upgrade pip setuptools wheel
"$ANIMIMIC_PYTHON" -m pip install 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' --index-url https://download.pytorch.org/whl/cu124
"$ANIMIMIC_PYTHON" -m pip install -e '.[video]'
"$ANIMIMIC_PYTHON" -m pip check
require_cuda
printf 'Next: bash scripts/03_install_pipeline.sh\n'
