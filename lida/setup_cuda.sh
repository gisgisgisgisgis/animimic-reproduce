#!/usr/bin/env bash
# Environment setup for a Linux CUDA machine. Not yet run: no CUDA machine was available when this was written.
# The tools need incompatible PyTorch versions, so they get separate conda environments.
set -euo pipefail

REPOS=${1:-$PWD/third_party}
mkdir -p "$REPOS"
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Main environment: rendering, Stage 1, Stage 2.
#    The PyTorch3D prebuilt wheel exists for exactly py3.10 + CUDA 12.1 + torch 2.4.1.
conda create -y -n animamimic python=3.10
conda activate animamimic
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html
pip install -r requirements-cuda.txt
conda deactivate

# 2. Preprocessing: SAM2 requires torch >= 2.5.1, VGGT runs alongside it.
conda create -y -n animamimic-prep python=3.10
conda activate animamimic-prep
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/sam2.git" "git+https://github.com/facebookresearch/vggt.git"
pip install imageio pillow
conda deactivate

# 3. AllTracker and UniRig each follow their own README.
git -C "$REPOS" clone https://github.com/aharley/alltracker.git
git -C "$REPOS" clone https://github.com/VAST-AI-Research/UniRig.git
echo "Next: create the AllTracker env (python 3.12.8) and the UniRig env (python 3.11) from their READMEs."
