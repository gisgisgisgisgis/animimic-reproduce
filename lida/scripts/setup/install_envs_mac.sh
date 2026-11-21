#!/usr/bin/env bash
# Mac (Apple Silicon) environments for SAM2 + VGGT (venvs/prep) and UniRig (venvs/unirig), inside third_party/.
# Needs uv and the clones from clone_third_party.sh. Each step logs to third_party/logs/.
cd "$(dirname "$0")/../../third_party" && mkdir -p logs venvs
( uv venv --python 3.11 venvs/prep --allow-existing && VIRTUAL_ENV=$PWD/venvs/prep uv pip install torch torchvision imageio imageio-ffmpeg pillow opencv-python einops scipy matplotlib huggingface_hub safetensors -e ./sam2 -e ./vggt && echo PREP_OK ) > logs/prep.log 2>&1 &
( uv venv --python 3.11 venvs/unirig --allow-existing && VIRTUAL_ENV=$PWD/venvs/unirig uv pip install torch==2.5.1 torchvision==0.20.1 "numpy==1.26.4" transformers==4.51.3 python-box einops omegaconf pytorch_lightning lightning addict timm fast-simplification "bpy==4.2.0" trimesh open3d pyrender huggingface_hub scipy && echo UNIRIG_BASE_OK ) > logs/unirig.log 2>&1 &
wait
