#!/usr/bin/env bash
# LTX-2.3 image-to-video on Apple GPU (MLX), inside third_party/ (venvs/ltx, models/). Only needed to generate
# videos for your own rigged assets. Needs uv and the ltx-2-mlx clone from clone_third_party.sh.
# LTX_BUILD_CONSTRAINTS may point to a pip build-constraints file if the ltx packages fail to build.
set -e
cd "$(dirname "$0")/../../third_party"
PY=$PWD/venvs/ltx/bin/python
uv venv --python 3.11 venvs/ltx --allow-existing
uv pip install --python $PY 'mlx==0.31.1' 'mlx-lm==0.31.1' 'mlx-metal==0.31.1' 'transformers>=5.0.0,<5.13.0'
uv pip install --python $PY ${LTX_BUILD_CONSTRAINTS:+--build-constraints $LTX_BUILD_CONSTRAINTS} ./ltx-2-mlx/packages/ltx-core-mlx ./ltx-2-mlx/packages/ltx-pipelines-mlx
uv pip install --python $PY --no-deps 'mlx-vlm==0.4.4'
uv pip install --python $PY certifi pillow numpy 'huggingface-hub>=1.5.0,<2.0' 'hf_transfer>=0.1.6'
echo LTX_ENV_OK
export HF_HUB_ENABLE_HF_TRANSFER=1
venvs/ltx/bin/hf download dgrauet/ltx-2.3-mlx-q4 --local-dir models/ltx-2.3-mlx-q4 --include '*.json' --include transformer-distilled.safetensors --include connector.safetensors --include vae_decoder.safetensors --include vae_encoder.safetensors --include audio_vae.safetensors --include vocoder.safetensors --include spatial_upscaler_x2_v1_1.safetensors
venvs/ltx/bin/hf download mlx-community/gemma-3-12b-it-4bit --local-dir models/gemma-3-12b-it-4bit
echo LTX_MODELS_OK
