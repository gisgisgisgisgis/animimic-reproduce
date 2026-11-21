#!/bin/bash
# Install Hunyuan3D-2's texture stage (Hunyuan3D-Paint: delight + multi-view diffusion + bake) on a CUDA
# server, entirely inside one workspace folder. Caches and build files are deleted at the end.
#   bash setup_hy3d_paint.sh /project/<user>/animamimic_mv
set -e
ROOT=$1
mkdir -p $ROOT/tmp
export TMPDIR=$ROOT/tmp XDG_CACHE_HOME=$ROOT/tmp/xdg HF_HOME=$ROOT/hf PIP_NO_CACHE_DIR=1
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9}  # a toolkit that matches the cu124 wheels; 8.9 = RTX 4090
export PATH=$CUDA_HOME/bin:$PATH
cd $ROOT
[ -d Hunyuan3D-2 ] || git clone --depth 1 https://github.com/Tencent/Hunyuan3D-2.git
# The system Python has no pip or ensurepip, so use uv (a single binary from the official GitHub release,
# checked against its published SHA-256) to make the venv and install packages.
UV_VER=0.13.9; UV_TGZ=uv-x86_64-unknown-linux-gnu.tar.gz
if [ ! -x $ROOT/bin/uv ]; then
  mkdir -p $ROOT/bin $ROOT/tmp/uv && cd $ROOT/tmp/uv
  curl -sSfLO https://github.com/astral-sh/uv/releases/download/$UV_VER/$UV_TGZ
  curl -sSfLO https://github.com/astral-sh/uv/releases/download/$UV_VER/$UV_TGZ.sha256
  sha256sum -c $UV_TGZ.sha256
  tar xzf $UV_TGZ && cp uv-x86_64-unknown-linux-gnu/uv $ROOT/bin/ && cd $ROOT
fi
export UV_NO_CACHE=1 UV_PYTHON_INSTALL_DIR=$ROOT/tmp/uvpy UV_PYTHON_DOWNLOADS=never
rm -rf venv && $ROOT/bin/uv venv -q --python /usr/bin/python3.12 venv
source venv/bin/activate
pip() { $ROOT/bin/uv pip "$@"; }
pip install -q wheel setuptools
pip install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -q ninja pybind11 "diffusers==0.31.0" "transformers==4.46.3" "accelerate==1.1.1" "huggingface_hub<0.26" \
  einops opencv-python-headless omegaconf tqdm trimesh pymeshlab pygltflib xatlas "numpy<2" scipy
cd Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer && pip install -q --no-build-isolation . && cd -
cd Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer && python setup.py build_ext --inplace -q && cd -
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("tencent/Hunyuan3D-2", allow_patterns=["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0/*"])
EOF
rm -rf $ROOT/tmp Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer/build Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer/build
python -c "import torch, custom_rasterizer; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
du -sh $ROOT/venv $ROOT/hf $ROOT/Hunyuan3D-2
echo SETUP_DONE
