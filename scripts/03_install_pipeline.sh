#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/03_install_pipeline.sh [ltx|wan]: install supervision, PyTorch3D and FEM dependencies. Requires nvcc 12.4."
backend="${1:-ltx}"
[[ "$backend" == ltx || "$backend" == wan ]] || die "Backend must be ltx or wan."
require_cuda
command -v git >/dev/null || die "git is required."
command -v nvcc >/dev/null || die "Load the server CUDA 12.4 toolkit/module first (nvcc missing)."
nvcc --version | grep -q 'release 12.4' || die "Use CUDA toolkit 12.4 with this cu124 installation."
if [[ -z "${CUDA_HOME:-}" ]]; then
    export CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")"
fi
export MAX_JOBS="${MAX_JOBS:-2}"
export PIP_CONSTRAINT="$ANIMIMIC_ROOT/scripts/runtime-constraints.txt"
"$ANIMIMIC_PYTHON" -m pip install -e '.[gpu,video]'
"$ANIMIMIC_PYTHON" scripts/bootstrap.py --video-backend "$backend"
"$ANIMIMIC_PYTHON" -m pip check
printf 'Next: bash scripts/04_build_tetwild.sh\n'
