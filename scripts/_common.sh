#!/usr/bin/env bash
# Shared by the numbered entrypoints; all installations are user-local.
set -Eeuo pipefail
ANIMIMIC_CALLER_DIR="$PWD"
ANIMIMIC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ANIMIMIC_ENV_DIR="${ANIMIMIC_ENV_DIR:-$ANIMIMIC_ROOT/.venv}"
if [[ "$ANIMIMIC_ENV_DIR" != /* ]]; then
    ANIMIMIC_ENV_DIR="$ANIMIMIC_CALLER_DIR/$ANIMIMIC_ENV_DIR"
fi
ANIMIMIC_PYTHON="$ANIMIMIC_ENV_DIR/bin/python"
export HF_HOME="${HF_HOME:-$ANIMIMIC_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ANIMIMIC_ROOT/.cache/torch}"
export PATH="$ANIMIMIC_ENV_DIR/bin:$ANIMIMIC_ROOT/third_party/fTetWild/build:$PATH"
cd -- "$ANIMIMIC_ROOT"
trap 'printf "\nFailed: %s, line %s (exit %s)\n" "$0" "$LINENO" "$?" >&2' ERR

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
show_help() {
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        printf '%s\n' "$2"
        exit 0
    fi
}
find_conda() {
    if command -v conda >/dev/null 2>&1; then command -v conda; return; fi
    local candidate
    for candidate in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda"; do
        if [[ -x "$candidate" ]]; then printf '%s\n' "$candidate"; return; fi
    done
    return 1
}
require_env() {
    [[ -x "$ANIMIMIC_PYTHON" ]] || die "Run bash scripts/01_create_env.sh first."
    "$ANIMIMIC_PYTHON" -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Use Python 3.11 or 3.12; create a new environment"'
}
require_cuda() {
    require_env
    "$ANIMIMIC_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check GPU allocation, driver and CUDA PyTorch"; print("GPU:", torch.cuda.get_device_name(0))'
}
caller_path() {
    if [[ "$1" == /* ]]; then printf '%s\n' "$1"; else printf '%s/%s\n' "$ANIMIMIC_CALLER_DIR" "$1"; fi
}
run_cli() { "$ANIMIMIC_PYTHON" -m animimic.cli "$@"; }

use_cuda_toolkit() {
    # Prefer an explicitly configured CUDA_HOME, then a toolkit already on PATH.
    if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
    elif command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")"
    else
        local candidate
        for candidate in "$ANIMIMIC_ROOT/.cache/cuda-12.4" "$ANIMIMIC_ENV_DIR" /usr/local/cuda-12.4; do
            if [[ -x "$candidate/bin/nvcc" ]]; then
                export CUDA_HOME="$candidate"
                export PATH="$CUDA_HOME/bin:$PATH"
                break
            fi
        done
    fi
    command -v nvcc >/dev/null || die "nvcc missing. Run bash scripts/12_install_cuda_toolkit.sh, then rerun this script."
    nvcc --version | grep -q 'release 12.4' || die "CUDA toolkit must be 12.4. Set CUDA_HOME to the correct toolkit, or run 12_install_cuda_toolkit.sh."
}
