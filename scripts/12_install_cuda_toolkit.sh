#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/12_install_cuda_toolkit.sh: install CUDA toolkit 12.4.1 in .cache/cuda-12.4 via Conda; no sudo or driver changes."
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "This installer supports Linux x86_64."
conda_bin="$(find_conda)" || die "Conda is required. Run bash scripts/00_install_miniforge.sh first."
prefix="$ANIMIMIC_ROOT/.cache/cuda-12.4"
if [[ -e "$prefix" && ! -d "$prefix/conda-meta" ]]; then
    die "Toolkit path exists but is not a Conda prefix: $prefix. Existing files were preserved."
fi
if [[ -d "$prefix/conda-meta" ]]; then
    action=install
else
    action=create
fi
"$conda_bin" "$action" --prefix "$prefix" --override-channels \
    -c nvidia/label/cuda-12.4.1 -c conda-forge 'cuda-toolkit=12.4.1' -y
export CUDA_HOME="$prefix"
export PATH="$CUDA_HOME/bin:$PATH"
use_cuda_toolkit
nvcc --version
printf 'Toolkit installed at %s\nNext: bash scripts/03_install_pipeline.sh ltx\n' "$CUDA_HOME"
