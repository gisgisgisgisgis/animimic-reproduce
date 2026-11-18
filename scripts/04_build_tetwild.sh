#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/04_build_tetwild.sh: compile fTetWild. Requires CMake, C++ compiler and GMP."
require_env
[[ -f third_party/fTetWild/CMakeLists.txt ]] || die "Run 03_install_pipeline.sh first."
if [[ -d "$ANIMIMIC_ENV_DIR/conda-meta" ]]; then
    conda_bin="$(find_conda)" || die "Conda executable not found."
    "$conda_bin" install --prefix "$ANIMIMIC_ENV_DIR" -c conda-forge cmake ninja gmp -y
fi
command -v cmake >/dev/null || die "Install CMake; on Ubuntu: sudo apt install cmake libgmp-dev build-essential."
cmake -S third_party/fTetWild -B third_party/fTetWild/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ANIMIMIC_ENV_DIR"
cmake --build third_party/fTetWild/build --parallel "${MAX_JOBS:-2}"
FloatTetwild_bin --help
printf 'Next: bash scripts/05_fetch_assets.sh horse\n'
