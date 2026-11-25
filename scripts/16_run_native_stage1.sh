#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/16_run_native_stage1.sh RUN_DIRECTORY [--prepare-only] [--from-stage stage1] [--iterations N]. Runs the independent native pipeline; no video generation or FEM."
[[ "$#" -ge 1 ]] || die "Provide the existing run directory."
run_directory="$(caller_path "$1")"
shift
require_env
[[ -f "$ANIMIMIC_ROOT/native-stage1/run_existing_video.py" ]] || die "The separate native-stage1 runner is missing from this checkout."
"$ANIMIMIC_PYTHON" -c 'import pygltflib' || die "Install the native GLB reader in the active environment: python -m pip install pygltflib==1.16.5"
"$ANIMIMIC_PYTHON" "$ANIMIMIC_ROOT/native-stage1/run_existing_video.py" "$run_directory" "$@"
