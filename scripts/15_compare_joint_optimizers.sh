#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/15_compare_joint_optimizers.sh RUN_DIRECTORY [--iterations N] [--batch-frames N] [--no-previews]. Reuses the exact saved rig/video/supervision; no generation or FEM."
[[ "$#" -ge 1 ]] || die "Provide the completed run directory."
run_directory="$(caller_path "$1")"
shift
require_env
"$ANIMIMIC_PYTHON" -m animimic.compare_optimizers "$run_directory" "$@"
