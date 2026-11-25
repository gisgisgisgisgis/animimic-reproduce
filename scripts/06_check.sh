#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/06_check.sh [CONFIG]. Default: configs/experiments/horse-textured-wan-leftforeleg-grouped-tracks.yaml. Checks the full pipeline."
require_env
config="$ANIMIMIC_ROOT/configs/experiments/horse-textured-wan-leftforeleg-grouped-tracks.yaml"
if [[ "$#" -gt 0 ]]; then config="$(caller_path "$1")"; fi
run_cli preflight --config "$config"
