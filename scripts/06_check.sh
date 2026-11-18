#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/06_check.sh [CONFIG]. Default: configs/horse-ltx.yaml. Checks the full pipeline."
require_env
config="$ANIMIMIC_ROOT/configs/horse-ltx.yaml"
if [[ "$#" -gt 0 ]]; then config="$(caller_path "$1")"; fi
run_cli preflight --config "$config"
