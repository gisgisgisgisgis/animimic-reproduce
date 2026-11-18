#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/08_run_pipeline.sh [CONFIG] [RUN FLAGS ...]. Default: horse-ltx.yaml. Supports --from-stage / --force-stage / --video-backend."
require_cuda
config="$ANIMIMIC_ROOT/configs/horse-ltx.yaml"
if [[ "$#" -gt 0 && "$1" != -* ]]; then config="$(caller_path "$1")"; shift; fi
run_cli run --config "$config" "$@"
