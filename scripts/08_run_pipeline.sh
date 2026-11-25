#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/08_run_pipeline.sh [CONFIG] [RUN FLAGS ...]. Defaults to the Wan horse config. Use --until-stage render|video|supervision|joints to preview a stage. Use --debug-images for supervision images."
require_env
config="$ANIMIMIC_ROOT/configs/experiments/horse-textured-wan-leftforeleg-grouped-tracks.yaml"
if [[ "$#" -gt 0 && "$1" != -* ]]; then config="$(caller_path "$1")"; shift; fi
run_cli run --config "$config" "$@"
