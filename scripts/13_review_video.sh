#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/13_review_video.sh [CONFIG] [--render-only] [RUN FLAGS ...]. Default: horse-textured-ltx-5s.yaml (121 frames at 24 fps). Stops before supervision/joints/FEM."
require_env
config="$ANIMIMIC_ROOT/configs/horse-textured-ltx-5s.yaml"
if [[ "$#" -gt 0 && "$1" != -* ]]; then config="$(caller_path "$1")"; shift; fi
until=video
if [[ "${1:-}" == --render-only ]]; then until=render; shift; fi
run_cli run --config "$config" --until-stage "$until" "$@"
