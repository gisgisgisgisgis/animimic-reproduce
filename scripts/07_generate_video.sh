#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/07_generate_video.sh IMAGE [CONFIG] [OUTPUT]. Default config: fox-ltx.yaml; output: runs/reference-ltx.mp4."
[[ "$#" -ge 1 && "$#" -le 3 ]] || die "Usage: bash scripts/07_generate_video.sh IMAGE [CONFIG] [OUTPUT]"
image="$(caller_path "$1")"
config="$ANIMIMIC_ROOT/configs/fox-ltx.yaml"
output="$ANIMIMIC_ROOT/runs/reference-ltx.mp4"
if [[ "$#" -ge 2 ]]; then config="$(caller_path "$2")"; fi
if [[ "$#" -ge 3 ]]; then output="$(caller_path "$3")"; fi
[[ -f "$image" ]] || die "Input image does not exist: $image"
require_env
# The Python CLI checks CUDA only on a cache miss.
run_cli video --config "$config" --image "$image" --output "$output"
