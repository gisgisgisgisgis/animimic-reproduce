#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/08_run_pipeline.sh [CONFIG] [RUN FLAGS ...]. Defaults to the Wan horse config. Use --video-dir DIR to process existing MP4 files, --until-stage to preview, and --debug-images for supervision images."
require_env
config="$ANIMIMIC_ROOT/configs/horse.yaml"
if [[ "$#" -gt 0 && "$1" != -* ]]; then config="$(caller_path "$1")"; shift; fi
flags=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --video-dir)
            [[ "$#" -ge 2 && "$2" != -* ]] || die "--video-dir requires a directory"
            flags+=(--video-dir "$(caller_path "$2")")
            shift 2
            ;;
        --video-dir=*)
            flags+=("--video-dir=$(caller_path "${1#*=}")")
            shift
            ;;
        *)
            flags+=("$1")
            shift
            ;;
    esac
done
run_cli run --config "$config" "${flags[@]}"
