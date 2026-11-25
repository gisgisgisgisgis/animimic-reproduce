#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/09_download_wan.sh: download the official Wan 5B weights."
require_env
"$ANIMIMIC_PYTHON" -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="Wan-AI/Wan2.2-TI2V-5B", local_dir="weights/Wan2.2-TI2V-5B")'
