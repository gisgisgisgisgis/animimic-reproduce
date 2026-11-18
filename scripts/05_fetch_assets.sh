#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/05_fetch_assets.sh [ASSET ...]. Default: horse. Example: horse puppy shark."
require_env
if [[ "$#" -eq 0 ]]; then set -- horse; fi
for asset in "$@"; do run_cli assets fetch --asset "$asset"; done
