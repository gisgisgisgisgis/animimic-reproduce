#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/11_preview_asset.sh [ASSET]. Default: horse. CPU preview; requires PyTorch3D."
require_env
run_cli assets preview --asset "${1:-horse}" --output runs/asset-previews
