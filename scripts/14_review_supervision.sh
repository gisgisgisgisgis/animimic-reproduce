#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/14_review_supervision.sh SUPERVISION_NPZ [OUTPUT]. CPU only; writes metrics and mask/track overlays."
[[ "$#" -ge 1 && "$#" -le 2 ]] || die "Usage: bash scripts/14_review_supervision.sh SUPERVISION_NPZ [OUTPUT]"
input="$(caller_path "$1")"
output="$(caller_path "${2:-runs/supervision-review}")"
require_env
"$ANIMIMIC_PYTHON" -m animimic.supervision_review "$input" --output "$output"
