#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/10_tests.sh: run offline numerical tests and pip dependency checks."
require_env
"$ANIMIMIC_PYTHON" -m unittest discover -s tests -v
"$ANIMIMIC_PYTHON" -m pip check
