#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "bash scripts/01_create_env.sh: reuse an active Conda/venv environment, or create repo-local Python 3.11 .venv. ANIMIMIC_ENV_DIR overrides automatic selection."
if [[ -e "$ANIMIMIC_ENV_DIR" ]]; then
    [[ -x "$ANIMIMIC_PYTHON" ]] || die "Environment path exists but has no Python. Choose a new ANIMIMIC_ENV_DIR; existing files are preserved."
    require_env
    printf 'Reusing environment: %s\n' "$ANIMIMIC_ENV_DIR"
elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$ANIMIMIC_ENV_DIR"
elif conda_bin="$(find_conda)"; then
    "$conda_bin" create --prefix "$ANIMIMIC_ENV_DIR" python=3.11 pip -y
else
    die "Python 3.11/Conda missing. Run bash scripts/00_install_miniforge.sh first."
fi
require_env
"$ANIMIMIC_PYTHON" --version
"$ANIMIMIC_PYTHON" -m pip --version
printf 'Next: bash scripts/02_install_video.sh\n'
