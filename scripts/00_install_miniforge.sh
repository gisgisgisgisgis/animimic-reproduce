#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"
show_help "${1:-}" "Optional: bash scripts/00_install_miniforge.sh [INSTALL_PREFIX]. Needed only without Python 3.11/Conda."
prefix="${1:-$HOME/miniforge3}"
if [[ "$prefix" != /* ]]; then prefix="$(caller_path "$prefix")"; fi
if [[ -x "$prefix/bin/conda" ]]; then
    printf 'Conda already exists: %s\n' "$prefix/bin/conda"
    exit 0
fi
[[ ! -e "$prefix" ]] || die "Install prefix already exists; choose another empty path."
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die "This installer supports Linux x86_64 only."
command -v curl >/dev/null || die "Install curl or ask the system administrator."
installer="$(mktemp /tmp/animimic-miniforge-XXXXXX.sh)"
trap 'rm -f -- "$installer"' EXIT
curl -fL --retry 3 https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o "$installer"
bash "$installer" -b -p "$prefix"
printf 'Installed. Next: bash scripts/01_create_env.sh\n'
