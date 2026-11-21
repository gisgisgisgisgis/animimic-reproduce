#!/usr/bin/env bash
# Clone the upstream tools into third_party/ at the revisions this project was run with.
#   bash scripts/setup/clone_third_party.sh            # all of them
#   bash scripts/setup/clone_third_party.sh sam2 vggt  # only some
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p third_party
# name url revision
REPOS="sam2 https://github.com/facebookresearch/sam2.git 2b90b9f5ceec907a1c18123530e92e794ad901a4
vggt https://github.com/facebookresearch/vggt.git a288dd0f14786c93483e45524328726ab7b1b4ce
alltracker https://github.com/aharley/alltracker.git e7553135e7b361590dbccd10e2b274b024f41cd6
UniRig https://github.com/VAST-AI-Research/UniRig.git 6793c6640ff01c8fb389f3993434124bb43d2933
Hunyuan3D-2 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git f8db63096c8282cb27354314d896feba5ba6ff8a
ltx-2-mlx https://github.com/dgrauet/ltx-2-mlx.git 1192051fd380e0501adb5f3c0e9a216e679cb123"
echo "$REPOS" | while read -r n url rev; do
  if [ $# -gt 0 ] && [[ " $* " != *" $n "* ]]; then continue; fi
  d=third_party/$n
  [ -d "$d/.git" ] || git clone -q "$url" "$d"
  git -C "$d" fetch -q origin "$rev" 2>/dev/null || true
  git -C "$d" checkout -q "$rev"
  echo "$n $(git -C "$d" rev-parse --short HEAD)"
done
