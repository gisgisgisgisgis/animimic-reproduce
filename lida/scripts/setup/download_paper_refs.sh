#!/usr/bin/env bash
# Download the 8 reference videos and the paper's own input-view results from the AniMimic project page
# (https://xpandora.github.io/AnimaMimic/) into data/paper_refs/ as ref_N.mp4 and ours_N.mp4.
# The videos belong to the paper's authors; they are not part of this repository.
#   bash scripts/setup/download_paper_refs.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
URL=https://xpandora.github.io/AnimaMimic/static/videos/comparison
mkdir -p data/paper_refs
for n in 1 2 3 4 5 6 7 8; do
  for kind in ref ours; do
    out=data/paper_refs/${kind}_$n.mp4
    [ -s "$out" ] || curl -sSfL -o "$out" "$URL/${n}_${kind}.mp4"
  done
done
ls -la data/paper_refs
