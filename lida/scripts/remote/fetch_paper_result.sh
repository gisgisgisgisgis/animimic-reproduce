#!/bin/zsh
# Fetch one finished sequence from the GPU server into the usual Mac locations:
# Stage 1 → outputs/<name>_sharp_rigid, comparison + playback → same folder, bake → data/<name>, hybrid etc. → outputs/<name>_mv
#   zsh scripts/remote/fetch_paper_result.sh paper_cat
set -e
# Set ANIMAMIMIC_HOST (user@server, as in your ssh config) and ANIMAMIMIC_REMOTE_ROOT (the workspace folder there).
a=$1; R=${ANIMAMIMIC_REMOTE_ROOT:?set ANIMAMIMIC_REMOTE_ROOT}; H=${ANIMAMIMIC_HOST:?set ANIMAMIMIC_HOST}
E="ssh -o BatchMode=yes"
out=outputs/${a}_sharp_rigid; mkdir -p $out outputs/${a}_mv
rsync -a -e "$E" $H:$R/stage1/$a/ $out/
rsync -a -e "$E" $H:$R/stage1/$a.log $out/stage1.log
rsync -a -e "$E" --exclude mv $H:$R/finish/$a/ $out/
rsync -a -e "$E" $H:$R/finish/$a/mv/ outputs/${a}_mv/
rsync -a -e "$E" $H:"$R/seqs/$a/texture_baked*" $H:$R/seqs/$a/textured_rest.glb data/$a/
echo "fetched $a"
