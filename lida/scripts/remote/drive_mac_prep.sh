#!/bin/zsh
# Local driver: camera fits run on the GPU server. Per animal: fetch the fit, run UniRig + SAM2 + AllTracker + VGGT on the
# Mac (scripts/run_paper_video.sh keeps the fit), upload the sequence and mark it READY for the server's Stage 1 queue.
# Set ANIMAMIMIC_HOST (user@server) and ANIMAMIMIC_REMOTE_ROOT (the workspace folder there).
cd "$(dirname "$0")/../.."
R=${ANIMAMIMIC_REMOTE_ROOT:?set ANIMAMIMIC_REMOTE_ROOT}
H=${ANIMAMIMIC_HOST:?set ANIMAMIMIC_HOST}
K=(-o BatchMode=yes -o ConnectTimeout=20)
retry() { for i in {1..20}; do "$@" && return 0; sleep 20; done; return 1; }
skip() { retry ssh $K $H "touch $R/seqs/$1/SKIP"; }
for item in cat:2 dragon:3 eagle:6 dino:1 seahorse:7 girl:8; do
  a=${item%%:*}; n=${item##*:}; s=paper_$a; seq=data/$s
  echo "=== $a $(date +%T)"
  F=fits.log; [[ $a == (dino|seahorse|girl) ]] && F=fits2.log
  until ssh $K $H "test -f $R/seqs/$s/fit_frame0.json || grep -q -E 'FIT $s exit [1-9]' $R/$F 2>/dev/null"; do sleep 20; done
  if ! retry scp $K -q $H:$R/seqs/$s/{input.glb,camera.json,fit_frame0.json,fit_frame0.png} $seq/; then echo "FIT_FAIL $a"; skip $s; continue; fi
  echo "fit $a: $(grep -o '"iou": [0-9.]*' $seq/fit_frame0.json) $(grep -o '"fov_deg": [0-9.]*' $seq/fit_frame0.json)"
  mkdir -p outputs/${s}_mv; retry scp $K -q "$H:$R/${a}_out/*" outputs/${s}_mv/ || echo "no paint for $a"
  if ! caffeinate -is zsh scripts/run_paper_video.sh $seq data/paper_refs/ref_$n.mp4; then echo "PREP_FAIL $a"; skip $s; continue; fi
  retry rsync -a -e "ssh ${K[*]}" $seq/{camera.json,rigged.glb,tracks.npz,depth.npz,frames,masks,reference_mask.png,input.glb} $H:$R/seqs/$s/ || { echo "UPLOAD_FAIL $a"; skip $s; continue; }
  retry ssh $K $H "touch $R/seqs/$s/READY"
  echo "uploaded $a $(date +%T)"
done
echo "DRIVER_DONE $(date +%T)"
