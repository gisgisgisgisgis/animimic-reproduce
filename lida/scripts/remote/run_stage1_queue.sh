# Wait for the shape/paint job and the camera fits to finish (one GPU at a time, no out-of-memory), then run Stage 1 on each sequence as soon as
# the Mac has uploaded it (seqs/<name>/READY) or skipped it (seqs/<name>/SKIP).
#   nohup bash run_stage1_queue.sh paper_dino paper_cat ... > stage1_queue.log 2>&1 &
ROOT=${ANIMAMIMIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}  # workspace folder on the server; these scripts sit in it
until grep -q ALL_DONE $ROOT/shapes.log && grep -q FITS_DONE $ROOT/fits2.log 2>/dev/null; do sleep 30; done
for a in "$@"; do
  until [ -f $ROOT/seqs/$a/READY ] || [ -f $ROOT/seqs/$a/SKIP ]; do sleep 30; done
  [ -f $ROOT/seqs/$a/SKIP ] && { echo "skip $a"; continue; }
  bash $ROOT/run_stage1.sh $a
done
echo "QUEUE_DONE $(date +%T)"
