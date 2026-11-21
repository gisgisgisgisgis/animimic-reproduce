# Run run_finish.sh for each sequence once its Stage 1 has ended (stage1_queue.log), in order.
#   nohup bash run_finish_queue.sh paper_dino:1 paper_cat:2 ... > finish_queue.log 2>&1 &
ROOT=${ANIMAMIMIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}  # workspace folder on the server; these scripts sit in it
for item in "$@"; do
  a=${item%%:*}; n=${item##*:}
  until grep -q -E -- "--- $a |skip $a" $ROOT/stage1_queue.log 2>/dev/null; do sleep 30; done
  [ -f $ROOT/stage1/$a/stage1.npz ] || { echo "no stage1 for $a"; continue; }
  bash $ROOT/run_finish.sh $a $n
done
echo "FINISH_QUEUE_DONE $(date +%T)"
