# Stage 1 for uploaded paper sequences on a CUDA server, one GPU, everything inside the workspace.
#   bash run_stage1.sh paper_dino paper_cat ...
# Expects code/ (animamimic/ + scripts/run_animamimic.py) and seqs/<name>/ next to this script.
export ROOT=${ANIMAMIMIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}  # workspace folder on the server; these scripts sit in it
export TMPDIR=$ROOT/tmp XDG_CACHE_HOME=$ROOT/tmp/xdg MPLCONFIGDIR=$ROOT/tmp/mpl
export CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=$ROOT/cuda_cache PYTHONUNBUFFERED=1
mkdir -p $TMPDIR $ROOT/stage1
source $ROOT/venv/bin/activate
cd $ROOT/code
for a in "$@"; do
  echo "=== $a $(date +%T)"
  python scripts/run_animamimic.py --seq $ROOT/seqs/$a --out $ROOT/stage1/$a --device cuda --backend raster \
    --image_size 384 --sigma 1e-5 --w_rigid 10 --iters 3000 --batch_frames 4 --depth_norm per_frame \
    --trans_max 0.01 --w_reg 1000 --skin_smooth 30 > $ROOT/stage1/$a.log 2>&1 || echo "STAGE1_FAIL $a"
  echo "--- $a $(date +%T)"
done
rm -rf $TMPDIR
echo "ALL_DONE $(date +%T)"
