# Stage 2 (tet FEM, Young's modulus fit) from a finished Stage 1, on one GPU, inside the workspace.
#   bash run_stage2.sh paper_cat 6.2     (fps of the 32 sampled frames: 6.2 for 121 frames at 24 fps, 7.75 at 30 fps)
# Needs seqs/<name>/tet.npz (scripts/prep_07_tetmesh.py on the Mac) and stage1/<name>/stage1.npz.
export ROOT=${ANIMAMIMIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}  # workspace folder on the server; these scripts sit in it
export TMPDIR=$ROOT/tmp_s2 XDG_CACHE_HOME=$ROOT/tmp_s2/xdg MPLCONFIGDIR=$ROOT/tmp_s2/mpl
export CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=$ROOT/cuda_cache PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
mkdir -p $TMPDIR $ROOT/stage2
source $ROOT/venv/bin/activate
cd $ROOT/code
a=$1; fps=$2
python scripts/run_animamimic.py --seq $ROOT/seqs/$a --out $ROOT/stage2/$a --device cuda --backend raster \
  --image_size 384 --sigma 1e-5 --batch_frames 4 --depth_norm per_frame --skin_smooth 30 \
  --stage1_from $ROOT/stage1/$a/stage1.npz --stage2 --stage2_iters 10 10 --fps $fps > $ROOT/stage2/$a.log 2>&1 \
  && echo "STAGE2_DONE $a $(date +%T)" || echo "STAGE2_FAIL $a $(date +%T)"
rm -rf $TMPDIR
