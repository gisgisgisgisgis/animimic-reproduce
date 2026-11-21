# Camera + fov fit (prep_09b) for several frame-0 meshes at once: CPU grid search in parallel, refinement on GPU 0.
#   bash run_fits.sh paper_dino paper_cat ...   (each seqs/<name>/ holds shape.obj, frame0.png, frame0_mask.png)
export ROOT=${ANIMAMIMIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}  # workspace folder on the server; these scripts sit in it
export TMPDIR=$ROOT/tmp_fit XDG_CACHE_HOME=$ROOT/tmp_fit/xdg MPLCONFIGDIR=$ROOT/tmp_fit/mpl
export CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=$ROOT/cuda_cache PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
mkdir -p $TMPDIR
source $ROOT/venv/bin/activate
cd $ROOT/code
for a in "$@"; do
  ( python scripts/prep_09b_fit_frame0.py --seq $ROOT/seqs/$a --fit_fov --device cuda > $ROOT/seqs/$a.fit.log 2>&1; echo "FIT $a exit $? $(date +%T)" ) &
done
wait
rm -rf $TMPDIR
echo "FITS_DONE $(date +%T)"
