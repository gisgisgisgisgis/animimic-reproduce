# Paper comparison for sequences finished earlier on the Mac (dog, wolf): ARAP version first, then Stage 1, Stage 2.
#   bash run_compare_old.sh paper_wolf 5 paper_dog_h2_fov 4
export ROOT=${ANIMAMIMIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}  # workspace folder on the server; these scripts sit in it
export TMPDIR=$ROOT/tmp_cmp XDG_CACHE_HOME=$ROOT/tmp_cmp/xdg MPLCONFIGDIR=$ROOT/tmp_cmp/mpl TORCH_HOME=$ROOT/hf/torch
export CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=$ROOT/cuda_cache PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
mkdir -p $TMPDIR
source $ROOT/venv/bin/activate
cd $ROOT/code
while [ $# -ge 2 ]; do
  a=$1; n=$2; shift 2; S=$ROOT/seqs/$a; O=$ROOT/finish/$a
  python tests/compare_with_paper.py --seq $S --paper refs/ours_$n.mp4 --result $O/stage1_arap.npz $O/stage1.npz $O/stage2.npz \
    --texture baked --size 512 --sigma 1e-6 --device cuda --out $O/compare_with_paper_baked_512 > $O/compare.log 2>&1 || echo "COMPARE_FAIL $a"
  V=(--variant "只用第0幀=$S"); [ -d $O/mv/hybrid ] && V+=(--variant "混合=$O/mv/hybrid")
  python tests/texture_anim_video.py --seq $S --result $O/stage1_arap.npz "${V[@]}" --out $O/anim_textures --device cuda > $O/anim.log 2>&1 || echo "ANIM_FAIL $a"
  echo "DONE $a $(date +%T)"
done
rm -rf $TMPDIR
