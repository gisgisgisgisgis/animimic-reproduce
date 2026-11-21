# After Stage 1 on a CUDA server: frame-0 texture bake, hybrid all-around texture (if Hunyuan3D-Paint ran),
# ARAP stretch removal (scripts/arap_refine.py, lam 0.01: no silhouette cost on the cat), comparison with the
# paper's own result (ARAP version first, so the video shows it; both scored), stretch numbers and heat map,
# four-angle playback of the ARAP version. One GPU, everything inside the workspace.
#   bash run_finish.sh paper_cat 2
export ROOT=${ANIMAMIMIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}  # workspace folder on the server; these scripts sit in it
export TMPDIR=$ROOT/tmp_finish XDG_CACHE_HOME=$ROOT/tmp_finish/xdg MPLCONFIGDIR=$ROOT/tmp_finish/mpl TORCH_HOME=$ROOT/hf/torch
export CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=$ROOT/cuda_cache PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8  # shared server: the bake otherwise takes every core
mkdir -p $TMPDIR
source $ROOT/venv/bin/activate
cd $ROOT/code
a=$1; n=$2; S=$ROOT/seqs/$a; O=$ROOT/finish/$a; P=$ROOT/${a#paper_}_out; R1=$ROOT/stage1/$a/stage1.npz
mkdir -p $O
[ -f $S/texture_baked.npz ] || python scripts/bake_texture.py --seq $S > $O/bake.log 2>&1 || echo "BAKE_FAIL $a"
V=(--variant "只用第0幀=$S")
if [ -f $P/painted.npz ]; then
  python scripts/transfer_painted_texture.py --seq $S --painted $P/painted.npz --out $O/mv > $O/transfer.log 2>&1 \
    && V+=(--variant "混合=$O/mv/hybrid") || echo "TRANSFER_FAIL $a"
fi
python scripts/arap_refine.py --seq $S --result $R1 --out $O/stage1_arap.npz --lam 0.01 > $O/arap.log 2>&1 || echo "ARAP_FAIL $a"
python tests/compare_with_paper.py --seq $S --paper refs/ours_$n.mp4 --result $O/stage1_arap.npz $R1 --texture baked \
  --size 512 --sigma 1e-6 --device cuda --out $O/compare_with_paper_baked_512 > $O/compare.log 2>&1 || echo "COMPARE_FAIL $a"
python tests/surface_distortion.py --seq $S --result $R1 $O/stage1_arap.npz --out $O/surface_distortion.json \
  > $O/distortion.log 2>&1 || echo "DISTORTION_FAIL $a"
python tests/strain_heatmap.py --seq $S --result $R1 $O/stage1_arap.npz --label "Stage 1" "ARAP" --out $O/strain \
  --device cuda > $O/strain.log 2>&1 || echo "STRAIN_FAIL $a"
python tests/texture_anim_video.py --seq $S --result $O/stage1_arap.npz "${V[@]}" --out $O/anim_textures --device cuda \
  > $O/anim.log 2>&1 || echo "ANIM_FAIL $a"
rm -rf $TMPDIR
echo "FINISHED $a $(date +%T)"
