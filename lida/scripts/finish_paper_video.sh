#!/bin/zsh
# After Stage 1 on one of the paper's videos: frame-0 texture bake (mirror fill), the all-around "hybrid" texture
# when Hunyuan3D-Paint output exists, ARAP stretch removal, the comparison with the paper's own result (ARAP version
# first, so the video shows it), stretch numbers and heat map, and a four-angle playback. Mac version of
# scripts/remote/run_finish.sh.
#   zsh scripts/finish_paper_video.sh data/paper_cat 2
# Expects outputs/<name>_sharp_rigid/stage1.npz and, optionally, outputs/<name>_mv/painted.npz.
set -e
seq=$1; n=$2
name=$(basename $seq)
out=outputs/${name}_sharp_rigid
mv=outputs/${name}_mv
[ -f $seq/texture_baked.npz ] || .venv/bin/python scripts/bake_texture.py --seq $seq > $out/bake.log 2>&1
variants=(--variant "只用第0幀=$seq")
if [ -f $mv/painted.npz ]; then
  .venv/bin/python scripts/transfer_painted_texture.py --seq $seq --painted $mv/painted.npz --out $mv > $mv/transfer.log 2>&1
  variants+=(--variant "混合=$mv/hybrid")
fi
[ -f $out/stage1_arap.npz ] || .venv/bin/python scripts/arap_refine.py --seq $seq --result $out/stage1.npz \
  --out $out/stage1_arap.npz --lam 0.01 > $out/arap.log 2>&1
.venv/bin/python tests/compare_with_paper.py --seq $seq --paper data/paper_refs/ours_$n.mp4 \
  --result $out/stage1_arap.npz $out/stage1.npz --texture baked --size 512 --sigma 1e-6 \
  --out $out/compare_with_paper_baked_512 > $out/compare.log 2>&1
.venv/bin/python tests/surface_distortion.py --seq $seq --result $out/stage1.npz $out/stage1_arap.npz \
  --out $out/surface_distortion.json > $out/distortion.log 2>&1
.venv/bin/python tests/strain_heatmap.py --seq $seq --result $out/stage1.npz $out/stage1_arap.npz \
  --label "Stage 1" "ARAP" --out $out/strain > $out/strain.log 2>&1
.venv/bin/python tests/texture_anim_video.py --seq $seq --result $out/stage1_arap.npz $variants \
  --out $out/anim_textures > $out/anim.log 2>&1
echo "finished $seq"
