#!/bin/zsh
# Whole route for one artist-rigged asset, after its video exists:
# import → SAM2 → AllTracker → VGGT → Stage 1 (its own rig) → animated GLB → four-view renders → side-by-side video.
#   zsh scripts/run_hf_asset.sh data/hf_bear data/hf_bear/candidates/sniff.mp4 graze_name
# Run from the repo root. The clip is 97 frames at 24 fps, so the 32 fitted frames span 4.0 s.
set -e
seq=$1; video=$2; name=${3:-motion}
out=outputs/$(basename $seq)
mkdir -p $out
cp $video $seq/video_ltx_$name.mp4
.venv/bin/python scripts/prep_02_import_video.py --seq $seq --video $video
third_party/venvs/prep/bin/python scripts/prep_04_sam2.py --seq $seq > $seq/prep_sam2.log 2>&1
third_party/venvs/prep/bin/python scripts/prep_05_alltracker.py --seq $seq --repo third_party/alltracker > $seq/prep_alltracker.log 2>&1
third_party/venvs/prep/bin/python scripts/prep_06_vggt.py --seq $seq > $seq/prep_vggt.log 2>&1
PYTHONUNBUFFERED=1 caffeinate -is .venv/bin/python scripts/run_animamimic.py --seq $seq --out $out --device mps --backend raster \
  --image_size 384 --sigma 1e-5 --w_rigid 10 --iters 3000 --batch_frames 4 --depth_norm per_frame --trans_max 0.01 --w_reg 1000 \
  --skin_smooth 0 > $out/stage1.log 2>&1
.venv/bin/python scripts/export_animated_glb.py --seq $seq --stage1 $out/stage1.npz --duration 4.0 --name $name \
  --out $out/$(basename $seq | sed 's/^hf_//')_animated.glb
.venv/bin/python tests/frame_qa_render.py --seq $seq --result $out/stage1.npz --out $out/frame_qa --texture rig --sigma 1e-6 --device cpu \
  > $out/frame_qa.log 2>&1
.venv/bin/python tests/rig_animation_video.py --qa $out/frame_qa --out $out/$(basename $seq | sed 's/^hf_//')_vs_video
echo "done $seq"
