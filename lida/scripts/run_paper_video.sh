#!/bin/zsh
# Mac part of the route for one of the paper's project-page videos, once <seq>/{frame0.png,frame0_mask.png,shape.obj}
# exist (Hunyuan3D-2 full, from prep_09a_frame0_to_3d.py, here or on a CUDA server):
# camera + fov fit → UniRig → reference view → frames → SAM2 → AllTracker → VGGT. Stage 1 runs separately.
#   zsh scripts/run_paper_video.sh data/paper_cat data/paper_refs/ref_2.mp4
# Run from the repo root.
set -e
seq=$1; video=$2
name=$(basename $seq)
log=outputs/${name}_prep.log
: > $log
# SAM2's first-frame prompt is the video's own mask at the camera resolution.
.venv/bin/python -c "
from PIL import Image; import numpy as np
m = Image.open('$seq/frame0_mask.png').convert('L').resize((512, 512), Image.BILINEAR)
Image.fromarray(((np.asarray(m) > 127) * 255).astype(np.uint8)).save('$seq/reference_mask.png')"
# The fit is a slow CPU grid search; several can be run in parallel beforehand, and are then kept.
while pgrep -f "prep_09b_fit_frame0.py --seq $seq " >/dev/null; do sleep 10; done
if [ -f $seq/fit_frame0.json ] && [ -f $seq/input.glb ]; then
  echo "keeping the existing camera fit" | tee -a $log
else
  .venv/bin/python scripts/prep_09b_fit_frame0.py --seq $seq --fit_fov 2>&1 | grep -v -i warn | tail -2 | tee -a $log
fi
bash scripts/prep_00_unirig_mac.sh third_party/UniRig third_party/venvs/unirig $seq/input.glb $seq > outputs/${name}_unirig.log 2>&1
.venv/bin/python scripts/prep_01_render_view.py --seq $seq --keep_camera 2>&1 | tail -2 | tee -a $log
.venv/bin/python scripts/prep_02_import_video.py --seq $seq --video $video | tee -a $log
third_party/venvs/prep/bin/python scripts/prep_04_sam2.py --seq $seq > outputs/${name}_sam2.log 2>&1
third_party/venvs/prep/bin/python scripts/prep_05_alltracker.py --seq $seq --repo third_party/alltracker > outputs/${name}_alltracker.log 2>&1
third_party/venvs/prep/bin/python scripts/prep_06_vggt.py --seq $seq > outputs/${name}_vggt.log 2>&1
tail -1 outputs/${name}_alltracker.log outputs/${name}_vggt.log | tee -a $log
echo "prep done $seq" | tee -a $log
