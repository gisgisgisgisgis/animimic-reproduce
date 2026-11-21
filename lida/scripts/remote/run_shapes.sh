# Upload frames/<name>.png (frame 0 of each video) and prep_09a_frame0_to_3d.py next to this script, then:
#   nohup bash run_shapes.sh dino cat ... > shapes.log 2>&1 &
# Shape (Hunyuan3D-2 full, as for the dog) then Hunyuan3D-Paint, for each paper video's frame 0. GPU 0 only.
export ROOT=${ANIMAMIMIC_ROOT:-$(cd "$(dirname "$0")" && pwd)}  # workspace folder on the server; these scripts sit in it
export TMPDIR=$ROOT/tmp XDG_CACHE_HOME=$ROOT/tmp/xdg HF_HOME=$ROOT/hf HY3DGEN_MODELS=$ROOT/hf/hy3dgen HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_PATH=$ROOT/cuda_cache MPLCONFIGDIR=$ROOT/tmp/mpl UV_CACHE_DIR=$ROOT/tmp/uv
mkdir -p $TMPDIR
source $ROOT/venv/bin/activate
cd $ROOT
for a in "$@"; do
  echo "=== $a $(date +%T)"
  python prep_09a_frame0_to_3d.py --video frames/$a.png --seq $a --repo Hunyuan3D-2 --model tencent/Hunyuan3D-2 \
    --subfolder hunyuan3d-dit-v2-0 --octree_resolution 512 --faces 60000 --device cuda || { echo "SHAPE_FAIL $a"; continue; }
  python run_paint.py --in_dir $a --out_dir ${a}_out || echo "PAINT_FAIL $a"
done
rm -rf $TMPDIR
echo "ALL_DONE $(date +%T)"
