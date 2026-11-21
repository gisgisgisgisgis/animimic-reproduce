#!/usr/bin/env bash
# Skeleton + skinning with UniRig on Apple Silicon, producing <seq>/rigged.glb.
# It writes Mac copies of UniRig's configs (no flash-attention, no bf16) and leaves the originals alone.
# Usage: ACCEL=mps bash scripts/prep_00_unirig_mac.sh third_party/UniRig third_party/venvs/unirig input.glb data/<name>
set -euo pipefail

# CPU by default: torch_cluster.fps fails on MPS with "src must be CPU tensor".
ACCEL=${ACCEL:-cpu}
UNIRIG=$(cd "$1" && pwd)
PY=$(cd "$2" && pwd)/bin/python
ORIGINAL=$(cd "$(dirname "$3")" && pwd)/$(basename "$3")
mkdir -p "$4"
SEQ=$(cd "$4" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTORCH_ENABLE_MPS_FALLBACK=1
# spconv and flash_attn have no macOS builds; UniRig gets small plain-torch stand-ins.
export PYTHONPATH="$HERE/mac_shims:${PYTHONPATH:-}"

# UniRig's merge can put the mesh and skeleton in different frames if the asset has node transforms.
INPUT="$SEQ/input_baked.glb"
"$PY" "$HERE/prep_00a_bake_mesh.py" "$ORIGINAL" "$INPUT"

cd "$UNIRIG"
sed 's/_attn_implementation: flash_attention_2/_attn_implementation: eager/' \
  configs/model/unirig_ar_350m_1024_81920_float32.yaml > configs/model/mac_unirig_ar.yaml
sed -e 's/model: unirig_ar_350m_1024_81920_float32/model: mac_unirig_ar/' \
    -e 's/precision: bf16-mixed/precision: 32-true/' -e "s/accelerator: gpu/accelerator: $ACCEL/" \
  configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml > configs/task/mac_skeleton.yaml
sed 's/^  upcast_softmax: False/  upcast_softmax: False\n  enable_flash: False/' configs/model/unirig_skin.yaml > configs/model/mac_unirig_skin.yaml
grep -q 'enable_flash: False' configs/model/mac_unirig_skin.yaml
sed -e 's/model: unirig_skin/model: mac_unirig_skin/' \
    -e 's/precision: bf16-mixed/precision: 32-true/' -e "s/accelerator: gpu/accelerator: $ACCEL/" \
  configs/task/quick_inference_unirig_skin.yaml > configs/task/mac_skin.yaml

SUFFIX="obj,fbx,FBX,dae,glb,gltf,vrm"
extract() {  # input file, then face target
  "$PY" -m src.data.extract --config=configs/data/quick_inference.yaml --require_suffix=$SUFFIX \
    --force_override=true --num_runs=1 --id=0 --time="$(date +%Y_%m_%d_%H_%M_%S)" \
    --faces_target_count=50000 --input="$1" --output_dir=tmp
}

extract "$INPUT"
"$PY" run.py --task=configs/task/mac_skeleton.yaml --seed=12345 --input="$INPUT" --output="$SEQ/skeleton.fbx" --npz_dir=tmp
# UniRig catches prediction errors and still exits 0, so check that the file exists.
[ -f "$SEQ/skeleton.fbx" ] || { echo "UniRig produced no skeleton; see the error above."; exit 1; }

extract "$SEQ/skeleton.fbx"
"$PY" run.py --task=configs/task/mac_skin.yaml --seed=12345 --input="$SEQ/skeleton.fbx" --output="$SEQ/skin.fbx" --npz_dir=tmp --data_name=raw_data.npz

"$PY" -m src.inference.merge --require_suffix=$SUFFIX --num_runs=1 --id=0 \
  --source="$SEQ/skin.fbx" --target="$INPUT" --output="$SEQ/rigged.glb"
echo "Wrote $SEQ/rigged.glb"
