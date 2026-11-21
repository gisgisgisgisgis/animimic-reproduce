#!/usr/bin/env bash
# Skeleton + skinning with UniRig, producing <seq>/rigged.glb.
# Run inside the UniRig repo's own conda env. Needs CUDA (the repo states 8 GB or more).
# Usage: bash scripts/prep_00_unirig.sh /path/to/UniRig input.glb data/<name>
set -euo pipefail

UNIRIG=$(cd "$1" && pwd)
ORIGINAL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
mkdir -p "$3"
SEQ=$(cd "$3" && pwd)

# UniRig's merge can put the mesh and skeleton in different frames if the asset has node transforms.
INPUT="$SEQ/input_baked.glb"
python "$(cd "$(dirname "$0")" && pwd)/prep_00a_bake_mesh.py" "$ORIGINAL" "$INPUT"

cd "$UNIRIG"
bash launch/inference/generate_skeleton.sh --input "$INPUT" --output "$SEQ/skeleton.fbx"
bash launch/inference/generate_skin.sh --input "$SEQ/skeleton.fbx" --output "$SEQ/skin.fbx"
bash launch/inference/merge.sh --source "$SEQ/skin.fbx" --target "$INPUT" --output "$SEQ/rigged.glb"
echo "Wrote $SEQ/rigged.glb"
