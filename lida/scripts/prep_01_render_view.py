"""Render the canonical reference view from <seq>/rigged.glb. Run in the `animamimic` env."""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animamimic.camera import Camera  # noqa: E402
from animamimic.device import pick_device  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.render import build_renderer, render_in_chunks  # noqa: E402
from animamimic.rig import bone_distance  # noqa: E402

# Median vertex-to-bone distance, on a mesh whose longest side is 1. Correct rigs measured 0.045
# (horse) and 0.047 (Fox). A horse whose skeleton was in the wrong frame measured 0.317.
MAX_BONE_OFFSET = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--dist", type=float, default=2.2)
    ap.add_argument("--elev", type=float, default=10.0)
    ap.add_argument("--azim", type=float, default=30.0)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--keep_camera", action="store_true",
                    help="use the existing camera.json, e.g. one fitted to a given video by prep_09b_fit_frame0.py")
    args = ap.parse_args()

    seq = Path(args.seq)
    device = pick_device(args.device)
    glb = RiggedGLB(seq / "rigged.glb")
    mesh = glb.mesh().to(device)
    offset = float(bone_distance(glb.rig().to(device), mesh.verts).median())
    if offset > MAX_BONE_OFFSET:
        raise SystemExit(
            f"{seq}/rigged.glb: the skeleton does not fit the mesh (median vertex-to-bone distance {offset:.3f}, "
            f"limit {MAX_BONE_OFFSET}). Stage 1 would tear this mesh. Rig the asset again with prep_00_unirig*.sh, "
            "which bakes node transforms first."
        )
    if args.keep_camera:
        camera = Camera.from_json(seq / "camera.json", device)
    else:
        camera = Camera.look_at(args.dist, args.elev, args.azim, args.fov, args.image_size, device=device)
    rgb, mask = render_in_chunks(build_renderer(mesh, camera, device), mesh.verts[None])

    imageio.imwrite(seq / "reference.png", (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    if args.keep_camera and (seq / "reference_mask.png").exists():
        print("Keeping reference_mask.png (the video's own mask); the render is in reference.png.")
    else:
        imageio.imwrite(seq / "reference_mask.png", ((mask[0] > 0.5).cpu().numpy() * 255).astype(np.uint8))
    camera.to_json(seq / "camera.json")
    print(f"Wrote {seq}/reference.png. Give it to the image-to-video model, then run prep_02_import_video.py.")


if __name__ == "__main__":
    main()
