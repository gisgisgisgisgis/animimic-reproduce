"""Render the undeformed rigged mesh (baked texture, sharp edges) from the four frame-check views, so a
reconstruction can be scored by tests/frame_qa_vlm.py before spending hours on Stage 1. Run in `animamimic`.

  python tests/rest_views.py --seq data/paper_dog_h2_fov --out outputs/mesh_choice/paper_dog_h2_fov
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_qa_render import VIEWS, side_camera, to_png  # noqa: E402

from animamimic.baked import load_baked_mesh  # noqa: E402
from animamimic.camera import Camera  # noqa: E402
from animamimic.render import RasterRenderer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--size", type=int, default=512)
args = ap.parse_args()
mesh, _ = load_baked_mesh(args.seq)
cam = Camera.from_json(Path(args.seq) / "camera.json")
cam.image_size = args.size
out = Path(args.out) / "rest"
out.mkdir(parents=True, exist_ok=True)
for view, deg in VIEWS.items():
    rgb, _ = RasterRenderer(mesh, side_camera(cam, deg), torch.device("cpu"), sigma=1e-6)(mesh.verts[None])
    to_png(rgb[0]).save(out / f"{view}.png")
print(f"wrote {out}")
