"""Head close-ups of every frame for tests/frame_qa_vlm.py, laid out like frame_qa_render.py
(frames/<view>/NN.png, rest/<view>.png). At 512 px full body the VLM judges the whole animal and does not
see a face problem; these crops let it judge the face alone. Run in `animamimic`:

  python tests/head_qa_render.py --seq data/paper_wolf --result outputs/paper_wolf_sharp_rigid/stage1.npz \
      --out outputs/paper_wolf_sharp_rigid/face_qa
  third_party/venvs/vlm/bin/python tests/frame_qa_vlm.py --dir outputs/paper_wolf_sharp_rigid/face_qa --animal "wolf's head"

The head is the part of the undeformed mesh in the front 22% and top 60% of the input view's silhouette
(--head_side says which way the animal faces in the input view). --texture_dir renders another bake
(texture_baked.npz/png) on the same mesh, to compare bakes.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from frame_qa_render import side_camera  # noqa: E402

from animamimic.baked import load_baked_mesh  # noqa: E402
from animamimic.camera import Camera  # noqa: E402
from animamimic.render import RasterRenderer  # noqa: E402

VIEWS = {"plus45": 45.0, "minus45": -45.0, "side90": 90.0}

ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--result", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--texture_dir", default=None)
ap.add_argument("--head_side", choices=("left", "right"), default="left")
ap.add_argument("--size", type=int, default=384)
args = ap.parse_args()

seq = Path(args.seq)
if args.texture_dir:
    tmp = Path(tempfile.mkdtemp())
    for f in ("texture_baked.npz", "texture_baked.png"):
        (tmp / f).symlink_to(Path(args.texture_dir).resolve() / f)
    (tmp / "rigged.glb").symlink_to((seq / "rigged.glb").resolve())
    mesh, src = load_baked_mesh(tmp)
else:
    mesh, src = load_baked_mesh(seq)
res = torch.from_numpy(np.load(args.result)["verts"]).float()
S = 2 * args.size
cam = Camera.from_json(seq / "camera.json")
cam.image_size = S
p, _ = cam.project(mesh.verts)
x0, x1 = p[:, 0].min(), p[:, 0].max()
y0, y1 = p[:, 1].min(), p[:, 1].max()
front = p[:, 0] < x0 + 0.22 * (x1 - x0) if args.head_side == "left" else p[:, 0] > x1 - 0.22 * (x1 - x0)
head = front & (p[:, 1] < y0 + 0.6 * (y1 - y0))
out = Path(args.out)
(out / "rest").mkdir(parents=True, exist_ok=True)
for view, deg in VIEWS.items():
    c = side_camera(cam, deg)
    r = RasterRenderer(mesh, c, torch.device("cpu"), sigma=1e-6)
    (out / "frames" / view).mkdir(parents=True, exist_ok=True)
    for t, V in [(t, res[t][src]) for t in range(len(res))] + [(None, mesh.verts)]:
        img = (r(V[None])[0][0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        q, _ = c.project(V[head])
        a0, a1, b0, b1 = q[:, 0].min().item(), q[:, 0].max().item(), q[:, 1].min().item(), q[:, 1].max().item()
        cx, cy, h = (a0 + a1) / 2, (b0 + b1) / 2, max(a1 - a0, b1 - b0) * 0.62
        box = (int(cx - h) + S, int(cy - h) + S, int(cx + h) + S, int(cy + h) + S)
        im = Image.fromarray(np.pad(img, ((S, S), (S, S), (0, 0)), constant_values=255)).crop(box)
        im = im.resize((args.size, args.size), Image.LANCZOS)
        im.save(out / "rest" / f"{view}.png" if t is None else out / "frames" / view / f"{t:02d}.png")
print(f"wrote head close-ups of {len(res)} frames x {len(VIEWS)} views to {out}")
