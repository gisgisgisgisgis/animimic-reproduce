"""Texture a frame-0 shape with Hunyuan3D-Paint (CUDA). Run on the GPU server inside the paint venv:

  python run_paint.py --in_dir wolf --out_dir wolf_out

<in_dir> holds shape.obj (Hunyuan3D's own output, in its canonical frame, where frame 0 is the front view),
frame0.png and frame0_mask.png. Writes <out_dir>/painted.npz (vertices, faces, uv, texture), painted.glb,
views.png (the six generated views: front, right, back, left, top, bottom) and delit.png (frame 0 after
the light/shadow remover, which is what the multi-view model is conditioned on).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--in_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--repo", default=str(Path(__file__).resolve().parent / "Hunyuan3D-2"))
ap.add_argument("--subfolder", default="hunyuan3d-paint-v2-0")
args = ap.parse_args()

sys.path.insert(0, args.repo)
from huggingface_hub import snapshot_download  # noqa: E402
from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: E402

src, out = Path(args.in_dir), Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
weights = snapshot_download("tencent/Hunyuan3D-2", allow_patterns=["hunyuan3d-delight-v2-0/*", f"{args.subfolder}/*"])
pipe = Hunyuan3DPaintPipeline.from_pretrained(weights, subfolder=args.subfolder)

saved = {}


class Keep:
    """Pass-through wrapper that keeps a model's output for the report images."""

    def __init__(self, model, key):
        self.model, self.key = model, key

    def __call__(self, *a, **k):
        r = self.model(*a, **k)
        saved[self.key] = r
        return r


pipe.models["delight_model"] = Keep(pipe.models["delight_model"], "delit")
pipe.models["multiview_model"] = Keep(pipe.models["multiview_model"], "views")

mesh = trimesh.load(src / "shape.obj", force="mesh", process=False)
frame = np.asarray(Image.open(src / "frame0.png").convert("RGB"))
mask = np.asarray(Image.open(src / "frame0_mask.png").convert("L"))
image = Image.fromarray(np.dstack([frame, mask]), "RGBA")
painted = pipe(mesh, image=image)

tex = np.asarray(painted.visual.material.baseColorTexture if hasattr(painted.visual.material, "baseColorTexture")
                 and painted.visual.material.baseColorTexture is not None else painted.visual.material.image)
np.savez(out / "painted.npz", vertices=np.asarray(painted.vertices), faces=np.asarray(painted.faces),
         uv=np.asarray(painted.visual.uv), texture=tex[..., :3])
painted.export(out / "painted.glb")
saved["delit"].save(out / "delit.png")
views = [v.resize((384, 384)) for v in saved["views"]]
Image.fromarray(np.concatenate([np.asarray(v.convert("RGB")) for v in views], 1)).save(out / "views.png")
print(f"painted {len(painted.vertices)} vertices, texture {tex.shape}; wrote {out}")
