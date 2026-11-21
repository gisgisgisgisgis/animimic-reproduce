"""Compare texture bakes on the undeformed mesh, all the way around. Run in `animamimic`:

  python tests/texture_compare.py --seq data/paper_wolf --out outputs/paper_wolf_mv/compare \
      --variant "現在的貼圖=data/paper_wolf" --variant "多視角生成=outputs/paper_wolf_mv/mv" ...

Each variant is a folder holding texture_baked.npz/png for the same rigged mesh. One row per variant, one
column per view: the input camera turned about the vertical axis by 0, 45, 90, 135, 180 and -45 degrees
(180 is the far side, which frame 0 never sees). Writes <out>.png and <out>_heads.png (head close-ups at
90, 135 and 180 degrees).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_qa_render import side_camera, to_png  # noqa: E402

from animamimic.baked import load_baked_mesh  # noqa: E402
from animamimic.camera import Camera  # noqa: E402
from animamimic.render import RasterRenderer  # noqa: E402

FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
ANGLES = [0, 45, 90, 135, 180, -45]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--variant", action="append", required=True, help="label=folder with texture_baked.npz/png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--tile", type=int, default=300)
    args = ap.parse_args()
    seq = Path(args.seq)
    cam = Camera.from_json(seq / "camera.json")
    cam.image_size = args.size
    font = ImageFont.truetype(FONT, 22)
    rows, heads = [], []
    head_box = None
    for v in args.variant:
        label, d = v.split("=", 1)
        tmp = Path(tempfile.mkdtemp())
        for f in ("texture_baked.npz", "texture_baked.png"):
            (tmp / f).symlink_to(Path(d).resolve() / f)
        (tmp / "rigged.glb").symlink_to((seq / "rigged.glb").resolve())
        mesh, _ = load_baked_mesh(tmp)
        if head_box is None:  # the head: front 22% along the long axis, on the side the snout points to
            x = mesh.verts[:, 0]
            span = float(x.max() - x.min())
            snout_low = float(mesh.verts[mesh.verts[:, 1] > np.quantile(mesh.verts[:, 1].numpy(), 0.7), 0].mean()) < float(x.mean())
            head_box = (x < x.min() + 0.22 * span) if snout_low else (x > x.max() - 0.22 * span)
        cells, hcells = [], []
        for a in ANGLES:
            c = side_camera(cam, a)
            rgb, _ = RasterRenderer(mesh, c, torch.device("cpu"), sigma=1e-6)(mesh.verts[None])
            im = to_png(rgb[0])
            if a in (90, 135, 180):
                uv, _ = c.project(mesh.verts[head_box])
                uv = uv.numpy()
                cx, cy = np.median(uv, 0)
                h = 0.6 * max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])) + 8
                pad = int(h) + 1  # white margin so a crop past the image edge is not filled black
                padded = ImageOps.expand(im, border=pad, fill="white")
                hc = padded.crop((int(cx - h) + pad, int(cy - h) + pad, int(cx + h) + pad, int(cy + h) + pad))
                hc = hc.resize((args.tile, args.tile), Image.LANCZOS)
                ImageDraw.Draw(hc).text((6, 4), f"{label} {a}°", fill=(200, 0, 0), font=font)
                hcells.append(np.asarray(hc))
            im = im.resize((args.tile, args.tile), Image.LANCZOS)
            ImageDraw.Draw(im).text((6, 4), f"{label} {a}°", fill=(200, 0, 0), font=font)
            cells.append(np.asarray(im))
        rows.append(np.concatenate(cells, 1))
        heads.append(np.concatenate(hcells, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(f"{args.out}.png")
    Image.fromarray(np.concatenate(heads, 0)).save(f"{args.out}_heads.png")
    print(f"wrote {args.out}.png and {args.out}_heads.png")


if __name__ == "__main__":
    main()
