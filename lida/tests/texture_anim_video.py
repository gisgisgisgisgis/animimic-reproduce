"""Play one Stage 1 result with several texture bakes, from the input camera and around the far side.

  python tests/texture_anim_video.py --seq data/paper_wolf --result outputs/paper_wolf_sharp_rigid/stage1.npz \
      --variant "現在的貼圖=data/paper_wolf" --variant "混合=outputs/paper_wolf_mv/hybrid" --out outputs/paper_wolf_mv/anim

One row per variant; columns are the input camera turned by 0, 45, 90 and 180 degrees (180 is the side
frame 0 never sees). Writes <out>.mp4 and <out>.png (first, middle and last frame).
"""
import argparse
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_qa_render import side_camera  # noqa: E402

from animamimic.baked import load_baked_mesh  # noqa: E402
from animamimic.camera import Camera  # noqa: E402
from animamimic.render import RasterRenderer, render_in_chunks  # noqa: E402

FONT = next((f for f in ("/System/Library/Fonts/STHeiti Medium.ttc",  # macOS
                          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")  # Linux servers
             if Path(f).exists()), "/System/Library/Fonts/STHeiti Medium.ttc")
ANGLES = [0, 45, 90, 180]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--fps", type=float, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    seq = Path(args.seq)
    device = torch.device(args.device)
    cam = Camera.from_json(seq / "camera.json")  # RasterRenderer moves it to the device
    cam.image_size = args.size
    verts_all = torch.from_numpy(np.load(args.result)["verts"]).float()
    font = ImageFont.truetype(FONT, 20)
    grid = []  # [variant][angle] -> (T, S, S, 3) uint8
    labels = []
    for v in args.variant:
        label, d = v.split("=", 1)
        labels.append(label)
        tmp = Path(tempfile.mkdtemp())
        for f in ("texture_baked.npz", "texture_baked.png"):
            (tmp / f).symlink_to(Path(d).resolve() / f)
        (tmp / "rigged.glb").symlink_to((seq / "rigged.glb").resolve())
        mesh, src = load_baked_mesh(tmp)
        verts = verts_all[:, src].to(device)
        row = []
        for a in ANGLES:
            rgb, _ = render_in_chunks(RasterRenderer(mesh, side_camera(cam, a), device, sigma=1e-6), verts)
            row.append((rgb.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
        grid.append(row)
    T = verts_all.shape[0]
    frames = []
    for t in range(T):
        rows = []
        for label, row in zip(labels, grid):
            cells = []
            for a, stack in zip(ANGLES, row):
                im = Image.fromarray(stack[t])
                ImageDraw.Draw(im).text((6, 4), f"{label} {a}°", fill=(200, 0, 0), font=font)
                cells.append(np.asarray(im))
            rows.append(np.concatenate(cells, 1))
        frames.append(np.concatenate(rows, 0))
    imageio.mimsave(f"{args.out}.mp4", frames * 2, fps=args.fps, macro_block_size=2, quality=8)
    Image.fromarray(np.concatenate([frames[0], frames[T // 2], frames[-1]], 0)).save(f"{args.out}.png")
    print(f"wrote {args.out}.mp4 ({T} frames) and {args.out}.png")


if __name__ == "__main__":
    main()
