"""Side-by-side Fox video: truth, Stage 1 from true inputs, Stage 1 from the real tools, at three views."""
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.render import RasterRenderer, render_in_chunks  # noqa: E402

SIZE = 256
LABEL_WIDTH = 200
FONT_PATHS = ["/System/Library/Fonts/STHeiti Medium.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]


def font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def banner(text, width, height, size):
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    f = font(size)
    box = d.textbbox((0, 0), text, font=f)
    d.text(((width - box[2] + box[0]) / 2, (height - box[3] + box[1]) / 2 - box[1]), text, fill="black", font=f)
    return np.asarray(im)


def main():
    device = torch.device("cpu")
    truth = torch.from_numpy(np.load(ROOT / "data/fox_real/gt.npz")["verts"]).float()
    mesh = RiggedGLB(ROOT / "data/fox_real/rigged.glb").mesh().to(device)
    diag = float((mesh.verts.max(0).values - mesh.verts.min(0).values).norm())
    runs = [
        ("真值", truth),
        ("全部真值輸入", torch.from_numpy(np.load(ROOT / "outputs/fox_real_allgt/stage1.npz")["verts"]).float()),
        ("全部真實工具", torch.from_numpy(np.load(ROOT / "outputs/fox_real_single_perframe/stage1.npz")["verts"]).float()),
    ]
    rows, report = [], {}
    for name, offset in (("輸入視角", 0.0), ("+45°（沒看過）", 45.0), ("-45°（沒看過）", -45.0)):
        cam = Camera.look_at(2.2, 15.0, 30.0 + offset, 40.0, SIZE)
        r = RasterRenderer(mesh, cam, device)
        renders = [render_in_chunks(r, v) for _, v in runs]
        gt_mask = renders[0][1] > 0.5
        for (label, _), (_, mask) in zip(runs[1:], renders[1:]):
            m = mask > 0.5
            report[f"{label} | {name}"] = float(((m & gt_mask).sum((1, 2)).float() / (m | gt_mask).sum((1, 2)).float()).mean())
        panels = [(rgb.clamp(0, 1).numpy() * 255).astype(np.uint8) for rgb, _ in renders]
        side = np.concatenate(panels, axis=2)
        tag = np.repeat(banner(name, LABEL_WIDTH, SIZE, 22)[None], side.shape[0], 0)
        rows.append(np.concatenate((tag, side), axis=2))
    grid = np.concatenate(rows, axis=1)

    for (label, v) in runs[1:]:
        report[f"{label} | 3D 頂點誤差 %"] = 100.0 * float((v - truth).norm(dim=-1).mean()) / diag
    heads = [banner(label, SIZE, 44, 24) for label, _ in runs]
    head = np.concatenate([banner("", LABEL_WIDTH, 44, 24), *heads], axis=1)
    grid = np.concatenate((np.repeat(head[None], grid.shape[0], 0), grid), axis=1)

    out = ROOT / "outputs/fox_comparison"
    out.mkdir(exist_ok=True)
    # Slow enough to read, looped so the short walk cycle is easy to follow.
    frames = list(grid) * 3
    imageio.mimsave(out / "fox_comparison.mp4", frames, fps=8, macro_block_size=2, quality=9)
    imageio.imwrite(out / "fox_comparison.png", grid[grid.shape[0] // 2 + 3])
    with open(out / "report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
