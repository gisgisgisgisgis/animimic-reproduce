"""Our result against the paper's own result on the same reference video (from the project page).

Both are scored against the reference video at the 32 frames prep_02 sampled:
- SSIM and LPIPS as in Table 1 (tests/paper_metrics.py).
- Silhouette IoU against the SAM2 masks. This one does not depend on texture: our texture is frame 0
  projected onto a reconstructed mesh, while the paper renders the original textured asset.
Also writes a side-by-side video: reference | paper | ours | ours at +90 deg | ours at -45 deg.

--texture baked uses the texture from scripts/bake_texture.py. --sigma sets the rasterizer's edge
softness: Stage 1 optimizes with 1e-4, which blurs edges by several pixels; 1e-6 is close to hard edges.

Usage: python tests/compare_with_paper.py --seq data/paper_wolf --paper data/paper_refs/ours_5.mp4 \
         --result outputs/paper_wolf/stage1.npz --out outputs/paper_wolf/compare_with_paper
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import lpips
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "outputs/horse_diagnosis"))

from compare_rigs import side_camera  # noqa: E402
from paper_metrics import scores  # noqa: E402

from animamimic.baked import load_baked_mesh  # noqa: E402
from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.render import RasterRenderer, render_in_chunks  # noqa: E402


def white_mask(rgb, threshold=15 / 255):
    return np.abs(rgb - 1.0).max(-1) > threshold


def iou(a, b):
    return float(np.mean([(x & y).sum() / max((x | y).sum(), 1) for x, y in zip(a, b)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--paper", required=True, help="the paper's result video for this sequence")
    ap.add_argument("--result", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--texture", choices=("rig", "baked"), default="rig")
    ap.add_argument("--sigma", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    S, device = args.size, torch.device(args.device)
    seq = Path(args.seq)
    frames = sorted((seq / "frames").glob("*.png"))
    load = lambda f: np.asarray(Image.open(f).convert("RGB").resize((S, S), Image.LANCZOS)).astype(np.float32) / 255.0  # noqa: E731
    video = np.stack([load(f) for f in frames])
    masks = np.stack([np.asarray(Image.open(f).convert("L").resize((S, S), Image.NEAREST)) > 127
                      for f in sorted((seq / "masks").glob("*.png"))])

    # Same frame indices as prep_02_import_video.py.
    paper_all = [f for f in imageio.get_reader(args.paper).iter_data()]
    pick = np.linspace(0, len(paper_all) - 1, len(frames)).round().astype(int)
    paper = np.stack([np.asarray(Image.fromarray(paper_all[i][..., :3]).resize((S, S), Image.LANCZOS))
                      for i in pick]).astype(np.float32) / 255.0

    if args.texture == "baked":
        mesh, src = load_baked_mesh(seq)
    else:
        mesh, src = RiggedGLB(seq / "rigged.glb").mesh(), None
    renderer = lambda c: RasterRenderer(mesh, c, device, sigma=args.sigma)  # noqa: E731
    draw = lambda c, v: [x.cpu() for x in render_in_chunks(renderer(c), v.to(device))]  # noqa: E731
    cam = Camera.from_json(seq / "camera.json", device)
    cam.image_size = S
    net = lpips.LPIPS(net="alex", verbose=False)

    report = {"paper (their mesh and texture)": {**scores(paper, video, net), "mask_iou": round(iou(white_mask(paper), masks), 4)}}
    rest = mesh.verts[None].expand(len(frames), -1, -1)
    rgb, alpha = draw(cam, rest)
    report["ours, rest pose (no motion)"] = {**scores(rgb.clamp(0, 1).numpy(), video, net),
                                             "mask_iou": round(iou(alpha.numpy() > 0.5, masks), 4)}
    panels = None
    for path in args.result:
        verts = torch.from_numpy(np.load(path)["verts"]).float()
        if src is not None:
            verts = verts[:, src]
        rgb, alpha = draw(cam, verts)
        rgb = rgb.clamp(0, 1).numpy()
        report[str(path)] = {**scores(rgb, video, net), "mask_iou": round(iou(alpha.numpy() > 0.5, masks), 4)}
        if panels is None:
            sides = [draw(side_camera(cam, a), verts)[0].clamp(0, 1).numpy()
                     for a in (90.0, -45.0)]
            panels = [video, paper, rgb] + sides

    grid = (np.concatenate(panels, axis=2) * 255).astype(np.uint8)
    imageio.mimsave(f"{args.out}.mp4", list(grid) * 2, fps=8, macro_block_size=2, quality=9)
    imageio.imwrite(f"{args.out}.png", np.concatenate([grid[i] for i in (0, len(grid) // 2, len(grid) - 1)], axis=0))
    print(json.dumps(report, indent=1, ensure_ascii=False))
    json.dump(report, open(f"{args.out}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
