"""Paper Table 1 image metrics: SSIM and LPIPS of the input-view render against the reference video.

The paper does not say whether frames are masked or which background is used. Here both the video
and the render are compared as full frames: the video as generated, the render over white.
A rest-pose render (no motion) is reported alongside, to show how much of the score is free.

Usage: python tests/paper_metrics.py --seq data/horse_rebaked --result outputs/horse_rebaked_tlimit_smooth/stage1.npz
"""
import argparse
import json
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.render import RasterRenderer, render_in_chunks  # noqa: E402


def scores(pred, video, net):
    """pred, video: (T, H, W, 3) float in [0, 1]."""
    ssim = [structural_similarity(p, v, channel_axis=2, data_range=1.0) for p, v in zip(pred, video)]
    with torch.no_grad():
        a = torch.from_numpy(pred).permute(0, 3, 1, 2).float() * 2 - 1
        b = torch.from_numpy(video).permute(0, 3, 1, 2).float() * 2 - 1
        lp = torch.cat([net(a[i:i + 8], b[i:i + 8]).flatten() for i in range(0, len(a), 8)])
    return {"ssim": round(float(np.mean(ssim)), 4), "lpips": round(float(lp.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--result", nargs="+", required=True, help="npz files with verts (T, V, 3)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cpu")
    seq = Path(args.seq)
    mesh = RiggedGLB(seq / "rigged.glb").mesh()
    cam = Camera.from_json(seq / "camera.json", device)
    cam.image_size = args.size
    frames = sorted((seq / "frames").glob("*.png"))
    video = np.stack([
        np.asarray(Image.open(f).convert("RGB").resize((args.size, args.size), Image.LANCZOS)) for f in frames
    ]).astype(np.float32) / 255.0
    renderer = RasterRenderer(mesh, cam, device)
    net = lpips.LPIPS(net="alex", verbose=False)

    def render(verts):
        rgb, _ = render_in_chunks(renderer, verts)
        return rgb.clamp(0, 1).numpy().astype(np.float32)

    report = {"rest_pose_no_motion": scores(render(mesh.verts[None].expand(len(frames), -1, -1)), video, net)}
    for path in args.result:
        verts = torch.from_numpy(np.load(path)["verts"]).float()
        report[str(path)] = scores(render(verts), video, net)
    report["paper_table1_ours"] = {"ssim": 0.9318, "lpips": 0.0849}
    print(json.dumps(report, indent=1))
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
