"""Silhouette IoU of one or more results against the sequence's SAM2 masks, input view, hard edges.

  python tests/silhouette_iou.py --seq data/paper_cat --result outputs/x/stage1.npz outputs/x/stage1_arap.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.render import RasterRenderer, render_in_chunks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--result", nargs="+", required=True)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
seq, S = Path(args.seq), args.size
mesh = RiggedGLB(seq / "rigged.glb").mesh()
cam = Camera.from_json(seq / "camera.json")
cam.image_size = S
masks = np.stack([np.asarray(Image.open(p).convert("L").resize((S, S), Image.NEAREST)) > 127
                  for p in sorted((seq / "masks").glob("*.png"))])
for r in args.result:
    v = torch.from_numpy(np.load(r)["verts"]).float()
    _, m = render_in_chunks(RasterRenderer(mesh, cam, torch.device(args.device), sigma=1e-6), v.to(args.device))
    m = m.cpu().numpy() > 0.5
    print(f"{r}: silhouette IoU {np.mean([(a & b).sum() / (a | b).sum() for a, b in zip(m, masks)]):.4f}")
