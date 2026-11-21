"""Write a prepared-sequence folder from the fox so the real command-line path can be exercised.

It stands in for the video model, SAM2, AllTracker and VGGT, using ground truth.
Run prep_01_render_view.py on the folder first, with the same camera as validate_fox.py.
"""
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pytorch3d.structures import Meshes  # noqa: E402

from animamimic.render import build_renderer  # noqa: E402
from scripts.validate_fox import build_fox_case, make_parser  # noqa: E402


def main():
    ap = make_parser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--video", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    seq_dir = Path(args.seq)
    (seq_dir / "masks").mkdir(parents=True, exist_ok=True)
    seq, gt = build_fox_case(args, device)

    imageio.mimsave(args.video, list((seq.rgb.cpu().numpy() * 255).astype(np.uint8)), fps=16, macro_block_size=1)
    for i, m in enumerate(seq.mask):
        imageio.imwrite(seq_dir / "masks" / f"{i:05d}.png", ((m > 0.5).cpu().numpy() * 255).astype(np.uint8))
    np.savez(seq_dir / "tracks.npz", xy=seq.tracks.xy.cpu().numpy(), visible=seq.tracks.visible.cpu().numpy())

    renderer = build_renderer(seq.mesh, seq.camera, device, "pytorch3d")
    depth = []
    with torch.no_grad():
        for v in gt:
            mesh = Meshes(verts=[v], faces=[seq.mesh.faces])
            z = renderer.rasterizer(mesh).zbuf[0, ..., 0]
            depth.append(z.where(z > 0, torch.full_like(z, float(z[z > 0].max()))))
    np.savez(seq_dir / "depth.npz", depth=(3.7 * torch.stack(depth) + 0.9).cpu().numpy().astype(np.float32))
    print(f"Wrote test sequence to {seq_dir}")


if __name__ == "__main__":
    main()
