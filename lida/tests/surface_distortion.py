"""How much a result stretches the surface relative to the rest mesh, and how much it jitters. Run in `animamimic`:

  python tests/surface_distortion.py --seq data/paper_wolf --result outputs/x/stage1.npz outputs/x/stage2.npz

The paper's claim for Stage 2 (Sec. 4.3, Fig. 6) is qualitative: skinning distorts regions with large local
deformation, and the simulation "removes these surface-mesh artifacts". Measured per frame on the surface mesh:
- strain: |edge length / rest length - 1|, mean and 99th percentile over edges (rest lengths floored at a quarter
  of the median edge, as in Stage 1's rigidity term, so sliver edges do not dominate);
- jitter: second difference of vertex positions between frames, mean over vertices, as % of the mesh size.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animamimic.gltf_rig import RiggedGLB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--result", nargs="+", required=True)
ap.add_argument("--out", default=None, help="write the numbers as json")
args = ap.parse_args()

mesh = RiggedGLB(Path(args.seq) / "rigged.glb").mesh()
V0, Fc = mesh.verts.double(), mesh.faces
# weld the triangle soup so seams count once
key = torch.round(V0 * 1e5).long()
_, weld = torch.unique(key, dim=0, return_inverse=True)
edges = torch.cat([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
we = torch.sort(weld[edges], dim=1).values
_, first = np.unique(we.numpy(), axis=0, return_index=True)
edges = edges[torch.from_numpy(np.sort(first))]
edges = edges[weld[edges[:, 0]] != weld[edges[:, 1]]]
l0 = (V0[edges[:, 0]] - V0[edges[:, 1]]).norm(dim=-1)
l0 = l0.clamp_min(0.25 * l0.median())
size = float((V0.max(0).values - V0.min(0).values).max())
out = {}
for path in args.result:
    V = torch.from_numpy(np.load(path)["verts"]).double()
    l = (V[:, edges[:, 0]] - V[:, edges[:, 1]]).norm(dim=-1)
    strain = (l / l0 - 1.0).abs()
    jitter = (V[2:] - 2 * V[1:-1] + V[:-2]).norm(dim=-1).mean(-1) / size * 100
    out[path] = {"strain_mean": strain.mean(1).numpy().round(5).tolist(),
                 "strain_p99": strain.quantile(0.99, dim=1).numpy().round(5).tolist(),
                 "jitter_pct": jitter.numpy().round(4).tolist()}
    s = out[path]
    print(f"{path}: strain mean {np.mean(s['strain_mean']):.4f} (worst frame {max(s['strain_mean']):.4f}), "
          f"p99 {np.mean(s['strain_p99']):.4f} (worst {max(s['strain_p99']):.4f}), jitter {np.mean(s['jitter_pct']):.3f}% of size")
if args.out:
    json.dump(out, open(args.out, "w"), indent=1)
