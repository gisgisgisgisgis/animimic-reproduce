"""What causes the surface stretch of a Stage 1 result: re-skin it with parts of the motion or weights changed.

  python tests/strain_ablation.py --seq data/paper_cat --result outputs/paper_cat_sharp_rigid/stage1.npz

Variants: as optimized; non-root joint translations zeroed (bones keep their length); and the same poses
with UniRig's raw weights or other smoothing steps. Strain = |edge length / rest length - 1| per edge of each
face (rest lengths floored at a quarter of the median), reported as median / p90 / p99 over faces, averaged
over frames, plus the worst frame's p99.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.rig import forward_kinematics, linear_blend_skinning, smooth_skin_weights  # noqa: E402


def strain(V, V0, F):
    s = []
    for a, b in [(0, 1), (1, 2), (2, 0)]:
        l0 = (V0[F[:, a]] - V0[F[:, b]]).norm(dim=-1)
        l0 = l0.clamp_min(0.25 * l0.median())
        s.append(((V[:, F[:, a]] - V[:, F[:, b]]).norm(dim=-1) / l0 - 1).abs())
    return torch.stack(s, -1).amax(-1)  # (T, F)


def summary(s):
    q = torch.quantile(s.float(), torch.tensor([0.5, 0.9, 0.99]), dim=1)  # (3, T)
    return f"median {q[0].mean():.3f}  p90 {q[1].mean():.3f}  p99 {q[2].mean():.3f}  worst-frame p99 {q[2].max():.3f}"


ap = argparse.ArgumentParser()
ap.add_argument("--seq", required=True)
ap.add_argument("--result", required=True)
ap.add_argument("--skin_smooth", type=int, default=30, help="the smoothing the result was optimized with")
args = ap.parse_args()
glb = RiggedGLB(Path(args.seq) / "rigged.glb")
mesh, rig0 = glb.mesh(), glb.rig()
d = np.load(args.result)
r6d, t = torch.from_numpy(d["r6d"]).float(), torch.from_numpy(d["t"]).float()
V0, F = mesh.verts.double(), mesh.faces
stored = torch.from_numpy(d["verts"]).double()
print(f"stored result          {summary(strain(stored, V0, F))}")
t0 = t.clone()
t0[:, 1:] = 0
for steps in (args.skin_smooth, 0, 100):
    rig = smooth_skin_weights(rig0, mesh.verts, mesh.faces, steps)
    for name, tt in (("as optimized", t), ("no joint translation", t0)):
        A_r, A_t = forward_kinematics(rig, r6d, tt)
        V = linear_blend_skinning(rig, mesh.verts, A_r, A_t).double()
        if steps == args.skin_smooth and name == "as optimized":
            print(f"  (re-skinned vs stored: max |dv| {float((V - stored).abs().max()):.2e})")
        print(f"smooth {steps:3d}, {name:21s} {summary(strain(V, V0, F))}")
