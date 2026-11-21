"""A rigged mesh with a baked texture (scripts/bake_texture.py) instead of the asset's own UVs.

The baked mesh has its own vertex list (an xatlas UV atlas splits vertices along chart seams), so
every per-frame vertex array from Stage 1 is mapped with `src_idx` before rendering:
`verts_baked = stage1_verts[:, src_idx]`.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .gltf_rig import MeshData, RiggedGLB


def weld(verts: torch.Tensor, tol: float = 1e-5):
    """Merge vertices that share a position. Returns (group (V,), first original index per group (G,))."""
    _, group = torch.unique(torch.round(verts.double() / tol).long(), dim=0, return_inverse=True)
    n = int(group.max()) + 1
    first = torch.full((n,), len(verts), dtype=torch.long).scatter_reduce(
        0, group, torch.arange(len(verts)), "amin", include_self=True)
    return group, first


def rasterize(tri: torch.Tensor, z: torch.Tensor, height: int, width: int, chunk: int = 4096, eps: float = 1e-4):
    """Hard z-buffer rasterization of triangles given in pixel coordinates whose centers are integers.

    tri (F, 3, 2) as (x, y), z (F, 3). Returns face id (H, W) with -1 for empty, barycentrics (H, W, 3)
    and depth (H, W) with inf for empty. Depth is interpolated linearly in screen space.
    """
    pid_all, fid_all, bary_all, z_all = [], [], [], []
    for s in range(0, len(tri), chunk):
        t, tz = tri[s:s + chunk].double(), z[s:s + chunk].double()
        lo = t.amin(1).ceil().long()
        hi = t.amax(1).floor().long()
        lo[:, 0].clamp_(0, width - 1), lo[:, 1].clamp_(0, height - 1)
        hi[:, 0].clamp_(-1, width - 1), hi[:, 1].clamp_(-1, height - 1)
        w = (hi[:, 0] - lo[:, 0] + 1).clamp_min(0)
        h = (hi[:, 1] - lo[:, 1] + 1).clamp_min(0)
        count = w * h
        f = torch.repeat_interleave(torch.arange(len(t)), count)
        start = torch.cat((torch.zeros(1, dtype=torch.long), count.cumsum(0)[:-1]))
        local = torch.arange(len(f)) - start[f]
        px = lo[f, 0] + local % w[f]
        py = lo[f, 1] + local // w[f]
        p = torch.stack((px, py), -1).double()
        a, b, c = t[f, 0], t[f, 1], t[f, 2]
        cross = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]  # noqa: E731
        area = cross(b - a, c - a)
        ok = area.abs() > 1e-12
        area = torch.where(ok, area, torch.ones_like(area))
        bary = torch.stack((cross(c - b, p - b), cross(a - c, p - c), cross(b - a, p - a)), -1) / area[:, None]
        keep = ok & (bary >= -eps).all(-1)
        f, px, py, bary = f[keep], px[keep], py[keep], bary[keep].clamp_min(0)
        bary = bary / bary.sum(-1, keepdim=True)
        pid_all.append(py * width + px)
        fid_all.append(f + s)
        bary_all.append(bary.float())
        z_all.append((tz[f] * bary).sum(-1).float())
    pid, fid, bary, zc = map(torch.cat, (pid_all, fid_all, bary_all, z_all))
    depth = torch.full((height * width,), float("inf")).scatter_reduce(0, pid, zc, "amin", include_self=True)
    win = zc <= depth[pid]
    face = torch.full((height * width,), -1, dtype=torch.long)
    face[pid[win]] = fid[win]
    out_bary = torch.zeros(height * width, 3)
    out_bary[pid[win]] = bary[win]
    return face.view(height, width), out_bary.view(height, width, 3), depth.view(height, width)


def load_baked_mesh(seq) -> tuple:
    """(MeshData at rest, src_idx) for <seq>/texture_baked.npz and <seq>/texture_baked.png."""
    seq = Path(seq)
    d = np.load(seq / "texture_baked.npz")
    rest = RiggedGLB(seq / "rigged.glb").mesh().verts
    src = torch.from_numpy(d["src_idx"]).long()
    tex = torch.from_numpy(np.asarray(Image.open(seq / "texture_baked.png").convert("RGB"), np.float32) / 255.0)
    mesh = MeshData(rest[src], torch.from_numpy(d["faces"]).long(), torch.from_numpy(d["uv"]).float(), tex)
    return mesh, src
