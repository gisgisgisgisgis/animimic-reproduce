"""Remove surface stretch from a skinned result with an as-rigid-as-possible (ARAP) projection per frame.

Not in the paper. Linear blend skinning stretches and shears the surface where a region is blended between
bones that rotate differently (on the cat, chest and shoulders between spine and neck). Per frame, this finds
vertex positions x that stay close to the skinned result y while every vertex's one-ring stays a rotated copy
of its rest shape (Sorkine & Alexa 2007):

    min_x,R  sum_ij w_ij |(x_i - x_j) - R_i (p_i - p_j)|^2 + lam * sum_i a_i |x_i - y_i|^2

w_ij are cotangent weights (clamped positive), a_i lumped vertex areas normalized to mean 1. Bending is free
(each one-ring may rotate); stretch and shear cost. Larger --lam follows the skinned result more closely.

  python scripts/arap_refine.py --seq data/paper_cat --result outputs/paper_cat_sharp_rigid/stage1.npz \
      --out outputs/paper_cat_sharp_rigid/stage1_arap.npz --lam 0.1

The output keeps r6d and t and replaces verts, so the comparison tools read it like stage1.npz.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animamimic.gltf_rig import RiggedGLB  # noqa: E402


def weld(verts, tol=1e-5):
    _, first, group = np.unique(np.round(verts / tol).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    return group.reshape(-1), first


def cotan_edges(P, F):
    """Undirected edges (i < j) with summed cotangent weights 0.5 * (cot a + cot b)."""
    rows, vals = [], []
    for k in range(3):
        a, b, c = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        u, v = P[b] - P[a], P[c] - P[a]
        cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-12)
        rows.append(np.sort(np.stack([b, c], 1), 1))  # the edge opposite corner a
        vals.append(0.5 * cot)
    e, w = np.concatenate(rows), np.concatenate(vals)
    key, inv = np.unique(e, axis=0, return_inverse=True)
    w = np.bincount(inv.reshape(-1), w, minlength=len(key))
    return key, np.maximum(w, 1e-3 * np.abs(w).mean())  # obtuse pairs give negative weights; keep them small


def vertex_areas(P, F, n):
    area = 0.5 * np.linalg.norm(np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]), axis=1)
    a = np.zeros(n)
    for k in range(3):
        np.add.at(a, F[:, k], area / 3)
    return a / a.mean()


def rotations(S):
    U, _, Vt = np.linalg.svd(S)
    R = np.einsum("nji,nkj->nik", Vt, U)  # V U^T
    bad = np.linalg.det(R) < 0
    if bad.any():
        Vt[bad, 2] *= -1
        R[bad] = np.einsum("nji,nkj->nik", Vt[bad], U[bad])
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.1, help="weight of staying at the skinned positions")
    ap.add_argument("--iters", type=int, default=15, help="local/global iterations per frame")
    args = ap.parse_args()
    t0 = time.time()
    mesh = RiggedGLB(Path(args.seq) / "rigged.glb").mesh()
    V0, Fs = mesh.verts.double().numpy(), mesh.faces.numpy()
    group, first = weld(V0)
    P, F = V0[first], group[Fs]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])]
    n = len(P)
    E, w = cotan_edges(P, F)
    i, j = E[:, 0], E[:, 1]
    d0 = P[i] - P[j]
    L = sp.coo_matrix((np.concatenate([w, w, -w, -w]), (np.concatenate([i, j, i, j]), np.concatenate([i, j, j, i]))),
                      shape=(n, n)).tocsc()
    a = vertex_areas(P, F, n)
    solve = splu((L + args.lam * sp.diags(a)).tocsc()).solve

    data = dict(np.load(args.result))
    Y_all = data["verts"].astype(np.float64)
    out = np.empty_like(Y_all)
    for t in range(len(Y_all)):
        # the welded copies of a vertex sit at the same skinned position; average them anyway
        y = np.zeros((n, 3))
        np.add.at(y, group, Y_all[t])
        y /= np.bincount(group, minlength=n)[:, None]
        x = y.copy()
        for _ in range(args.iters):
            dx = x[i] - x[j]
            outer = w[:, None, None] * np.einsum("ea,eb->eab", d0, dx)
            S = np.zeros((n, 3, 3))
            np.add.at(S, i, outer)
            np.add.at(S, j, outer)
            R = rotations(S)
            rhs_e = 0.5 * w[:, None] * np.einsum("eab,eb->ea", R[i] + R[j], d0)
            b = np.zeros((n, 3))
            np.add.at(b, i, rhs_e)
            np.add.at(b, j, -rhs_e)
            b += args.lam * a[:, None] * y
            x = np.stack([solve(b[:, k]) for k in range(3)], 1)
        out[t] = x[group]
    data["verts"] = out.astype(np.float32)
    data["arap_lam"] = np.float32(args.lam)
    np.savez(args.out, **data)
    move = np.linalg.norm(out - Y_all, axis=-1)
    print(f"{n} welded vertices, {len(E)} edges; moved mean {move.mean():.4f}, p99 {np.percentile(move, 99):.4f} "
          f"(mesh size 1); {time.time() - t0:.0f} s; wrote {args.out}")


if __name__ == "__main__":
    main()
