"""Per-frame surface ARAP projection from the Stage 1 skinned trajectory."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


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
    return a / max(a.mean(), 1e-12)


def rotations(S):
    U, _, Vt = np.linalg.svd(S)
    R = np.einsum("nji,nkj->nik", Vt, U)  # V U^T
    bad = np.linalg.det(R) < 0
    if bad.any():
        Vt[bad, 2] *= -1
        R[bad] = np.einsum("nji,nkj->nik", Vt[bad], U[bad])
    return R


def refine_surface(rest, faces, trajectory, *, weight=.01, iterations=15):
    """Preserve local shape while anchoring the result to the input animation."""
    if not np.isfinite(weight) or weight <= 0 or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("ARAP weight and iteration count must be positive")
    rest = np.asarray(rest, dtype=np.float64)
    group, first = weld(rest)
    points, faces = rest[first], group[faces]
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    faces = faces[distinct]
    if not len(faces):
        return trajectory.copy()
    edges, weights = cotan_edges(points, faces)
    i, j = edges.T
    n = len(points)
    counts = np.bincount(group, minlength=n)[:, None]
    rest_edges = points[i] - points[j]
    laplacian = sp.coo_matrix((np.r_[weights, weights, -weights, -weights],
                               (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n))
    anchors = weight*np.maximum(vertex_areas(points, faces, n), 1e-8)
    solve = splu((laplacian + sp.diags(anchors)).tocsc()).solve
    result = np.empty_like(trajectory)
    for frame, vertices in enumerate(trajectory):
        target = np.zeros_like(points)
        np.add.at(target, group, vertices)
        target /= counts
        position = target.copy()
        for _ in range(iterations):
            outer = weights[:, None, None]*np.einsum("ea,eb->eab", rest_edges, position[i]-position[j])
            covariance = np.zeros((n, 3, 3))
            np.add.at(covariance, i, outer)
            np.add.at(covariance, j, outer)
            rotation = rotations(covariance)
            edge_rhs = .5*weights[:, None]*np.einsum("eab,eb->ea", rotation[i]+rotation[j], rest_edges)
            rhs = anchors[:, None]*target
            np.add.at(rhs, i, edge_rhs)
            np.add.at(rhs, j, -edge_rhs)
            position = solve(rhs)
        result[frame] = position[group]
    return result
