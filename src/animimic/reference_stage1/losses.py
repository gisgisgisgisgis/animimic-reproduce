"""Loss terms of paper Eq. 4-9 and the barycentric lifting of tracked pixels."""
from dataclasses import dataclass

import torch

from .camera import Camera


@dataclass
class TrackTargets:
    face: torch.Tensor  # (N,) mesh face index of each tracked point
    bary: torch.Tensor  # (N, 3) barycentric weights
    xy: torch.Tensor  # (T, N, 2) pixel coordinates
    visible: torch.Tensor  # (T, N) bool
    depth: torch.Tensor = None  # (T, N) raw monocular depth at the tracked pixels


def lift_query_points(verts, faces, camera: Camera, query_xy, chunk=512):
    """Find, for each query pixel, the front-most triangle and perspective-correct barycentrics.

    Returns (keep, face, bary), where keep marks queries that hit the mesh.
    """
    with torch.no_grad():
        xy, z = camera.project(verts)
        a, b, c = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
        ab, ac = b - a, c - a
        denom = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        valid = denom.abs() > 1e-9
        zf = z[faces]

        n = query_xy.shape[0]
        best_z = torch.full((n,), float("inf"), device=verts.device)
        best_f = torch.full((n,), -1, device=verts.device, dtype=torch.long)
        best_b = torch.zeros(n, 3, device=verts.device, dtype=verts.dtype)
        for s in range(0, n, chunk):
            q = query_xy[s:s + chunk]
            d = q[:, None, :] - a[None]
            w1 = (d[..., 0] * ac[:, 1] - d[..., 1] * ac[:, 0]) / denom.where(valid, torch.ones_like(denom))
            w2 = (ab[:, 0] * d[..., 1] - ab[:, 1] * d[..., 0]) / denom.where(valid, torch.ones_like(denom))
            w0 = 1.0 - w1 - w2
            screen = torch.stack((w0, w1, w2), -1)
            inside = valid & (screen >= 0).all(-1)
            inv = screen / zf.clamp_min(1e-6)[None]
            persp = inv / inv.sum(-1, keepdim=True).clamp_min(1e-12)
            depth = (persp * zf[None]).sum(-1).where(inside, torch.full_like(w0, float("inf")))
            val, idx = depth.min(dim=1)
            rows = torch.arange(q.shape[0], device=verts.device)
            hit = torch.isfinite(val)
            best_z[s:s + chunk] = val
            best_f[s:s + chunk] = idx.where(hit, torch.full_like(idx, -1))
            best_b[s:s + chunk] = persp[rows, idx]
        keep = best_f >= 0
        return keep, best_f[keep], best_b[keep]


def point_positions(verts_t, faces, face_idx, bary):
    """verts_t: (T, V, 3). Returns (T, N, 3)."""
    tri = verts_t[:, faces[face_idx]]  # (T, N, 3, 3)
    return (tri * bary[None, :, :, None]).sum(-2)


def normalize_depth_targets(raw, reference_z, visible, mode="first_frame"):
    """Operator N(.) of Eq. 8. The paper does not define it, so both options are configurable.

    first_frame: one scale and shift from frame 0, where mesh depth is known. Keeps motion in depth.
    per_frame: each frame is standardized on its own and matched to the frame-0 statistics.
    """
    v0 = visible[0]
    mean_z, std_z = reference_z[v0].mean(), reference_z[v0].std().clamp_min(1e-6)
    if mode == "first_frame":
        mean_d, std_d = raw[0][v0].mean(), raw[0][v0].std().clamp_min(1e-6)
        return (raw - mean_d) / std_d * std_z + mean_z
    if mode == "per_frame":
        out = raw.clone()
        for t in range(raw.shape[0]):
            v = visible[t]
            out[t] = (raw[t] - raw[t][v].mean()) / raw[t][v].std().clamp_min(1e-6) * std_z + mean_z
        return out
    raise ValueError(mode)


def track_loss(xy_pred, targets: TrackTargets, image_size):
    err = (xy_pred - targets.xy).norm(dim=-1) / image_size
    return err[targets.visible].mean()


def depth_loss(z_pred, z_target, visible):
    return (z_pred - z_target).abs()[visible].mean()


def mean_depth_loss(z_pred, z_target, visible):
    """Match only each frame's mean depth over the visible tracked points.

    Not in the paper. Per-point VGGT depth was noisier than the motion itself on the fox, and pulled
    the horse's shape; dropping depth entirely let the whole horse slide toward the camera. This keeps
    the one thing depth is needed for, the distance of the body, and leaves the shape to the image terms.
    """
    m = visible.to(z_pred.dtype)
    count = m.sum(1).clamp_min(1.0)
    return (((z_pred - z_target) * m).sum(1) / count).abs().mean()


def smooth_loss(verts_t):
    v1 = verts_t[1:] - verts_t[:-1]
    v2 = v1[1:] - v1[:-1]
    return v1.pow(2).sum(-1).mean() + v2.pow(2).sum(-1).mean()


def pose_regularization(d_r6, t, rot_max, trans_max, non_root):
    """Eq. 9, written on the offsets from the identity pose."""
    d_r6, t = d_r6[:, non_root], t[:, non_root]
    r = d_r6 - d_r6.clamp(-rot_max, rot_max)
    q = t - t.clamp(-trans_max, trans_max)
    return r.pow(2).sum(-1).mean() + q.pow(2).sum(-1).mean()
