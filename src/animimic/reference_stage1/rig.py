"""Skeleton forward kinematics and linear blend skinning (paper Eq. 1-2)."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

IDENTITY_6D = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def rotation_6d_to_matrix(r6: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 2019 Gram-Schmidt map, same row convention as PyTorch3D."""
    a1, a2 = r6[..., :3], r6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(m: torch.Tensor) -> torch.Tensor:
    return m[..., :2, :].flatten(start_dim=-2)


@dataclass
class Rig:
    joints: torch.Tensor  # (K, 3) rest positions
    parents: list  # length K, parents[i] < i, -1 for roots
    skin_idx: torch.Tensor  # (V, M) joint indices
    skin_w: torch.Tensor  # (V, M) weights, sum to 1 per vertex

    @property
    def num_joints(self) -> int:
        return self.joints.shape[0]

    def to(self, device, dtype=None) -> "Rig":
        dtype = dtype or self.joints.dtype
        return Rig(
            self.joints.to(device, dtype), self.parents,
            self.skin_idx.to(device), self.skin_w.to(device, dtype),
        )

    def offsets(self) -> torch.Tensor:
        out = self.joints.clone()
        for j, p in enumerate(self.parents):
            if p >= 0:
                out[j] = self.joints[j] - self.joints[p]
        return out


def forward_kinematics(rig: Rig, r6d: torch.Tensor, t: torch.Tensor):
    """Return rest-to-posed skinning transforms (A_r, A_t).

    r6d: (..., K, 6) local rotations, t: (..., K, 3) local translations.
    A_r: (..., K, 3, 3), A_t: (..., K, 3). A posed point is A_r X + A_t.
    """
    R = rotation_6d_to_matrix(r6d)
    offsets = rig.offsets()
    g_r, g_t = [None] * rig.num_joints, [None] * rig.num_joints
    for j, p in enumerate(rig.parents):
        local = offsets[j] + t[..., j, :]
        if p < 0:
            g_r[j], g_t[j] = R[..., j, :, :], local
        else:
            g_r[j] = g_r[p] @ R[..., j, :, :]
            g_t[j] = g_t[p] + (g_r[p] @ local[..., None])[..., 0]
    A_r = torch.stack(g_r, dim=-3)
    G_t = torch.stack(g_t, dim=-2)
    A_t = G_t - (A_r @ rig.joints[..., None])[..., 0]
    return A_r, A_t


def linear_blend_skinning(rig: Rig, verts: torch.Tensor, A_r, A_t) -> torch.Tensor:
    """verts: (V, 3). Returns (..., V, 3)."""
    Rv = A_r[..., rig.skin_idx, :, :]
    tv = A_t[..., rig.skin_idx, :]
    posed = (Rv @ verts[:, None, :, None])[..., 0] + tv
    return (posed * rig.skin_w[..., None]).sum(-2)


def bone_distance(rig: Rig, verts: torch.Tensor) -> torch.Tensor:
    """Distance from each vertex (V, 3) to its nearest bone segment. Returns (V,)."""
    pairs = [(j, p) for j, p in enumerate(rig.parents) if p >= 0]
    head = rig.joints[[p for _, p in pairs]]
    bone = rig.joints[[j for j, _ in pairs]] - head
    rel = verts[:, None] - head
    u = ((rel * bone).sum(-1) / (bone * bone).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
    return (rel - u[..., None] * bone).norm(dim=-1).min(dim=1).values


def smooth_skin_weights(rig: Rig, verts: torch.Tensor, faces: torch.Tensor, steps: int, lam: float = 0.5,
                        weld_tol: float = 1e-5) -> Rig:
    """Laplacian-smooth the skin weights over the surface, keeping the same number of influences per vertex.

    UniRig's weights can switch from one joint to another across a single edge. Any relative motion of
    those two joints then stretches that edge. Vertices that share a position (UV seams) are merged first,
    so both sides of a seam keep identical weights. Runs once on the CPU.
    """
    if steps <= 0:
        return rig
    device, dtype = rig.skin_w.device, rig.skin_w.dtype
    idx, w = rig.skin_idx.cpu(), rig.skin_w.cpu().double()
    V, K, M = idx.shape[0], rig.num_joints, idx.shape[1]
    dense = torch.zeros(V, K, dtype=torch.float64).scatter_add_(1, idx, w)

    _, group = torch.unique(torch.round(verts.cpu().double() / weld_tol).long(), dim=0, return_inverse=True)
    n = int(group.max()) + 1
    count = torch.zeros(n, dtype=torch.float64).index_add_(0, group, torch.ones(V, dtype=torch.float64))
    x = torch.zeros(n, K, dtype=torch.float64).index_add_(0, group, dense) / count[:, None]

    f = group[faces.cpu().long()]
    e = torch.cat((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])).sort(dim=1).values
    e = torch.unique(e[e[:, 0] != e[:, 1]], dim=0)
    a, b = e[:, 0], e[:, 1]
    ones = torch.ones(len(a), dtype=torch.float64)
    degree = torch.zeros(n, dtype=torch.float64).index_add_(0, a, ones).index_add_(0, b, ones)
    for _ in range(steps):
        total = torch.zeros_like(x).index_add_(0, a, x[b]).index_add_(0, b, x[a])
        mean = torch.where(degree[:, None] > 0, total / degree.clamp_min(1.0)[:, None], x)
        x = (1.0 - lam) * x + lam * mean

    val, new_idx = x[group].topk(M, dim=1)
    val = val / val.sum(1, keepdim=True)
    return Rig(rig.joints, rig.parents, new_idx.to(device), val.to(device, dtype))


def identity_pose(num_frames: int, num_joints: int, device, dtype=torch.float32):
    r6d = torch.tensor(IDENTITY_6D, device=device, dtype=dtype).repeat(num_frames, num_joints, 1)
    return r6d, torch.zeros(num_frames, num_joints, 3, device=device, dtype=dtype)
