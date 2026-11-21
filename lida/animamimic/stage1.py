"""Stage 1: optimize joint rotations and translations against the reference video (paper Sec. 3.1)."""
from dataclasses import asdict, dataclass

import torch

from .losses import (
    depth_loss, mean_depth_loss, normalize_depth_targets, point_positions, pose_regularization, smooth_loss, track_loss,
)
from .render import build_renderer
from .rig import IDENTITY_6D, forward_kinematics, identity_pose, linear_blend_skinning
from .sequence import Sequence


@dataclass
class Stage1Config:
    iters: int = 3000
    lr: float = 1e-3  # stated in the paper
    batch_frames: int = 4  # frames rendered per iteration; assumption
    # The paper does not publish these weights or thresholds.
    w_rgb: float = 1.0
    w_mask: float = 1.0
    w_track: float = 10.0
    w_depth: float = 1.0
    w_smooth: float = 10.0
    w_reg: float = 1.0
    rot_max: float = 1.0  # threshold on 6D offsets
    trans_max: float = 0.05  # threshold in units of the normalized mesh
    depth_norm: str = "first_frame"
    depth_mode: str = "points"  # points: Eq. 8 per tracked point. mean: only each frame's mean depth.
    sigma: float = 1e-4  # rasterizer edge softness for L_rgb and L_mask
    # Not in the paper: keep every mesh edge near its rest length (squared relative change). Stops leaf
    # joints that few tracks constrain (snout tip, ear tips, paws) from twisting the surface around them.
    w_rigid: float = 0.0


def run_stage1(seq: Sequence, cfg: Stage1Config, device, backend="auto", log_every=100, log=print):
    T, rig, mesh, cam, tr = seq.num_frames, seq.rig, seq.mesh, seq.camera, seq.tracks
    renderer = build_renderer(mesh, cam, device, backend, cfg.sigma)
    edges = torch.cat([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]).long()
    rest_len = (mesh.verts[edges[:, 0]] - mesh.verts[edges[:, 1]]).norm(dim=-1)
    keep = rest_len > 1e-6
    edges, rest_len = edges[keep], rest_len[keep]
    rest_len = rest_len.clamp_min(0.25 * rest_len.median())  # very short edges would dominate a relative measure
    if not renderer.has_rgb:
        log("[warn] This renderer has no RGB output, so L_rgb is off. Use CUDA for the full loss.")

    z_target = None
    if tr.depth is not None:
        rest = point_positions(mesh.verts[None], mesh.faces, tr.face, tr.bary)[0]
        _, z0 = cam.project(rest)
        z_target = normalize_depth_targets(tr.depth, z0, tr.visible, cfg.depth_norm)

    r6d, t = identity_pose(T, rig.num_joints, device)
    identity = torch.tensor(IDENTITY_6D, device=device, dtype=r6d.dtype)
    r6d.requires_grad_(True)
    t.requires_grad_(True)
    opt = torch.optim.Adam([r6d, t], lr=cfg.lr)
    non_root = torch.tensor([j for j, p in enumerate(rig.parents) if p >= 0], device=device, dtype=torch.long)

    history = []
    for it in range(cfg.iters):
        A_r, A_t = forward_kinematics(rig, r6d, t)
        verts = linear_blend_skinning(rig, mesh.verts, A_r, A_t)

        xy, z = cam.project(point_positions(verts, mesh.faces, tr.face, tr.bary))
        terms = {
            "track": track_loss(xy, tr, cam.image_size),
            "smooth": smooth_loss(verts),
            "reg": pose_regularization(r6d - identity, t, cfg.rot_max, cfg.trans_max, non_root),
        }
        if cfg.w_rigid > 0:
            length = (verts[:, edges[:, 0]] - verts[:, edges[:, 1]]).norm(dim=-1)
            terms["rigid"] = ((length - rest_len) / rest_len).pow(2).mean()
        if z_target is not None:
            if cfg.depth_mode == "points":
                terms["depth"] = depth_loss(z, z_target, tr.visible)
            elif cfg.depth_mode == "mean":
                terms["depth"] = mean_depth_loss(z, z_target, tr.visible)
            else:
                raise ValueError(cfg.depth_mode)

        frames = torch.randperm(T, device=device)[:cfg.batch_frames]
        rgb_pred, mask_pred = renderer(verts[frames])
        terms["mask"] = (mask_pred - seq.mask[frames]).abs().mean()
        if rgb_pred is not None:
            terms["rgb"] = (rgb_pred - seq.rgb[frames]).abs().mean()

        loss = sum(getattr(cfg, f"w_{k}") * v for k, v in terms.items())
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % log_every == 0 or it == cfg.iters - 1:
            row = {"iter": it, "loss": float(loss), **{k: float(v) for k, v in terms.items()}}
            history.append(row)
            log(" ".join(f"{k}={v:.5f}" if k != "iter" else f"iter={v}" for k, v in row.items()))

    with torch.no_grad():
        A_r, A_t = forward_kinematics(rig, r6d, t)
        verts = linear_blend_skinning(rig, mesh.verts, A_r, A_t)
    return {
        "r6d": r6d.detach(), "t": t.detach(), "A_r": A_r, "A_t": A_t,
        "verts": verts, "history": history, "config": asdict(cfg),
    }
