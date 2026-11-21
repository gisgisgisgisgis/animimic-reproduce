"""Stage 2: optimize Young's modulus so the simulated body matches the video (paper Sec. 3.2)."""
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from scipy.spatial import cKDTree

from .device import physics_device
from .fem import TetFEM
from .losses import depth_loss, normalize_depth_targets, point_positions, smooth_loss, track_loss
from .render import build_renderer
from .rig import Rig, linear_blend_skinning
from .sequence import Sequence


@dataclass
class Stage2Config:
    iters_per_level: list = field(default_factory=lambda: [30, 30])  # coarse-to-fine levels; assumption
    lr: float = 1e-3  # stated in the paper, applied to log E
    init_E: float = 1e5  # assumption
    nu: float = 0.4  # assumption
    density: float = 1000.0  # assumption
    fps: float = 24.0  # frames per second of the sequence as loaded: N frames sampled from a clip of length T
                       # seconds is (N - 1) / T, e.g. 32 frames from 121 frames at 24 fps is 6.2, not 24
    substeps: int = 2  # assumption
    newton_rel_tol: float = 1e-4  # Newton stops when the force residual has dropped this much
    batch_frames: int = 4
    w_rgb: float = 1.0
    w_mask: float = 1.0
    w_track: float = 10.0
    w_depth: float = 1.0
    w_smooth: float = 10.0
    depth_norm: str = "first_frame"
    sigma: float = 1e-4  # rasterizer edge softness; use the same value as Stage 1


def embed_points(points, tet_verts, tets, k=24, clamp=False):
    """For each point, find a containing (or nearest) tet and its barycentric weights (paper Sec. 3.2.2).

    fTetWild only approximates the surface, and about 70% of the wolf's and dog's surface vertices lie slightly
    outside every tet. Their weights are then partly negative, which extrapolates the tet's affine map and
    reproduces the rest surface exactly. Clamping them to [0, 1] (the first version, clamp=True) moved the rest
    surface by up to half an edge and added about 2% of stretch on every frame, all over the body."""
    centroids = tet_verts[tets].mean(1)
    _, cand = cKDTree(centroids).query(points, k=min(k, len(tets)))
    cand = cand.reshape(len(points), -1)
    tet = tet_verts[tets[cand]]  # (P, k, 4, 3)
    Dm = np.swapaxes(tet[:, :, 1:] - tet[:, :, :1], -1, -2)
    w = np.linalg.solve(Dm, (points[:, None] - tet[:, :, 0])[..., None])[..., 0]
    bary = np.concatenate((1.0 - w.sum(-1, keepdims=True), w), -1)
    best = bary.min(-1).argmax(1)
    rows = np.arange(len(points))
    chosen, b = cand[rows, best], bary[rows, best]
    if clamp:
        b = np.clip(b, 0.0, None)
        b = b / b.sum(1, keepdims=True)
    return chosen, b


def split_clusters(cluster, centroids):
    """Bisect every cluster along its principal axis."""
    out = np.zeros_like(cluster)
    parent = []
    next_id = 0
    for c in np.unique(cluster):
        idx = np.where(cluster == c)[0]
        if len(idx) < 2:
            out[idx] = next_id
            parent.append(c)
            next_id += 1
            continue
        p = centroids[idx] - centroids[idx].mean(0)
        axis = np.linalg.svd(p, full_matrices=False)[2][0]
        side = (p @ axis) > 0
        out[idx[~side]], out[idx[side]] = next_id, next_id + 1
        parent += [c, c]
        next_id += 2
    return out, np.asarray(parent)


def run_stage2(seq: Sequence, stage1, tet_verts, tets, cfg: Stage2Config, device, backend="auto", log=print,
               E_fixed=None):
    """E_fixed (per tet): skip the optimization and only simulate with these moduli."""
    mesh, rig, cam, tr = seq.mesh, seq.rig, seq.camera, seq.tracks
    dev = physics_device(device)
    if dev != device:
        log(f"[info] Physics runs on {dev} because {device} has no float64 support.")
    renderer = build_renderer(mesh, cam, device, backend, cfg.sigma)

    # Copy skin weights to tet nodes from the nearest surface vertex.
    surface = mesh.verts.cpu().numpy()
    near = cKDTree(surface).query(tet_verts)[1]
    near = torch.from_numpy(near).to(device)
    tet_rig = Rig(rig.joints, rig.parents, rig.skin_idx[near], rig.skin_w[near])
    X_f32 = torch.as_tensor(tet_verts, dtype=torch.float32, device=device)
    # Move between devices and change precision as two steps: MPS cannot hold float64 even briefly.
    x0 = linear_blend_skinning(tet_rig, X_f32, stage1["A_r"][0], stage1["A_t"][0]).to(dev).to(torch.float64)

    fem = TetFEM(tet_verts, tets, cfg.density, cfg.nu, dev)
    tets_np = fem.tets.cpu().numpy()
    centroids = tet_verts[tets_np].mean(1)

    tet_of_v, beta = embed_points(surface.astype(np.float64), tet_verts, tets_np)
    tet_of_v = torch.from_numpy(tet_of_v).to(dev)
    beta = torch.from_numpy(beta).to(dev, torch.float64)

    # Each joint drives the four nodes of its nearest tet.
    joints = rig.joints.cpu().numpy()
    near_tet = cKDTree(centroids).query(joints)[1]
    pinned, joint_of_pin = [], []
    for j, tt in enumerate(near_tet):
        for node in tets_np[tt]:
            if node not in pinned:
                pinned.append(int(node))
                joint_of_pin.append(j)
    pinned_idx = torch.tensor(pinned, device=dev)
    joint_of_pin = torch.tensor(joint_of_pin, device=device)

    A_r, A_t = stage1["A_r"][:, joint_of_pin], stage1["A_t"][:, joint_of_pin]
    X_pin = X_f32[torch.tensor(pinned, device=device)]
    frames_pin = ((A_r @ X_pin[..., None])[..., 0] + A_t).to(dev).to(torch.float64)
    u = torch.linspace(0.0, 1.0, cfg.substeps + 1, device=dev, dtype=torch.float64)[:-1][:, None, None]
    traj = torch.cat([
        (1.0 - u) * frames_pin[t][None] + u * frames_pin[t + 1][None] for t in range(frames_pin.shape[0] - 1)
    ] + [frames_pin[-1][None]])
    h = 1.0 / (cfg.fps * cfg.substeps)
    # Newton starts every step from the skinned pose of all tet nodes (Stage 1), interpolated like the pins.
    lbs = torch.stack([linear_blend_skinning(tet_rig, X_f32, stage1["A_r"][t], stage1["A_t"][t])
                       for t in range(stage1["A_r"].shape[0])]).to(dev).to(torch.float64)
    guess = torch.cat([(1.0 - u) * lbs[t][None] + u * lbs[t + 1][None] for t in range(lbs.shape[0] - 1)] + [lbs[-1][None]])
    sim_kw = {"guess_traj": guess, "rel_tol": cfg.newton_rel_tol}

    cluster = cKDTree(joints).query(centroids)[1]
    _, cluster = np.unique(cluster, return_inverse=True)
    theta = torch.full((int(cluster.max()) + 1,), float(np.log(cfg.init_E)), device=dev, dtype=torch.float64)

    z_target = None
    if tr.depth is not None:
        rest = point_positions(mesh.verts[None], mesh.faces, tr.face, tr.bary)[0]
        z_target = normalize_depth_targets(tr.depth, cam.project(rest)[1], tr.visible, cfg.depth_norm)

    v0 = torch.zeros_like(x0)
    history, xs, E = [], None, None
    for level, iters in enumerate(cfg.iters_per_level):
        if level > 0:
            cluster, parent = split_clusters(cluster, centroids)
            theta = theta[torch.from_numpy(parent).to(dev)]
        cl = torch.from_numpy(cluster).to(dev)
        theta = theta.detach().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=cfg.lr)
        for it in range(iters):
            E = torch.exp(theta.detach())[cl]
            xs = fem.simulate(x0, v0, E, pinned_idx, traj, h, **sim_kw)

            frames_x = xs[:: cfg.substeps].clone().requires_grad_(True)
            tet_nodes = fem.tets[tet_of_v]
            verts = (frames_x[:, tet_nodes] * beta[None, ..., None]).sum(-2).to(torch.float32).to(device)
            xy, z = cam.project(point_positions(verts, mesh.faces, tr.face, tr.bary))
            terms = {"track": track_loss(xy, tr, cam.image_size), "smooth": smooth_loss(verts)}
            if z_target is not None:
                terms["depth"] = depth_loss(z, z_target, tr.visible)
            pick = torch.randperm(seq.num_frames, device=device)[:cfg.batch_frames]
            rgb_pred, mask_pred = renderer(verts[pick])
            terms["mask"] = (mask_pred - seq.mask[pick]).abs().mean()
            if rgb_pred is not None:
                terms["rgb"] = (rgb_pred - seq.rgb[pick]).abs().mean()
            loss = sum(getattr(cfg, f"w_{k}") * v for k, v in terms.items())
            loss.backward()

            dL_dx = torch.zeros_like(xs)
            dL_dx[:: cfg.substeps] = frames_x.grad.to(xs)
            dL_dE = fem.adjoint_gradient(xs, E, pinned_idx, h, dL_dx)
            g = torch.zeros_like(theta)
            g.index_add_(0, cl, E * dL_dE)
            opt.zero_grad()
            theta.grad = g
            opt.step()

            row = {"level": level, "iter": it, "loss": float(loss), **{k: float(v) for k, v in terms.items()}}
            history.append(row)
            log(" ".join(f"{k}={v:.5f}" if k not in ("iter", "level") else f"{k}={v}" for k, v in row.items()))

    with torch.no_grad():
        E = torch.exp(theta.detach())[torch.from_numpy(cluster).to(dev)]
        if E_fixed is not None:
            E = torch.as_tensor(E_fixed, dtype=torch.float64, device=dev)
        xs = fem.simulate(x0, v0, E, pinned_idx, traj, h, **sim_kw)
        verts = (xs[:: cfg.substeps][:, fem.tets[tet_of_v]] * beta[None, ..., None]).sum(-2).to(torch.float32).to(device)
    return {"E": E, "cluster": cluster, "verts": verts, "tet_x": xs[:: cfg.substeps].to(torch.float32).cpu(),
            "history": history, "config": asdict(cfg)}
