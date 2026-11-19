"""Joint fitting and the shared objective following paper Eqs. 4–9."""
import numpy as np
import torch

from .core import save_history
from .losses import normalize_depth, frame_l2, joint_regularization, smoothness, weighted_huber
from .losses import temporal_acceleration, stationary_track_weights
from .rig import fk, lbs, pose_local
from .rig import rotation_6d
from .progress import Progress


def tensors(rig, device):
    return {k: torch.as_tensor(v, device=device,
             dtype=torch.long if k in ("faces", "parents") else torch.float32)
            for k, v in rig.items()}


def video_loss(vertices, rig, target, renderer, progress=None):
    frames, masks = target["frames"], target["masks"]
    tri, beta = target["triangles"], target["barycentric"]
    points = (vertices[:, rig["faces"][tri]] * beta[None, :, :, None]).sum(-2)
    projected = torch.stack([renderer.project(p) for p in points])
    # Exclude only coordinates outside the image (no depth observation exists).
    # The paper does not prescribe a confidence-weighted or Huber objective.
    tracks = target["tracks"]
    h, w = renderer.size
    valid = ((tracks[..., 0] >= 0) & (tracks[..., 0] <= w-1)
             & (tracks[..., 1] >= 0) & (tracks[..., 1] <= h-1))
    track = frame_l2(projected, tracks, valid)
    depths = torch.stack([renderer.depth(p) for p in points])
    canonical_points = (rig["vertices"][rig["faces"][tri]]*beta[..., None]).sum(-2)
    normalized_target = normalize_depth(target["depth"], renderer.depth(canonical_points), valid)
    depth = frame_l2(depths, normalized_target, valid)
    rgb, mask = [], []
    for frame, value in enumerate(vertices, 1):
        if progress:
            progress.detail(f"rendering frame {frame}/{len(vertices)}")
        image, alpha = renderer(value, rig["faces"], rig["colors"])
        rgb.append(image)
        mask.append(alpha)
    rgb, mask = torch.stack(rgb), torch.stack(mask)
    return {"rgb": frame_l2(rgb, frames), "mask": frame_l2(mask, masks),
            "track": track, "depth": depth, "smooth": smoothness(vertices)}


def fit_joints(rig, supervision, renderer, output, config):
    device = rig["vertices"].device
    n = len(supervision["frames"])
    j = len(rig["parents"])
    rotation = torch.tensor([1., 0, 0, 0, 1, 0], device=device).repeat(n, j, 1)
    rotation.requires_grad_()
    translation = torch.zeros(n, j, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([rotation, translation], lr=config.get("learning_rate", 1e-3))
    # Old phase counts remain accepted as a total budget, with one full objective.
    old_counts = ("root_iterations", "geometry_iterations", "full_iterations")
    iterations = config.get("iterations", sum(config.get(k, 0) for k in old_counts)
                            if any(k in config for k in old_counts) else 700)
    if not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Joint iterations must be a positive integer")
    history = []
    parents = rig["parents"].tolist()
    # Include scene ancestors in the root exclusion for Eq. 9.
    roots = rig["parents"] < 0
    # Paper §3.1.2 optimizes rotation AND translation for every joint.
    # Freezing child offsets is an optional experiment, not the baseline.
    movable = torch.ones_like(roots) if config.get("optimize_joint_translations", True) else roots
    times = supervision.get("frame_times", torch.arange(n, device=device)/24)
    if len(times) != n or not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any():
        raise ValueError("Joint frame times must be finite and strictly increasing")
    weights = {"rotation_smooth": 0., "root_smooth": 0., "stationary": 0.,
               **config.get("loss_weights", {})}
    if weights["stationary"]:
        stationary = stationary_track_weights(supervision["tracks"],
            supervision["visibility"]*supervision["confidence"], config.get("stationary_track_px", 2.))
    progress = Progress("joints full", iterations)
    for iteration in range(iterations):
        optimizer.zero_grad()
        effective_translation = translation*movable[None, :, None]
        local = pose_local(rig["local"], rotation, effective_translation)
        world = fk(local, parents)
        vertices = lbs(rig["vertices"], rig["weights"], world, rig["inverse_bind"])
        losses = video_loss(vertices, rig, supervision, renderer, progress=progress)
        matrices = rotation_6d(rotation)
        if weights["rotation_smooth"]:
            losses["rotation_smooth"] = temporal_acceleration(matrices, times)
        if weights["root_smooth"]:
            losses["root_smooth"] = temporal_acceleration(world[:, roots, :3, 3], times)
        if weights["stationary"]:
            points = (vertices[:, rig["faces"][supervision["triangles"]]]
                      * supervision["barycentric"][None, :, :, None]).sum(-2)
            losses["stationary"] = weighted_huber(points/.01, points[:1].expand_as(points)/.01, stationary)
        losses["regularization"] = joint_regularization(rotation, translation, roots,
            config.get("rotation_bound", 1.), config.get("translation_bound", .1))
        loss = sum(weights.get(k, 1)*v for k, v in losses.items())
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite joint objective")
        progress.detail("backward pass")
        loss.backward()
        optimizer.step()
        history.append({"phase": "full", "iteration": iteration,
                        **{k: float(v.detach()) for k, v in losses.items()}})
        progress.advance(float(loss.detach()))
    save_history(output/"joint_losses.json", history)
    np.savez_compressed(output/"joint_checkpoint.npz",
                        rotation=rotation.detach().cpu(), translation=(translation*movable[None, :, None]).detach().cpu())
    with torch.no_grad():
        translation = translation*movable[None, :, None]
        local = pose_local(rig["local"], rotation, translation)
        world = fk(local, parents)
        vertices = lbs(rig["vertices"], rig["weights"], world, rig["inverse_bind"])
    np.savez_compressed(output/"joint_trajectory.npz", vertices=vertices.cpu(),
                        local=local.cpu(), world=world.cpu(),
                        rotation=rotation.detach().cpu(), translation=translation.detach().cpu())
    return ["joint_trajectory.npz", "joint_checkpoint.npz", "joint_losses.json", "joint_losses.csv"]
