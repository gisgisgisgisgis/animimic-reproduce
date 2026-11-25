"""Joint fitting and the shared objective following paper Eqs. 4–9."""
import numpy as np
import torch

from .core import save_history, save_json
from .losses import normalize_depth, frame_l2, joint_regularization, smoothness, weighted_huber
from .losses import temporal_acceleration, stationary_track_weights
from .losses import standardize_depth, offset_regularization, mesh_edges, edge_rigidity
from .rig import fk, lbs, pose_local
from .rig import rotation_6d
from .progress import Progress


def tensors(rig, device):
    return {k: torch.as_tensor(v, device=device,
             dtype=torch.long if k in ("faces", "parents") else torch.float32)
            for k, v in rig.items()}


def objective_weights(config):
    mode = config.get("objective", "paper")
    if mode not in ("paper", "normalized"):
        raise ValueError("joints.objective must be paper or normalized")
    defaults = {"rotation_smooth": 0., "root_smooth": 0., "stationary": 0., "rigid": 0.}
    if mode == "normalized":
        defaults.update(rgb=1., mask=1., track=10., depth=1., smooth=10., regularization=1000.)
    return {**defaults, **config.get("loss_weights", {})}


def tracking_geometry(vertices, rig, target, renderer):
    triangles = rig["faces"][target["triangles"]]
    weights = target["barycentric"][..., None]
    points = (vertices[:, triangles]*weights).sum(-2)
    canonical = (rig["vertices"][triangles]*weights).sum(-2)
    projected = torch.stack([renderer.project(points_at_frame) for points_at_frame in points])
    depths = torch.stack([renderer.depth(points_at_frame) for points_at_frame in points])
    tracks = target["tracks"]
    h, w = renderer.size
    in_frame = ((tracks[..., 0] >= 0) & (tracks[..., 0] <= w-1)
                & (tracks[..., 1] >= 0) & (tracks[..., 1] <= h-1))
    return projected, depths, renderer.depth(canonical), in_frame


def video_loss(vertices, rig, target, renderer, progress=None, config=None, render_indices=None):
    if config and config.get("objective", "paper") == "normalized":
        return normalized_video_loss(vertices, rig, target, renderer, progress, config, render_indices)
    frames, masks = target["frames"], target["masks"]
    # Exclude only coordinates outside the image (no depth observation exists).
    # The paper does not prescribe a confidence-weighted or Huber objective.
    projected, depths, canonical_depth, valid = tracking_geometry(vertices, rig, target, renderer)
    track = frame_l2(projected, target["tracks"], valid)
    normalized_target = normalize_depth(target["depth"], canonical_depth, valid)
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


def normalized_video_loss(vertices, rig, target, renderer, progress, config, render_indices):
    """Normalized image and point losses with confidence-filtered supervision.

    Keep all frames for track/depth/smoothness; minibatch only RGB and mask.
    Our renderer and tracker both use integer pixel centers.
    """
    projected, depths, canonical_depth, valid = tracking_geometry(vertices, rig, target, renderer)
    h, w = renderer.size
    tracks = target["tracks"]
    # Threshold AllTracker channel 1 (our confidence), not channel0*channel1.
    visible = target["confidence"] > config.get("confidence_threshold", .1)
    visible = visible.clone()
    visible[0] = True
    valid = valid & visible
    if "exclude_tracks" in target:
        valid = valid & ~target["exclude_tracks"].bool()
    depth_target = standardize_depth(target["depth"], canonical_depth, valid)
    def masked_mean(error):
        return torch.where(valid, error, torch.zeros_like(error)).sum()/valid.sum().clamp_min(1)
    indices = range(len(vertices)) if render_indices is None else render_indices.tolist()
    rgb, masks = [], []
    for frame in indices:
        if progress:
            progress.detail(f"rendering sampled frame {frame+1}/{len(vertices)}")
        image, alpha = renderer(vertices[frame], rig["faces"], rig["colors"])
        # Composite input RGB over white using the SAM2 mask.
        mask = target["masks"][frame]
        truth = target["frames"][frame]*mask[..., None]+1-mask[..., None]
        rgb.append((image-truth).abs().mean())
        masks.append((alpha-mask).abs().mean())
    losses = {"rgb": torch.stack(rgb).mean(), "mask": torch.stack(masks).mean(),
            "track": masked_mean((projected-tracks).norm(dim=-1)/max(h, w)),
            "depth": masked_mean((depths-depth_target).abs()),
            "smooth": 3*smoothness(vertices)}
    if "anchor_xy" in target:
        faces = rig["faces"][target["anchor_triangles"]]
        surface = (vertices[:, faces]*target["anchor_barycentric"][None, :, :, None]).sum(-2)
        predicted = torch.stack([renderer.project(p) for p in surface])
        error = (predicted-target["anchor_xy"]).norm(dim=-1)/max(h, w)
        losses["anchor"] = error[target["anchor_valid"].bool()].mean()
    return losses


def fit_joints(rig, supervision, renderer, output, config):
    weights = objective_weights(config)
    normalized = config.get("objective", "paper") == "normalized"
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
                            if any(k in config for k in old_counts) else (3000 if normalized else 700))
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
    batch_frames = config.get("batch_frames", 4 if normalized else n)
    if not isinstance(batch_frames, int) or batch_frames < 1:
        raise ValueError("joints.batch_frames must be a positive integer")
    generator = torch.Generator().manual_seed(config.get("seed", 0))
    if weights["rigid"]:
        edges, rest_length = mesh_edges(rig["vertices"], rig["faces"])
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
        if normalized:
            pick = torch.randperm(n, generator=generator)[:batch_frames]
            losses = video_loss(vertices, rig, supervision, renderer, progress=progress,
                                config=config, render_indices=pick)
        else:
            losses = video_loss(vertices, rig, supervision, renderer, progress=progress)
        if weights["rigid"]:
            losses["rigid"] = edge_rigidity(vertices, edges, rest_length)
        matrices = rotation_6d(rotation)
        if weights["rotation_smooth"]:
            losses["rotation_smooth"] = temporal_acceleration(matrices, times)
        if weights["root_smooth"]:
            losses["root_smooth"] = temporal_acceleration(world[:, roots, :3, 3], times)
        if weights["stationary"]:
            points = (vertices[:, rig["faces"][supervision["triangles"]]]
                      * supervision["barycentric"][None, :, :, None]).sum(-2)
            losses["stationary"] = weighted_huber(points/.01, points[:1].expand_as(points)/.01, stationary)
        regularizer = offset_regularization if normalized else joint_regularization
        losses["regularization"] = regularizer(rotation, effective_translation, roots,
            config.get("rotation_bound", 1.), config.get("translation_bound", .01 if normalized else .1))
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
    from .evaluate import deformation_metrics
    save_json(output/"joint_diagnostics.json", deformation_metrics(vertices, rig, times))
    return ["joint_trajectory.npz", "joint_checkpoint.npz", "joint_losses.json", "joint_losses.csv",
            "joint_diagnostics.json"]
