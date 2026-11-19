"""Three-phase pose fitting and the shared image-space objective."""
import numpy as np
import torch

from .core import save_history
from .losses import normalize_depth, silhouette, smoothness, weighted_huber, masked_rgb
from .losses import temporal_acceleration, stationary_track_weights
from .rig import fk, lbs, pose_local
from .rig import rotation_6d
from .progress import Progress


def tensors(rig, device):
    return {k: torch.as_tensor(v, device=device,
             dtype=torch.long if k in ("faces", "parents") else torch.float32)
            for k, v in rig.items()}


def video_loss(vertices, rig, target, renderer, phase="full", progress=None):
    frames, masks = target["frames"], target["masks"]
    tri, beta = target["triangles"], target["barycentric"]
    points = (vertices[:, rig["faces"][tri]] * beta[None, :, :, None]).sum(-2)
    projected = torch.stack([renderer.project(p) for p in points])
    scale = max(renderer.size)
    weight = target["visibility"] * target["confidence"]
    track = weighted_huber(projected/scale, target["tracks"]/scale, weight)
    depths = torch.stack([renderer.depth(p) for p in points])
    valid = weight > 0.1
    depth = weighted_huber(normalize_depth(depths, valid),
                           normalize_depth(target["depth"], valid), weight)
    rgb, mask = [], []
    for frame, value in enumerate(vertices, 1):
        if progress:
            progress.detail(f"rendering frame {frame}/{len(vertices)}")
        image, alpha = renderer(value, rig["faces"], rig["colors"])
        rgb.append(image)
        mask.append(alpha)
    rgb, mask = torch.stack(rgb), torch.stack(mask)
    losses = {"mask": silhouette(mask, masks), "track": track, "depth": depth,
              "smooth": smoothness(vertices)}
    if phase == "full":
        losses["rgb"] = masked_rgb(rgb, frames, masks)
    return losses


def fit_joints(rig, supervision, renderer, output, config):
    device = rig["vertices"].device
    n = len(supervision["frames"])
    j = len(rig["parents"])
    rotation = torch.tensor([1., 0, 0, 0, 1, 0], device=device).repeat(n, j, 1)
    rotation.requires_grad_()
    translation = torch.zeros(n, j, 3, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([rotation, translation], lr=config.get("learning_rate", 1e-3))
    phases = [("root", config.get("root_iterations", 100)),
              ("geometry", config.get("geometry_iterations", 300)),
              ("full", config.get("full_iterations", 300))]
    history = []
    parents = rig["parents"].tolist()
    # Root phase can change all scene-root transforms, including an inserted scene ancestor.
    roots = rig["parents"] < 0
    # Only scene roots may translate by default; child offsets define bone lengths.
    movable = torch.ones_like(roots) if config.get("optimize_joint_translations", False) else roots
    times = supervision.get("frame_times", torch.arange(n, device=device)/24)
    if len(times) != n or not torch.isfinite(times).all() or (times[1:] <= times[:-1]).any():
        raise ValueError("Joint frame times must be finite and strictly increasing")
    stationary = stationary_track_weights(supervision["tracks"],
        supervision["visibility"]*supervision["confidence"], config.get("stationary_track_px", 2.))
    weights = {"rotation_smooth": .01, "root_smooth": .1, "stationary": 1.,
               **config.get("loss_weights", {})}
    lengths = rig["local"][:, :3, 3].norm(dim=-1).clamp_min(1e-4)
    translation_bound = lengths*config.get("max_translation_bone_fraction", .1)
    for phase_index, (phase, iterations) in enumerate(phases, 1):
        progress = Progress(f"joints {phase_index}/3 {phase}", iterations)
        for iteration in range(iterations):
            optimizer.zero_grad()
            effective_translation = translation*movable[None, :, None]
            local = pose_local(rig["local"], rotation, effective_translation)
            world = fk(local, parents)
            vertices = lbs(rig["vertices"], rig["weights"], world, rig["inverse_bind"])
            losses = video_loss(vertices, rig, supervision, renderer, phase, progress)
            matrices = rotation_6d(rotation)
            losses["rotation_smooth"] = temporal_acceleration(matrices, times)
            losses["root_smooth"] = temporal_acceleration(world[:, roots, :3, 3], times)
            points = (vertices[:, rig["faces"][supervision["triangles"]]]
                      * supervision["barycentric"][None, :, :, None]).sum(-2)
            # Reliable stationary 2D points disambiguate unwanted depth motion.
            # Moving or occluded tracks do not impose a planted-foot assumption.
            losses["stationary"] = weighted_huber(points/.01, points[:1].expand_as(points)/.01, stationary)
            losses["regularization"] = (
                (translation[:, ~roots].abs()-translation_bound[None, ~roots, None]).clamp_min(0).square().mean()
                if (~roots).any() else translation.sum()*0)
            if (~roots).any():
                cosine = ((matrices[:, ~roots].diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1+1e-6, 1-1e-6)
                angle = torch.acos(cosine)
                bound = config.get("max_rotation_deg", 75)*np.pi/180
                losses["regularization"] = losses["regularization"] + (angle-bound).clamp_min(0).square().mean()
            loss = sum(weights.get(k, 1)*v for k, v in losses.items())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite joint objective")
            progress.detail("backward pass")
            loss.backward()
            if phase == "root":
                rotation.grad[:, ~roots] = 0
                translation.grad[:, ~roots] = 0
            torch.nn.utils.clip_grad_norm_([rotation, translation], 1, error_if_nonfinite=True)
            optimizer.step()
            history.append({"phase": phase, "iteration": iteration,
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
