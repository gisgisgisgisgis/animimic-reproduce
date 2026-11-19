"""Normalized video objectives shared by both optimization stages."""
import torch
import torch.nn.functional as F


def weighted_huber(pred, target, weight):
    error = F.huber_loss(pred, target, reduction="none")
    while weight.ndim < error.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1)


def frame_l2(pred, target, valid=None):
    """Mean over frames of the Euclidean residual norm (Eqs. 5–8).

    L2 is an explicit implementation choice where the paper leaves the norm
    unspecified. A boolean validity mask only excludes missing observations.
    """
    error = pred-target
    if valid is not None:
        while valid.ndim < error.ndim:
            valid = valid.unsqueeze(-1)
        error = torch.where(valid, error, torch.zeros_like(error))
    return torch.linalg.vector_norm(error.flatten(1), dim=1).mean()


def joint_regularization(rotation, translation, roots, rotation_bound=1., translation_bound=.1):
    """Eq. 9: componentwise clipping of non-root 6D rotation/translation.

    Bounds apply to the learned rest-local parameters; values are project
    choices, since the paper provides no numerical thresholds.
    """
    parameters = torch.cat((rotation[:, ~roots], translation[:, ~roots]), dim=-1)
    bounds = parameters.new_tensor([rotation_bound]*6+[translation_bound]*3)
    if (bounds <= 0).any() or not torch.isfinite(bounds).all():
        raise ValueError("Joint clipping bounds must be finite and positive")
    return frame_l2(parameters, parameters.clamp(-bounds, bounds))


def smoothness(vertices):
    loss = vertices.sum() * 0
    if len(vertices) > 1:
        velocity = vertices[1:] - vertices[:-1]
        loss = loss + velocity.square().mean()
        if len(vertices) > 2:
            loss = loss + (velocity[1:] - velocity[:-1]).square().mean()
    return loss


def temporal_acceleration(value, times):
    """Acceleration in seconds, rather than differences per sampled frame."""
    if len(value) < 3:
        return value.sum()*0
    dt = (times[1:]-times[:-1]).reshape(-1, *([1]*(value.ndim-1)))
    velocity = (value[1:]-value[:-1])/dt
    acceleration = 2*(velocity[1:]-velocity[:-1])/(dt[1:]+dt[:-1])
    return acceleration.square().mean()


def stationary_track_weights(tracks, weight, threshold=2.):
    """Anchor only tracks stationary over the clip with reliable support."""
    reliable = weight > .5
    displacement = (tracks-tracks[:1]).norm(dim=-1)
    displacement = torch.where(reliable, displacement, torch.zeros_like(displacement))
    stable = (reliable[0] & (reliable.float().mean(0) >= .8)
              & (displacement.amax(0) <= threshold))
    return torch.where(reliable & stable[None], weight, torch.zeros_like(weight))


def normalize_depth(depth, reference, valid):
    """Instantiate target-only N(D) in Eq. 8; no pose-dependent normalization.

    Assumption (not specified by the paper): align each observed frame's
    median/MAD to the fixed canonical mesh's camera-depth median/MAD.
    This keeps the target in camera units rather than centering it at zero.
    """
    with torch.no_grad():
        result = torch.zeros_like(depth)
        reference = reference.detach()
        if not valid[0].any():
            raise ValueError("Depth normalization needs valid first-frame tracks")
        canonical = reference[valid[0]]
        center = canonical.median()
        spread = (canonical-center).abs().median()
        for i in range(len(depth)):
            selected = depth[i][valid[i]]
            if selected.numel():
                median = selected.median()
                mad = (selected-median).abs().median().clamp_min(1e-6)
                result[i] = (depth[i]-median)/mad*spread+center
        return result
