"""Normalized video objectives shared by both optimization stages."""
import torch
import torch.nn.functional as F


def weighted_huber(pred, target, weight):
    error = F.huber_loss(pred, target, reduction="none")
    while weight.ndim < error.ndim:
        weight = weight.unsqueeze(-1)
    weight = weight.expand_as(error)
    return (error * weight).sum() / weight.sum().clamp_min(1)


def masked_rgb(pred, target, mask):
    error = ((pred-target).square()+1e-6).sqrt()
    weight = mask[..., None].expand_as(error)
    return (error*weight).sum()/weight.sum().clamp_min(1)


def silhouette(pred, target):
    bce = F.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), target)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return bce + 1 - (intersection + 1e-6) / (union + 1e-6)


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


def normalize_depth(depth, valid):
    result = torch.zeros_like(depth)
    for i in range(len(depth)):
        selected = depth[i][valid[i]]
        if selected.numel():
            median = selected.median()
            mad = (selected - median).abs().median().clamp_min(1e-6)
            result[i] = (depth[i] - median) / mad
    return result
