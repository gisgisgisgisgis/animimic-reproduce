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


def normalize_depth(depth, valid):
    result = torch.zeros_like(depth)
    for i in range(len(depth)):
        selected = depth[i][valid[i]]
        if selected.numel():
            median = selected.median()
            mad = (selected - median).abs().median().clamp_min(1e-6)
            result[i] = (depth[i] - median) / mad
    return result
