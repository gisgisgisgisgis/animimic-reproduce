"""Differentiable skeletal motion; matrices use column-vector convention."""
import torch
import torch.nn.functional as F


def rotation_6d(value):
    a, b = value[..., :3], value[..., 3:]
    x = F.normalize(a, dim=-1)
    y = F.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
    return torch.stack((x, y, torch.linalg.cross(x, y)), dim=-1)


def fk(local, parents):
    world = []
    for i, parent in enumerate(parents):
        if parent >= i:
            raise ValueError("Skeleton must be topologically ordered")
        world.append(local[..., i, :, :] if parent < 0
                     else world[parent] @ local[..., i, :, :])
    return torch.stack(world, dim=-3)


def lbs(vertices, weights, world, inverse_bind):
    homogeneous = F.pad(vertices, (0, 1), value=1)
    transformed = torch.einsum("...jab,vb->...vja", world @ inverse_bind, homogeneous)
    return torch.einsum("vj,...vjk->...vk", weights, transformed[..., :3])


def pose_local(rest, rotation, translation):
    delta = torch.eye(4, device=rotation.device, dtype=rotation.dtype).expand(
        *rotation.shape[:-1], 4, 4).clone()
    delta[..., :3, :3] = rotation_6d(rotation)
    delta[..., :3, 3] = translation
    return rest @ delta
