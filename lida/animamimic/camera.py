"""Perspective camera in the PyTorch3D convention, implemented in plain torch.

Keeping projection independent of PyTorch3D lets the tracking and depth losses
run on any device. tests/test_core.py checks it against PyTorch3D's cameras.
"""
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Camera:
    R: torch.Tensor  # (3, 3), world-to-view is X_view = X_world @ R + T
    T: torch.Tensor  # (3,)
    fov_deg: float
    image_size: int

    @classmethod
    def look_at(cls, dist, elev_deg, azim_deg, fov_deg, image_size, at=(0.0, 0.0, 0.0), device="cpu"):
        elev, azim = math.radians(elev_deg), math.radians(azim_deg)
        at = torch.tensor(at, dtype=torch.float32, device=device)
        eye = at + dist * torch.tensor(
            [math.cos(elev) * math.sin(azim), math.sin(elev), math.cos(elev) * math.cos(azim)],
            dtype=torch.float32, device=device,
        )
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
        z = F.normalize(at - eye, dim=0)
        x = F.normalize(torch.cross(up, z, dim=0), dim=0)
        y = F.normalize(torch.cross(z, x, dim=0), dim=0)
        R = torch.stack((x, y, z)).T
        T = -(R.T @ eye)
        return cls(R, T, fov_deg, image_size)

    def to(self, device) -> "Camera":
        return Camera(self.R.to(device), self.T.to(device), self.fov_deg, self.image_size)

    def to_view(self, pts: torch.Tensor) -> torch.Tensor:
        return pts @ self.R + self.T

    def project(self, pts: torch.Tensor):
        """Return pixel coordinates (..., 2) and camera-space depth (...,)."""
        v = self.to_view(pts)
        s = 1.0 / math.tan(math.radians(self.fov_deg) / 2.0)
        ndc = s * v[..., :2] / v[..., 2:].clamp_min(1e-6)
        return (1.0 - ndc) * self.image_size / 2.0, v[..., 2]

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump({
                "R": self.R.cpu().tolist(), "T": self.T.cpu().tolist(),
                "fov_deg": self.fov_deg, "image_size": self.image_size,
            }, f, indent=2)

    @classmethod
    def from_json(cls, path, device="cpu"):
        with open(path) as f:
            d = json.load(f)
        return cls(
            torch.tensor(d["R"], dtype=torch.float32, device=device),
            torch.tensor(d["T"], dtype=torch.float32, device=device),
            d["fov_deg"], d["image_size"],
        )
