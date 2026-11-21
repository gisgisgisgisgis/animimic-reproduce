"""On-disk format of one input sequence. See README for how each file is produced.

<seq>/rigged.glb    mesh + skeleton + skin weights (UniRig merge output)
<seq>/camera.json   the camera used to render the reference view
<seq>/frames/*.png  reference video
<seq>/masks/*.png   SAM2 foreground masks
<seq>/tracks.npz    xy (T, N, 2) in pixels; visible (T, N)
<seq>/depth.npz     depth (T, H, W), monocular depth from VGGT
"""
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from .camera import Camera
from .gltf_rig import MeshData, RiggedGLB
from .losses import TrackTargets, lift_query_points
from .rig import Rig, smooth_skin_weights


@dataclass
class Sequence:
    mesh: MeshData
    rig: Rig
    camera: Camera
    rgb: torch.Tensor  # (T, H, W, 3), composited over white
    mask: torch.Tensor  # (T, H, W)
    tracks: TrackTargets

    @property
    def num_frames(self) -> int:
        return self.mask.shape[0]


def build_tracks(mesh, camera, xy, visible, depth, device) -> TrackTargets:
    """Lift frame-0 pixels onto the rest mesh and drop those that miss it."""
    keep, face, bary = lift_query_points(mesh.verts, mesh.faces, camera, xy[0])
    if keep.sum() == 0:
        raise ValueError("No tracked point hits the mesh; check that camera.json matches the frames.")
    return TrackTargets(
        face, bary, xy[:, keep], visible[:, keep], None if depth is None else depth[:, keep],
    )


def _load_images(folder, size, device):
    paths = sorted(Path(folder).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG files in {folder}")
    frames = [torch.from_numpy(np.asarray(iio.imread(p))).float() / 255.0 for p in paths]
    x = torch.stack([f[..., None] if f.ndim == 2 else f for f in frames]).permute(0, 3, 1, 2)
    return F.interpolate(x, size=(size, size), mode="area").to(device)


def load_sequence(seq_dir, device, image_size=None, skin_smooth=0) -> Sequence:
    """skin_smooth: Laplacian smoothing steps on the skin weights; 0 keeps UniRig's weights."""
    seq_dir = Path(seq_dir)
    glb = RiggedGLB(seq_dir / "rigged.glb")
    mesh, rig = glb.mesh().to(device), glb.rig().to(device)
    rig = smooth_skin_weights(rig, mesh.verts, mesh.faces, skin_smooth)
    camera = Camera.from_json(seq_dir / "camera.json", device)
    source_size = camera.image_size
    if image_size:
        camera.image_size = image_size
    size = camera.image_size

    rgb = _load_images(seq_dir / "frames", size, device)[:, :3].permute(0, 2, 3, 1)
    mask = _load_images(seq_dir / "masks", size, device)[:, :1, ...].squeeze(1)
    rgb = rgb * mask[..., None] + (1.0 - mask[..., None])

    tr = np.load(seq_dir / "tracks.npz")
    scale = size / source_size
    xy = torch.from_numpy(tr["xy"]).float().to(device) * scale
    visible = torch.from_numpy(tr["visible"]).bool().to(device)

    d = torch.from_numpy(np.load(seq_dir / "depth.npz")["depth"]).float().to(device)
    grid = xy / size * 2.0 - 1.0
    raw = F.grid_sample(d[:, None], grid[:, None], align_corners=False, mode="bilinear")[:, 0, 0]

    return Sequence(mesh, rig, camera, rgb, mask, build_tracks(mesh, camera, xy, visible, raw, device))
