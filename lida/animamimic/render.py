"""Differentiable rendering.

Default on CUDA and CPU: PyTorch3D SoftPhong + silhouette, as in the paper.
Default on MPS: RasterRenderer, the same shading and coverage formulas in plain torch.
SoftSilhouetteRenderer is an older silhouette-only option, kept for comparison.
"""
import math

import torch

from .camera import Camera
from .gltf_rig import MeshData


class PyTorch3DRenderer:
    has_rgb = True

    def __init__(self, mesh: MeshData, camera: Camera, device, sigma=1e-4, gamma=1e-4, faces_per_pixel=10):
        from pytorch3d.renderer import (
            BlendParams, FoVPerspectiveCameras, MeshRasterizer, PointLights,
            RasterizationSettings, SoftPhongShader, SoftSilhouetteShader, TexturesUV,
        )
        from pytorch3d.structures import Meshes

        self.device = device
        verts, faces = mesh.verts.to(device), mesh.faces.to(device)
        textures = TexturesUV(
            maps=mesh.texture.to(device)[None], faces_uvs=faces[None], verts_uvs=mesh.uv.to(device)[None],
        )
        self.base = Meshes(verts=[verts], faces=[faces], textures=textures)
        self.cameras = FoVPerspectiveCameras(
            device=device, R=camera.R.to(device)[None], T=camera.T.to(device)[None], fov=camera.fov_deg,
        )
        eye = (-camera.R.to(device) @ camera.T.to(device))[None]
        lights = PointLights(
            device=device, location=eye,
            ambient_color=((0.75, 0.75, 0.75),), diffuse_color=((0.25, 0.25, 0.25),), specular_color=((0.0, 0.0, 0.0),),
        )
        blend = BlendParams(sigma=sigma, gamma=gamma, background_color=(1.0, 1.0, 1.0))
        self.rasterizer = MeshRasterizer(self.cameras, RasterizationSettings(
            image_size=camera.image_size,
            blur_radius=math.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=faces_per_pixel,
            perspective_correct=False,
        ))
        self.phong = SoftPhongShader(device=device, cameras=self.cameras, lights=lights, blend_params=blend)
        self.silhouette = SoftSilhouetteShader(blend_params=blend)

    def __call__(self, verts: torch.Tensor):
        """verts: (B, V, 3). Returns rgb (B, H, W, 3) and mask (B, H, W)."""
        meshes = self.base.extend(verts.shape[0]).update_padded(verts)
        fragments = self.rasterizer(meshes)
        rgb = self.phong(fragments, meshes)[..., :3]
        mask = self.silhouette(fragments, meshes)[..., 3]
        return rgb, mask


class SoftSilhouetteRenderer:
    """SoftRas-style coverage in plain torch. Costs O(faces x pixels), so low-poly and low-res only."""

    has_rgb = False

    def __init__(self, mesh: MeshData, camera: Camera, device, sigma_px=1.0, chunk=256):
        self.faces = mesh.faces.to(device)
        self.camera = camera.to(device)
        self.sigma, self.chunk = sigma_px, chunk
        n = camera.image_size
        ys, xs = torch.meshgrid(
            torch.arange(n, device=device, dtype=torch.float32) + 0.5,
            torch.arange(n, device=device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        self.pixels = torch.stack((xs, ys), -1).reshape(-1, 2)
        self.n = n

    def __call__(self, verts: torch.Tensor):
        xy, _ = self.camera.project(verts)
        return None, torch.stack([self._one(p) for p in xy])

    def _one(self, xy):
        tri = xy[self.faces]  # (F, 3, 2)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        sign = torch.sign(area.detach())
        # Degenerate faces are zeroed rather than filtered: a tensor shape that changes
        # every iteration forces MPS to recompile its graph, which is extremely slow.
        keep = (torch.abs(area.detach()) > 1e-6).to(xy.dtype)
        log_miss = torch.zeros(self.pixels.shape[0], device=self.pixels.device, dtype=xy.dtype)
        for s in range(0, tri.shape[0], self.chunk):
            k = keep[s:s + self.chunk]
            t = tri[s:s + self.chunk]
            sg = sign[s:s + self.chunk]
            p0 = t
            p1 = torch.roll(t, -1, dims=1)
            edge = p1 - p0
            length = edge.norm(dim=-1).clamp_min(1e-6)
            d = self.pixels[None, None, :, :] - p0[:, :, None, :]
            signed = (edge[:, :, None, 0] * d[..., 1] - edge[:, :, None, 1] * d[..., 0]) / length[:, :, None]
            inside = (signed * sg[:, None, None]).min(dim=1).values
            cover = (torch.sigmoid(inside / self.sigma) * k[:, None]).clamp(max=1.0 - 1e-6)
            log_miss = log_miss + torch.log1p(-cover).sum(0)
        return (1.0 - torch.exp(log_miss)).reshape(self.n, self.n)


class RasterRenderer:
    """SoftPhong + silhouette in plain torch, so it also runs on Apple GPUs (MPS).

    PyTorch3D's rasterizer has no MPS path. This one uses the same coverage formula
    (sigmoid of the signed squared NDC distance to the triangle) and the same Phong
    lighting. Two things differ: only pixels near each triangle's bounding box are
    visited, and color is composited with the soft alpha, so edges do not bleed.

    Choosing which pixels each triangle touches is discrete, so it runs without
    gradients on `enum_device`. Everything differentiable runs on `device`. The
    candidate list is padded to a fixed capacity, because a tensor shape that changes
    every iteration forces MPS to recompile its graph.
    """

    has_rgb = True

    def __init__(self, mesh: MeshData, camera: Camera, device, sigma=1e-4, gamma=1e-4, znear=1.0, zfar=100.0):
        self.device = device
        self.camera = camera.to(device)
        self.enum_camera = camera.to("cpu") if device.type == "mps" else self.camera
        self.enum_device = self.enum_camera.R.device
        self.faces = mesh.faces.to(device)
        self.enum_faces = mesh.faces.to(self.enum_device)
        self.uv = mesh.uv.to(device)
        # PyTorch3D's TexturesUV has v pointing up, so the image rows are flipped before sampling.
        self.texture = torch.flip(mesh.texture.to(device), [0])
        # glTF textures repeat by default, and some assets rely on UVs outside [0, 1].
        self.wrap_uv = bool(mesh.uv.min() < 0.0 or mesh.uv.max() > 1.0)
        self.sigma, self.gamma, self.znear, self.zfar = sigma, gamma, znear, zfar
        self.n = camera.image_size
        self.max_sq_ndc = math.log(1.0 / 1e-4 - 1.0) * sigma
        self.margin_px = math.sqrt(self.max_sq_ndc) * self.n / 2.0
        self.eye = (-self.camera.R @ self.camera.T)
        self.capacity = 0

    def _signed_sq_dist(self, tri, p):
        """Signed squared distance in NDC from points p (N, 2) to triangles tri (N, 3, 2), in pixels on input.

        Returns the distance and screen-space barycentrics. Negative means inside.
        """
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        cross = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        area = cross(b - a, c - a)
        area = area + (area.detach().abs() < 1e-9).to(area.dtype) * 1e-9
        w0, w1, w2 = cross(c - b, p - b) / area, cross(a - c, p - c) / area, cross(b - a, p - a) / area
        bary = torch.stack((w0, w1, w2), -1)

        def seg(u, v):
            e = v - u
            s = (((p - u) * e).sum(-1) / (e * e).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
            d = p - (u + s[:, None] * e)
            return (d * d).sum(-1)

        sq_px = torch.stack((seg(a, b), seg(b, c), seg(c, a)), -1).amin(-1)
        inside = (bary.detach() >= 0.0).all(-1)
        scale = (2.0 / self.n) ** 2
        return (sq_px * scale) * (1.0 - 2.0 * inside.to(sq_px.dtype)), bary

    @torch.no_grad()
    def _candidates(self, verts):
        """List the (frame, face, pixel) triples that can receive coverage, and each pixel's nearest depth."""
        B, F = verts.shape[0], self.enum_faces.shape[0]
        xy, z = self.enum_camera.project(verts.to(self.enum_device))
        tri = xy[:, self.enum_faces].reshape(B * F, 3, 2)
        tri_z = z[:, self.enum_faces].reshape(B * F, 3)
        lo = (tri.amin(1) - self.margin_px - 0.5).ceil().long().clamp(0, self.n)
        hi = (tri.amax(1) + self.margin_px - 0.5).floor().long().clamp(-1, self.n - 1)
        w = (hi[:, 0] - lo[:, 0] + 1).clamp_min(0)
        h = (hi[:, 1] - lo[:, 1] + 1).clamp_min(0)
        count = w * h * (tri_z.amin(1) > 1e-4).long()

        fid = torch.repeat_interleave(torch.arange(B * F, device=self.enum_device), count)
        start = torch.cat((torch.zeros(1, dtype=torch.long, device=self.enum_device), count.cumsum(0)[:-1]))
        local = torch.arange(fid.shape[0], device=self.enum_device) - start[fid]
        px = lo[fid, 0] + local % w[fid]
        py = lo[fid, 1] + local // w[fid]
        p = torch.stack((px, py), -1).to(tri.dtype) + 0.5
        d, bary = self._signed_sq_dist(tri[fid], p)
        keep = d < self.max_sq_ndc
        fid, px, py, bary = fid[keep], px[keep], py[keep], bary[keep]

        pid = (torch.div(fid, F, rounding_mode="floor") * self.n + py) * self.n + px
        bary = bary.clamp_min(0.0)
        bary = bary / bary.sum(-1, keepdim=True).clamp_min(1e-9)
        z_cand = (tri_z[fid] * bary).sum(-1)
        z_inv = (self.zfar - z_cand) / (self.zfar - self.znear)
        z_max = torch.zeros(B * self.n * self.n, device=self.enum_device, dtype=z_inv.dtype)
        z_max = z_max.scatter_reduce(0, pid, z_inv, "amax", include_self=True)
        return fid, pid, z_max[pid]

    def _pad(self, fid, pid, z_max):
        n = fid.shape[0]
        if n > self.capacity:
            self.capacity = int(n * 1.25) + 1024
        pad = self.capacity - n
        valid = torch.cat((torch.ones(n, device=self.enum_device), torch.zeros(pad, device=self.enum_device)))
        zeros = torch.zeros(pad, dtype=torch.long, device=self.enum_device)
        return (
            torch.cat((fid, zeros)).to(self.device), torch.cat((pid, zeros)).to(self.device),
            torch.cat((z_max, zeros.to(z_max.dtype))).to(self.device), valid.to(self.device),
        )

    def _sample_texture(self, uv):
        """Bilinear lookup. Written by hand because MPS has no backward pass for grid_sample."""
        H, W = self.texture.shape[:2]
        x, y = uv[:, 0] * (W - 1), uv[:, 1] * (H - 1)
        x0 = x.detach().floor().long().clamp(0, W - 2)
        y0 = y.detach().floor().long().clamp(0, H - 2)
        fx, fy = (x - x0.to(x.dtype))[:, None], (y - y0.to(y.dtype))[:, None]
        tex = self.texture
        top = tex[y0, x0] * (1.0 - fx) + tex[y0, x0 + 1] * fx
        bottom = tex[y0 + 1, x0] * (1.0 - fx) + tex[y0 + 1, x0 + 1] * fx
        return top * (1.0 - fy) + bottom * fy

    def _vertex_normals(self, verts):
        V = verts.shape[1]
        tri = verts[:, self.faces]
        fn = torch.linalg.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
        out = []
        for b in range(verts.shape[0]):
            n = torch.zeros(V, 3, device=verts.device, dtype=verts.dtype)
            for k in range(3):
                n = n.index_add(0, self.faces[:, k], fn[b])
            out.append(n)
        return torch.nn.functional.normalize(torch.stack(out), dim=-1, eps=1e-8)

    def __call__(self, verts: torch.Tensor):
        """verts: (B, V, 3). Returns rgb (B, H, W, 3) and mask (B, H, W)."""
        B, F, n = verts.shape[0], self.faces.shape[0], self.n
        fid, pid, z_max, valid = self._pad(*self._candidates(verts))
        batch = torch.div(fid, F, rounding_mode="floor")
        face = self.faces[fid % F]

        xy, z = self.camera.project(verts)
        tri = xy[batch[:, None], face]
        tri_z = z[batch[:, None], face]
        ij = torch.stack((pid % n, torch.div(pid, n, rounding_mode="floor") % n), -1).to(tri.dtype) + 0.5
        dist, bary = self._signed_sq_dist(tri, ij)
        # Guard before the sigmoid: padded rows hold arbitrary geometry.
        prob = torch.sigmoid(-dist / self.sigma) * valid
        bary = bary.clamp_min(0.0)
        bary = bary / bary.sum(-1, keepdim=True).clamp_min(1e-9)

        P = B * n * n
        log_miss = torch.zeros(P, device=self.device, dtype=tri.dtype).index_add(
            0, pid, torch.log1p(-prob.clamp(max=1.0 - 1e-6)),
        )
        alpha = 1.0 - torch.exp(log_miss)

        # Softmax over depth, as in PyTorch3D's softmax_rgb_blend.
        z_inv = (self.zfar - (tri_z * bary).sum(-1)) / (self.zfar - self.znear)
        weight = prob * torch.exp((z_inv - z_max).clamp_max(0.0) / self.gamma)
        total = torch.zeros(P, device=self.device, dtype=tri.dtype).index_add(0, pid, weight)

        pos = (verts[batch[:, None], face] * bary[..., None]).sum(1)
        normals = self._vertex_normals(verts)
        normal = torch.nn.functional.normalize((normals[batch[:, None], face] * bary[..., None]).sum(1), dim=-1, eps=1e-8)
        light = torch.nn.functional.normalize(self.eye[None] - pos, dim=-1, eps=1e-8)
        lambert = (normal * light).sum(-1).clamp_min(0.0)
        uv = (self.uv[face] * bary[..., None]).sum(1)
        uv = uv - uv.detach().floor() if self.wrap_uv else uv.clamp(0.0, 1.0)
        color = self._sample_texture(uv) * (0.75 + 0.25 * lambert)[:, None]

        rgb = torch.zeros(P, 3, device=self.device, dtype=tri.dtype).index_add(0, pid, color * weight[:, None])
        rgb = rgb / total[:, None].clamp_min(1e-8)
        rgb = alpha[:, None] * rgb + (1.0 - alpha[:, None])  # white background
        return rgb.reshape(B, n, n, 3), alpha.reshape(B, n, n)


def build_renderer(mesh: MeshData, camera: Camera, device, backend: str = "auto", sigma: float = 1e-4):
    """sigma: edge softness of the soft rasterizer. 1e-4 (PyTorch3D's usual value) spreads each edge over
    about 4 px at 256 px; smaller is sharper but gives the mask loss gradients only near the edges."""
    if backend == "auto":
        backend = "raster" if device.type == "mps" else "pytorch3d"
    if backend == "pytorch3d":
        return PyTorch3DRenderer(mesh, camera, device, sigma=sigma)
    if backend == "raster":
        return RasterRenderer(mesh, camera, device, sigma=sigma)
    if backend == "soft":
        return SoftSilhouetteRenderer(mesh, camera, device)
    raise ValueError(backend)


def render_in_chunks(renderer, verts: torch.Tensor, chunk: int = 4):
    """No-grad rendering of a whole sequence, for targets and previews."""
    rgb, mask = [], []
    with torch.no_grad():
        for s in range(0, verts.shape[0], chunk):
            r, m = renderer(verts[s:s + chunk])
            mask.append(m)
            if r is not None:
                rgb.append(r)
    return (torch.cat(rgb) if rgb else None), torch.cat(mask)
