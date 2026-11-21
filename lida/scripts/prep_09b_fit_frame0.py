"""Place a reconstructed mesh so the camera sees it as in frame 0, and texture it by projecting frame 0.

Run in the `animamimic` env after prep_09a_frame0_to_3d.py:

  python scripts/prep_09b_fit_frame0.py --seq data/paper_dog

The mesh stays upright: only its yaw is fitted. The camera gets elevation, distance and look-at point.
1. Coarse: every yaw x elevation on a grid; distance and look-at are solved so the projected bounding
   box matches the mask's, then the silhouette IoU of projected surface samples is scored.
2. Fine: the best candidates are refined against the mask with the differentiable rasterizer.

Texture: UV = each vertex's pixel position in frame 0, texture = frame 0. Hidden surfaces get the
color of whatever covers them in frame 0, which is wrong but close for one-colored animals.
Writes <seq>/input.glb (normalized: centered, longest side 1), <seq>/camera.json, <seq>/fit_frame0.png.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animamimic.camera import Camera  # noqa: E402
from animamimic.device import pick_device  # noqa: E402
from animamimic.gltf_rig import MeshData  # noqa: E402
from animamimic.render import RasterRenderer  # noqa: E402


def look_at_torch(dist, elev, azim, at):
    """Differentiable Camera.look_at; angles in radians. Returns R (3, 3), T (3,)."""
    eye = at + dist * torch.stack([torch.cos(elev) * torch.sin(azim), torch.sin(elev), torch.cos(elev) * torch.cos(azim)])
    up = torch.tensor([0.0, 1.0, 0.0], dtype=at.dtype, device=at.device)
    z = F.normalize(at - eye, dim=0)
    x = F.normalize(torch.linalg.cross(up, z), dim=0)
    y = F.normalize(torch.linalg.cross(z, x), dim=0)
    R = torch.stack((x, y, z)).T
    return R, -(R.T @ eye)


def yaw_matrix(yaw):
    c, s = torch.cos(yaw), torch.sin(yaw)
    o, z = torch.ones_like(yaw), torch.zeros_like(yaw)
    return torch.stack([torch.stack([c, z, s]), torch.stack([z, o, z]), torch.stack([-s, z, c])])


def project(pts, R, T, fov, size):
    v = pts @ R + T
    s = 1.0 / math.tan(math.radians(fov) / 2.0)
    return (1.0 - s * v[:, :2] / v[:, 2:].clamp_min(1e-6)) * size / 2.0


def bbox(xy):
    return torch.cat([xy.min(0).values, xy.max(0).values])


def solve_framing(pts, elev, target_box, fov, size, iters=150):
    """Distance and look-at point that make the projected bounding box match the target's."""
    log_dist = torch.tensor(math.log(2.2), requires_grad=True)
    at = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([log_dist, at], lr=0.03)
    azim = torch.tensor(0.0)
    for _ in range(iters):
        R, T = look_at_torch(log_dist.exp(), elev, azim, torch.cat([at, torch.zeros(1)]))
        loss = ((bbox(project(pts, R, T, fov, size)) - target_box) / size).square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return log_dist.detach(), at.detach(), float(loss)


def splat_iou(pts, R, T, fov, size, target):
    xy = project(pts, R, T, fov, size).round().long()
    ok = (xy >= 0).all(1) & (xy < size).all(1)
    img = np.zeros((size, size), bool)
    img[xy[ok, 1].numpy(), xy[ok, 0].numpy()] = True
    img = ndimage.binary_fill_holes(ndimage.binary_closing(img, iterations=2))
    return (img & target).sum() / max((img | target).sum(), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--fov", type=float, default=40.0)
    ap.add_argument("--image_size", type=int, default=512, help="camera.json resolution, as in prep_01")
    ap.add_argument("--fit_size", type=int, default=256)
    ap.add_argument("--refine_iters", type=int, default=400)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fit_fov", action="store_true",
                    help="also fit the field of view; --fov is then the starting value. A wrong fov shows as the near "
                         "part of the animal being too small in the render and the far part too large")
    args = ap.parse_args()

    seq = Path(args.seq)
    device = pick_device(args.device)
    frame = np.asarray(Image.open(seq / "frame0.png").convert("RGB"))
    H = frame.shape[0]
    mask_full = np.asarray(Image.open(seq / "frame0_mask.png").convert("L")) > 127

    tm = trimesh.load(seq / "shape.obj", force="mesh", process=False)
    v = np.asarray(tm.vertices, np.float64)
    v = (v - (v.min(0) + v.max(0)) / 2) / (v.max(0) - v.min(0)).max()
    tm.vertices = v
    verts = torch.from_numpy(v).float()
    faces = torch.from_numpy(np.asarray(tm.faces)).long()
    samples = torch.from_numpy(np.asarray(trimesh.sample.sample_surface(tm, 60000, seed=0)[0])).float()

    S = args.fit_size
    target = np.asarray(Image.fromarray(mask_full).resize((S, S), Image.NEAREST))
    ys, xs = np.nonzero(target)
    target_box = torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=torch.float32)

    # 1. Coarse grid.
    coarse = []
    for yaw_deg in range(0, 360, 15):
        pts = samples @ yaw_matrix(torch.tensor(math.radians(yaw_deg))).T
        for elev_deg in (-10, 0, 10, 20, 30, 45):
            elev = torch.tensor(math.radians(elev_deg))
            log_dist, at, box_err = solve_framing(pts, elev, target_box, args.fov, S)
            R, T = look_at_torch(log_dist.exp(), elev, torch.tensor(0.0), torch.cat([at, torch.zeros(1)]))
            iou = splat_iou(pts, R, T, args.fov, S, target)
            coarse.append((iou, yaw_deg, elev_deg, log_dist, at))
    coarse.sort(key=lambda c: -c[0])
    for c in coarse[:8]:
        print(f"coarse IoU={c[0]:.3f} yaw={c[1]} elev={c[2]}")

    # 2. Refine with the differentiable rasterizer.
    dummy = MeshData(verts, faces, torch.zeros(len(verts), 2), torch.ones(4, 4, 3))
    tgt = torch.from_numpy(target.astype(np.float32)).to(device)
    best = None
    for iou0, yaw_deg, elev_deg, log_dist, at in coarse[:args.top]:
        params = {
            "log_k": torch.zeros((), device=device, requires_grad=args.fit_fov),  # image-plane zoom = fov change
            "yaw": torch.tensor(math.radians(yaw_deg), device=device, requires_grad=True),
            "elev": torch.tensor(math.radians(elev_deg), device=device, requires_grad=True),
            "log_dist": log_dist.clone().to(device).requires_grad_(True),
            "at": torch.cat([at, torch.zeros(1)]).to(device).requires_grad_(True),
        }
        opt = torch.optim.Adam([p for p in params.values() if p.requires_grad], lr=0.01)
        vd = verts.to(device)
        for it in range(args.refine_iters):
            R, T = look_at_torch(params["log_dist"].exp(), params["elev"], torch.zeros((), device=device), params["at"])
            k = torch.stack([params["log_k"].exp(), params["log_k"].exp(), torch.ones((), device=device)])
            renderer = RasterRenderer(dummy, Camera(R * k, T * k, args.fov, S), device)
            _, m = renderer((vd @ yaw_matrix(params["yaw"]).T)[None])
            loss = (m[0] - tgt).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            hard = (m[0] > 0.5).cpu().numpy()
            iou = (hard & target).sum() / max((hard | target).sum(), 1)
        fov_k = math.degrees(2 * math.atan(math.tan(math.radians(args.fov) / 2) / float(params["log_k"].exp())))
        print(f"refined yaw={math.degrees(params['yaw']):.1f} elev={math.degrees(params['elev']):.1f} "
              f"dist={float(params['log_dist'].exp()):.3f} fov={fov_k:.1f} IoU {iou0:.3f} -> {iou:.3f}")
        if best is None or iou > best[0]:
            best = (iou, {k: p.detach().cpu() for k, p in params.items()})

    iou, p = best
    # Zooming the image plane by k is the same as a camera with tan(fov/2) divided by k.
    fov = math.degrees(2 * math.atan(math.tan(math.radians(args.fov) / 2) / float(p["log_k"].exp())))
    v_world = (verts @ yaw_matrix(p["yaw"]).T).numpy()
    R, T = look_at_torch(p["log_dist"].exp(), p["elev"], torch.tensor(0.0), p["at"])
    # UniRig re-centers the bounding box and scales the longest side to 1. Do it here so the rigged
    # mesh stays in this frame: X_view = (X/s + c) @ R + T, and dividing by 1/s leaves the projection
    # unchanged, so T becomes (c @ R + T) * s.
    c = (v_world.min(0) + v_world.max(0)) / 2
    s = 1.0 / (v_world.max(0) - v_world.min(0)).max()
    v_world = ((v_world - c) * s).astype(np.float32)
    T = (torch.from_numpy(c).float() @ R + T) * s
    camera = Camera(R, T, fov, args.image_size)
    camera.to_json(seq / "camera.json")

    # Texture by projection into frame 0 (any resolution: UV is relative).
    cam_full = Camera(R, T, fov, 1)
    uv_px, _ = cam_full.project(torch.from_numpy(v_world).float())
    uv = uv_px.numpy().astype(np.float64)
    uv[:, 1] = 1.0 - uv[:, 1]  # v points up (OBJ / trimesh convention)
    # Where the mesh pokes out past the video's silhouette, UVs land on background. Fill the
    # background with the nearest foreground color so those rims are not painted white.
    inner = ndimage.binary_erosion(mask_full, iterations=max(2, H // 256))
    _, (iy, ix) = ndimage.distance_transform_edt(~inner, return_indices=True)
    tex = frame[iy, ix]
    out = trimesh.Trimesh(v_world, tm.faces, process=False,
                          visual=trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(tex)))
    out.export(seq / "input.glb")

    # Check image: frame 0, fitted render, overlap.
    dev_mesh = MeshData(torch.from_numpy(v_world).float(), faces, torch.from_numpy(uv).float(),
                        torch.from_numpy(tex.astype(np.float32) / 255.0))
    rgb, m = RasterRenderer(dev_mesh, Camera(R, T, fov, 512), device)(torch.from_numpy(v_world).float()[None].to(device))
    f512 = np.asarray(Image.fromarray(frame).resize((512, 512), Image.LANCZOS))
    t512 = np.asarray(Image.fromarray(mask_full).resize((512, 512), Image.NEAREST))
    over = np.full((512, 512, 3), 255, np.uint8)
    hard = (m[0] > 0.5).cpu().numpy()
    over[t512 & ~hard] = (220, 60, 60)
    over[hard & ~t512] = (60, 60, 220)
    over[hard & t512] = (150, 150, 150)
    sheet = np.concatenate([f512, (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), over], 1)
    Image.fromarray(sheet).save(seq / "fit_frame0.png")
    json.dump({"iou": float(iou), "yaw_deg": math.degrees(p["yaw"]), "elev_deg": math.degrees(p["elev"]),
               "dist": float(p["log_dist"].exp()), "at": p["at"].tolist(), "fov_deg": fov, "fov_fitted": bool(args.fit_fov)},
              open(seq / "fit_frame0.json", "w"), indent=2)
    print(f"IoU {iou:.3f}. Wrote {seq}/input.glb, camera.json, fit_frame0.png "
          "(frame 0 | render | red = only in video, blue = only in render)")


if __name__ == "__main__":
    main()
