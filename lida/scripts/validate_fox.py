"""Check the optimizer on a case with a known answer.

A rigged, textured fox is animated with its own Walk cycle and rendered from one
camera. Stage 1 sees only that video and has to recover the 3D motion. Error is
measured against the true 3D motion and from two unseen cameras at +/-45 degrees,
the paper's novel-view protocol.

This tests the optimizer. It is not a reproduction of the paper's numbers: there
are no video-model artifacts, and the rig is perfect.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.camera import Camera  # noqa: E402
from animamimic.device import pick_device  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.losses import TrackTargets, lift_query_points, point_positions  # noqa: E402
from animamimic.render import build_renderer, render_in_chunks  # noqa: E402
from animamimic.rig import forward_kinematics, linear_blend_skinning, matrix_to_rotation_6d  # noqa: E402
from animamimic.sequence import Sequence  # noqa: E402
from animamimic.stage1 import Stage1Config, run_stage1  # noqa: E402


def gt_local_pose(glb, rig, name, times):
    """Convert the asset's animation to this project's local (6D rotation, translation) parameters."""
    A_r, A_t = glb.sample_skin_transforms(name, times)
    G_t = (A_r @ rig.joints[..., None])[..., 0] + A_t
    offsets = rig.offsets()
    R_local, t_local = A_r.clone(), G_t - offsets
    for j, p in enumerate(rig.parents):
        if p >= 0:
            pr = A_r[:, p].transpose(-1, -2)
            R_local[:, j] = pr @ A_r[:, j]
            t_local[:, j] = (pr @ (G_t[:, j] - G_t[:, p])[..., None])[..., 0] - offsets[j]
    return matrix_to_rotation_6d(R_local), t_local


def ramp_from_rest(r6, t, ramp):
    """Blend from the rest pose over the first frames: a real video starts at the rendered rest mesh."""
    from animamimic.rig import rotation_6d_to_matrix

    T, K = r6.shape[:2]
    w = torch.clamp(torch.arange(T, dtype=torch.float32) / max(ramp, 1), max=1.0)
    rotvec = Rotation.from_matrix(rotation_6d_to_matrix(r6).reshape(-1, 3, 3).numpy().astype(np.float64)).as_rotvec()
    rotvec = rotvec.reshape(T, K, 3) * w.numpy()[:, None, None]
    m = torch.from_numpy(Rotation.from_rotvec(rotvec.reshape(-1, 3)).as_matrix()).float().reshape(T, K, 3, 3)
    return matrix_to_rotation_6d(m), t * w[:, None, None]


def build_tracks(verts, faces, camera, mask0, num_points, seed, depth_noise):
    H = camera.image_size
    rng = np.random.default_rng(seed)
    ys, xs = np.where(mask0.cpu().numpy() > 0.5)
    pick = rng.choice(len(xs), size=min(num_points, len(xs)), replace=False)
    query = torch.tensor(np.c_[xs[pick] + 0.5, ys[pick] + 0.5], device=verts.device, dtype=torch.float32)
    keep, face, bary = lift_query_points(verts[0], faces, camera, query)
    xy, z = camera.project(point_positions(verts, faces, face, bary))
    visible = torch.zeros(xy.shape[:2], dtype=torch.bool, device=verts.device)
    for t in range(verts.shape[0]):
        inside = ((xy[t] >= 0) & (xy[t] < H)).all(-1)
        hit, f_vis, b_vis = lift_query_points(verts[t], faces, camera, xy[t])
        front = torch.full_like(z[t], float("inf"))
        front[hit] = camera.project((verts[t][faces[f_vis]] * b_vis[..., None]).sum(1))[1]
        visible[t] = inside & (z[t] <= front + 0.01)
    # Monocular depth is only defined up to scale and shift.
    raw = 3.7 * z + 0.9 + depth_noise * torch.randn_like(z)
    return TrackTargets(face, bary, xy, visible, raw)


def psnr(a, b):
    return float(-10.0 * torch.log10((a - b).pow(2).mean((1, 2, 3)).clamp_min(1e-10)).mean())


def to_frames(x):
    return (x.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def build_fox_case(args, device):
    """Build the known-answer sequence. Returns (seq, ground-truth vertices)."""
    glb = RiggedGLB(args.glb)
    mesh, rig = glb.mesh(), glb.rig()
    duration = glb.animation_duration(args.animation)
    times = np.linspace(0.0, duration, args.frames, endpoint=False)
    r6_gt, t_gt = ramp_from_rest(*gt_local_pose(glb, rig, args.animation, times), args.ramp)
    gt_verts = linear_blend_skinning(rig, mesh.verts, *forward_kinematics(rig, r6_gt, t_gt)).to(device)

    mesh, rig = mesh.to(device), rig.to(device)
    cam = Camera.look_at(2.2, 15.0, 30.0, 40.0, args.image_size, device=device)
    renderer = build_renderer(mesh, cam, device, args.backend)
    rgb, mask = render_in_chunks(renderer, gt_verts)
    if rgb is None:
        rgb = torch.ones(*mask.shape, 3, device=device)
    tracks = build_tracks(gt_verts, mesh.faces, cam, mask[0], args.num_points, 0, args.depth_noise)
    print(f"tracked points: {tracks.xy.shape[1]}, visible fraction: {float(tracks.visible.float().mean()):.2f}")
    return Sequence(mesh, rig, cam, rgb, mask, tracks), gt_verts


def make_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default=str(ROOT / "data/fox/Fox.glb"))
    ap.add_argument("--animation", default="Walk")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--ramp", type=int, default=4)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--num_points", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch_frames", type=int, default=4)
    ap.add_argument("--depth_noise", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--out", default=str(ROOT / "outputs/fox"))
    return ap


def main():
    args = make_parser().parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} backend={args.backend}")

    seq, gt_verts = build_fox_case(args, device)
    mesh, rig = seq.mesh, seq.rig
    cfg = Stage1Config(iters=args.iters, batch_frames=args.batch_frames)
    start = time.time()
    res = run_stage1(seq, cfg, device, args.backend, log_every=max(args.iters // 20, 1))
    seconds = time.time() - start

    pred = res["verts"]
    diag = float((mesh.verts.max(0).values - mesh.verts.min(0).values).norm())
    rest = mesh.verts[None].expand_as(pred)
    metrics = {
        "device": str(device),
        "seconds": seconds,
        "seconds_per_iter": seconds / args.iters,
        "num_vertices": int(mesh.verts.shape[0]),
        "num_joints": rig.num_joints,
        "bbox_diag": diag,
        "vertex_error_pct_of_diag": 100.0 * float((pred - gt_verts).norm(dim=-1).mean()) / diag,
        "no_motion_baseline_pct_of_diag": 100.0 * float((rest - gt_verts).norm(dim=-1).mean()) / diag,
        "final_frame_error_pct_of_diag": 100.0 * float((pred[-1] - gt_verts[-1]).norm(dim=-1).mean()) / diag,
    }

    panels = []
    for azim_offset in (0.0, 45.0, -45.0):
        view = Camera.look_at(2.2, 15.0, 30.0 + azim_offset, 40.0, args.image_size, device=device)
        r = build_renderer(mesh, view, device, args.backend)
        gt_rgb, gt_mask = render_in_chunks(r, gt_verts)
        pr_rgb, pr_mask = render_in_chunks(r, pred)
        key = f"view{azim_offset:+.0f}"
        iou = ((gt_mask > 0.5) & (pr_mask > 0.5)).sum((1, 2)).float() / ((gt_mask > 0.5) | (pr_mask > 0.5)).sum((1, 2)).float().clamp_min(1)
        metrics[f"mask_iou_{key}"] = float(iou.mean())
        if gt_rgb is not None:
            metrics[f"psnr_{key}"] = psnr(gt_rgb, pr_rgb)
            panels.append(np.concatenate((to_frames(gt_rgb), to_frames(pr_rgb)), axis=2))

    with open(out / "metrics.json", "w") as f:
        json.dump({"metrics": metrics, "history": res["history"], "config": res["config"]}, f, indent=2)
    np.savez(out / "stage1.npz", r6d=res["r6d"].cpu().numpy(), t=res["t"].cpu().numpy(), verts=pred.cpu().numpy())
    if panels:
        grid = np.concatenate(panels, axis=1)
        imageio.mimsave(out / "gt_vs_recovered.mp4", list(grid), fps=8, macro_block_size=1)
        imageio.imwrite(out / "gt_vs_recovered_frame.png", grid[args.frames // 2])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
