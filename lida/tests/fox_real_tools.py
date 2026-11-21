"""Measure the real preprocessing tools and Stage 1 on a video whose true 3D motion is known.

`make`  renders the fox's Walk cycle to <seq>/frames and stores the truth in <seq>/gt.npz.
`eval`  compares what SAM2, AllTracker, VGGT and Stage 1 produced against that truth.

The video has no generative-model artifacts and the rig is perfect, so this measures
the tools and the optimizer only.
"""
import json
import shutil
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.losses import lift_query_points, point_positions  # noqa: E402
from animamimic.sequence import load_sequence  # noqa: E402
from scripts.validate_fox import build_fox_case, make_parser  # noqa: E402


def make(args):
    device = torch.device(args.device)
    seq_dir = Path(args.seq)
    (seq_dir / "frames").mkdir(parents=True, exist_ok=True)
    seq, gt = build_fox_case(args, device)
    shutil.copy(args.glb, seq_dir / "rigged.glb")
    seq.camera.to_json(seq_dir / "camera.json")
    for i, f in enumerate(seq.rgb):
        imageio.imwrite(seq_dir / "frames" / f"{i:05d}.png", (f.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    imageio.imwrite(seq_dir / "reference.png", (seq.rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    imageio.imwrite(seq_dir / "reference_mask.png", ((seq.mask[0] > 0.5).cpu().numpy() * 255).astype(np.uint8))
    np.savez(seq_dir / "gt.npz", verts=gt.cpu().numpy(), mask=(seq.mask > 0.5).cpu().numpy())
    print(f"Wrote {seq.num_frames} frames at {seq.camera.image_size}px to {seq_dir}")


def eval_tools(args):
    device = torch.device("cpu")
    seq_dir = Path(args.seq)
    truth = np.load(seq_dir / "gt.npz")
    gt_verts, gt_mask = torch.from_numpy(truth["verts"]).float(), torch.from_numpy(truth["mask"])
    mesh = RiggedGLB(seq_dir / "rigged.glb").mesh().to(device)
    from animamimic.camera import Camera
    cam = Camera.from_json(seq_dir / "camera.json", device)
    H = cam.image_size
    report = {}

    masks = torch.stack([
        torch.from_numpy(np.asarray(imageio.imread(p)) > 127) for p in sorted((seq_dir / "masks").glob("*.png"))
    ])
    iou = (masks & gt_mask).sum((1, 2)).float() / (masks | gt_mask).sum((1, 2)).float()
    report["sam2_mask_iou_mean"] = float(iou.mean())
    report["sam2_mask_iou_last"] = float(iou[-1])

    tr = np.load(seq_dir / "tracks.npz")
    xy = torch.from_numpy(tr["xy"]).float()
    vis = torch.from_numpy(tr["visible"]).bool()
    keep, face, bary = lift_query_points(gt_verts[0], mesh.faces, cam, xy[0])
    true_xy, true_z = cam.project(point_positions(gt_verts, mesh.faces, face, bary))
    err = (xy[:, keep] - true_xy).norm(dim=-1)
    v = vis[:, keep]
    report["alltracker_points"] = int(keep.sum())
    report["alltracker_visible_fraction"] = float(v.float().mean())
    report["alltracker_epe_px_visible"] = float(err[v].mean())
    report["alltracker_epe_px_median_visible"] = float(err[v].median())
    report["alltracker_frac_within_3px"] = float((err[v] < 3.0).float().mean())
    report["image_size"] = H

    d = torch.from_numpy(np.load(seq_dir / "depth.npz")["depth"]).float()
    grid = true_xy / H * 2.0 - 1.0
    raw = torch.nn.functional.grid_sample(d[:, None], grid[:, None], align_corners=False, mode="bilinear")[:, 0, 0]
    pred = raw[v].numpy()
    target = true_z[v].numpy()
    scale, shift = np.polyfit(pred, target, 1)
    fitted = scale * pred + shift
    report["vggt_depth_absrel_after_scale_shift"] = float(np.mean(np.abs(fitted - target) / target))
    report["vggt_depth_corr"] = float(np.corrcoef(pred, target)[0, 1])

    if args.stage1:
        p = torch.from_numpy(np.load(args.stage1)["verts"]).float()
        diag = float((mesh.verts.max(0).values - mesh.verts.min(0).values).norm())
        rest = mesh.verts[None].expand_as(gt_verts)
        report["stage1_vertex_error_pct_of_diag"] = 100.0 * float((p - gt_verts).norm(dim=-1).mean()) / diag
        report["no_motion_baseline_pct_of_diag"] = 100.0 * float((rest - gt_verts).norm(dim=-1).mean()) / diag
        report["stage1_final_frame_error_pct_of_diag"] = 100.0 * float((p[-1] - gt_verts[-1]).norm(dim=-1).mean()) / diag

    print(json.dumps(report, indent=2))
    with open(seq_dir / "tool_report.json", "w") as f:
        json.dump(report, f, indent=2)


def ablate(args):
    """Write two sibling sequences that each swap one real tool output back to the truth."""
    from pytorch3d.structures import Meshes

    from animamimic.camera import Camera
    from animamimic.render import PyTorch3DRenderer

    device = torch.device("cpu")
    seq_dir = Path(args.seq).resolve()
    truth = np.load(seq_dir / "gt.npz")
    gt_verts = torch.from_numpy(truth["verts"]).float()
    mesh = RiggedGLB(seq_dir / "rigged.glb").mesh().to(device)
    cam = Camera.from_json(seq_dir / "camera.json", device)

    renderer = PyTorch3DRenderer(mesh, cam, device)
    # The soft settings return the nearest face within the blur radius, which biases depth toward the camera.
    renderer.rasterizer.raster_settings.blur_radius = 0.0
    renderer.rasterizer.raster_settings.faces_per_pixel = 1
    maps = []
    with torch.no_grad():
        for v in gt_verts:
            z = renderer.rasterizer(Meshes(verts=[v], faces=[mesh.faces])).zbuf[0, ..., 0]
            maps.append(z.where(z > 0, torch.full_like(z, float(z[z > 0].max()))))

    tr = np.load(seq_dir / "tracks.npz")
    xy = torch.from_numpy(tr["xy"]).float()
    keep, face, bary = lift_query_points(gt_verts[0], mesh.faces, cam, xy[0])
    true_xy, true_z = cam.project(point_positions(gt_verts, mesh.faces, face, bary))
    visible = []
    for t in range(gt_verts.shape[0]):
        hit, f_vis, b_vis = lift_query_points(gt_verts[t], mesh.faces, cam, true_xy[t])
        front = torch.full_like(true_z[t], float("inf"))
        front[hit] = cam.project((gt_verts[t][mesh.faces[f_vis]] * b_vis[..., None]).sum(1))[1]
        visible.append(true_z[t] <= front + 0.01)

    for suffix in ("gtdepth", "gttracks"):
        out = seq_dir.parent / f"{seq_dir.name}_{suffix}"
        out.mkdir(exist_ok=True)
        for name in ("frames", "masks", "rigged.glb", "camera.json", "tracks.npz", "depth.npz"):
            link = out / name
            if link.exists() or link.is_symlink():
                link.unlink()
            if not ((suffix == "gtdepth" and name == "depth.npz") or (suffix == "gttracks" and name == "tracks.npz")):
                link.symlink_to(seq_dir / name)
    np.savez(seq_dir.parent / f"{seq_dir.name}_gtdepth" / "depth.npz", depth=(3.7 * torch.stack(maps) + 0.9).numpy().astype(np.float32))
    np.savez(
        seq_dir.parent / f"{seq_dir.name}_gttracks" / "tracks.npz",
        xy=true_xy.numpy().astype(np.float32), visible=torch.stack(visible).numpy(),
    )
    print("Wrote ablation sequences")


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    ap = make_parser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--stage1", default=None)
    args = ap.parse_args()
    {"make": make, "eval": eval_tools, "ablate": ablate}[mode](args)
