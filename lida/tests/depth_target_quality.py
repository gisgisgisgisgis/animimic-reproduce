"""How close is each depth normalization N(.) to the true depth, on the fox video whose truth is known.

No optimization: this only compares the depth targets that Stage 1 would chase.
Usage: python tests/depth_target_quality.py [--seq data/fox_real] [--depth depth_single.npz]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.losses import lift_query_points, normalize_depth_targets, point_positions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=str(ROOT / "data/fox_real"))
    ap.add_argument("--depth", default="depth_single.npz")
    args = ap.parse_args()
    seq = Path(args.seq)
    gt = torch.from_numpy(np.load(seq / "gt.npz")["verts"]).float()
    mesh = RiggedGLB(seq / "rigged.glb").mesh()
    cam = Camera.from_json(seq / "camera.json", torch.device("cpu"))
    tr = np.load(seq / "tracks.npz")
    xy = torch.from_numpy(tr["xy"]).float()
    vis = torch.from_numpy(tr["visible"]).bool()
    keep, face, bary = lift_query_points(mesh.verts, mesh.faces, cam, xy[0])
    xy, vis = xy[:, keep], vis[:, keep]
    _, true_z = cam.project(point_positions(gt, mesh.faces, face, bary))
    _, rest_z = cam.project(point_positions(mesh.verts[None], mesh.faces, face, bary))
    rest_z = rest_z[0]

    d = torch.from_numpy(np.load(seq / args.depth)["depth"]).float()
    grid = xy / cam.image_size * 2.0 - 1.0
    raw = F.grid_sample(d[:, None], grid[:, None], align_corners=False)[:, 0, 0]

    T = raw.shape[0]
    diag = float((mesh.verts.max(0).values - mesh.verts.min(0).values).norm())

    def err(target):
        e = (target - true_z).abs()
        per_t = [float(e[t][vis[t]].mean()) / diag * 100 for t in range(T)]
        return {"mean_pct_diag": round(float(np.mean(per_t)), 3), "last_pct_diag": round(per_t[-1], 3)}

    out = {}
    for mode in ("first_frame", "per_frame"):
        out[mode] = err(normalize_depth_targets(raw, rest_z, vis, mode))
    # Oracle per-frame affine fit to the truth: the best any scale-shift-invariant loss could do.
    oracle = raw.clone()
    for t in range(T):
        v = vis[t]
        A = torch.stack((raw[t][v], torch.ones_like(raw[t][v])), 1)
        sol = torch.linalg.lstsq(A, true_z[t][v][:, None]).solution[:, 0]
        oracle[t] = raw[t] * sol[0] + sol[1]
    out["oracle_per_frame_affine"] = err(oracle)
    out["rest_depth_no_motion"] = err(rest_z[None].expand_as(true_z))
    # Delta: rest depth plus the change of the raw depth since frame 0, scaled by the frame-0 fit.
    # A fixed per-point bias of the depth network (wrong shape) cancels; only its change over time is used.
    v0 = vis[0]
    a = torch.linalg.lstsq(torch.stack((raw[0][v0], torch.ones_like(raw[0][v0])), 1), rest_z[v0][:, None]).solution[0, 0]
    out["delta_from_frame0"] = err(rest_z[None] + a * (raw - raw[0:1]))
    # Same, but each frame's change is re-centred (median), which removes per-frame shift drift.
    delta = a * (raw - raw[0:1])
    centred = delta - torch.stack([delta[t][vis[t]].median() for t in range(T)])[:, None]
    out["delta_recentred"] = err(rest_z[None] + centred)
    # Points away from the silhouette, where bilinear sampling mixes in background depth.
    m = torch.from_numpy(np.load(seq / "gt.npz")["mask"]).float()
    eroded = -F.max_pool2d(-m[:, None], 9, 1, 4)[:, 0]
    inner = F.grid_sample(eroded[:, None], grid[:, None], align_corners=False)[:, 0, 0] > 0.99
    e = (normalize_depth_targets(raw, rest_z, vis, "per_frame") - true_z).abs()
    out["per_frame_inner_points_only"] = round(float(e[vis & inner].mean()) / diag * 100, 3)
    out["per_frame_edge_points_only"] = round(float(e[vis & ~inner].mean()) / diag * 100, 3)
    out["fraction_inner"] = round(float((vis & inner).float().mean()), 3)
    std_ratio = [float(true_z[t][vis[t]].std() / true_z[0][vis[0]].std()) for t in range(T)]
    mean_shift = [float(true_z[t][vis[t]].mean() - true_z[0][vis[0]].mean()) / diag * 100 for t in range(T)]
    out["true_depth_std_ratio_vs_frame0_minmax"] = [round(min(std_ratio), 3), round(max(std_ratio), 3)]
    out["true_depth_mean_shift_pct_diag_minmax"] = [round(min(mean_shift), 3), round(max(mean_shift), 3)]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
