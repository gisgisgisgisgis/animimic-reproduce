"""Run Stage 1, and optionally Stage 2, on one prepared sequence. CUDA is the default target."""
import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animamimic.device import pick_device  # noqa: E402
from animamimic.render import build_renderer, render_in_chunks  # noqa: E402
from animamimic.rig import forward_kinematics  # noqa: E402
from animamimic.sequence import load_sequence  # noqa: E402
from animamimic.stage1 import Stage1Config, run_stage1  # noqa: E402
from animamimic.stage2 import Stage2Config, run_stage2  # noqa: E402


def save_preview(seq, verts, device, backend, path, sigma=1e-4):
    rgb, mask = render_in_chunks(build_renderer(seq.mesh, seq.camera, device, backend, sigma), verts)
    if rgb is None:
        rgb = mask[..., None].repeat(1, 1, 1, 3)
    side = np.concatenate([(seq.rgb.cpu().numpy() * 255).astype(np.uint8), (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)], axis=2)
    imageio.mimsave(path, list(side), fps=12, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--image_size", type=int, default=None, help="optimize at a lower resolution than the frames")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch_frames", type=int, default=4)
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--stage2_iters", type=int, nargs="*", default=[30, 30])
    ap.add_argument("--stage2_E", default=None, help="stage2.npz whose moduli to simulate with, without optimizing")
    ap.add_argument("--fps", type=float, default=24.0,
                    help="frame rate of the loaded frames: (frames - 1) / clip seconds; 6.2 for 32 frames of a 121-frame 24 fps clip")
    ap.add_argument("--depth_norm", default="first_frame", choices=["first_frame", "per_frame"])
    ap.add_argument("--w_depth", type=float, default=1.0)
    ap.add_argument("--depth_mode", default="points", choices=["points", "mean"],
                    help="points: paper Eq. 8 on every tracked point. mean: only each frame's mean depth (not in the paper)")
    ap.add_argument("--w_reg", type=float, default=Stage1Config.w_reg, help="weight of the Eq. 9 pose limits")
    ap.add_argument("--trans_max", type=float, default=Stage1Config.trans_max, help="Eq. 9 limit on non-root joint translation, per axis")
    ap.add_argument("--skin_smooth", type=int, default=0, help="Laplacian smoothing steps on UniRig's skin weights; 0 = off")
    ap.add_argument("--stage1_from", default=None, help="skip Stage 1 and load this stage1.npz")
    ap.add_argument("--sigma", type=float, default=Stage1Config.sigma,
                    help="rasterizer edge softness for L_rgb/L_mask; 1e-4 blurs edges by ~4 px at 256 px")
    ap.add_argument("--w_rigid", type=float, default=Stage1Config.w_rigid,
                    help="edge-length rigidity weight; 0 = off (the paper has no such term)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device(args.device)
    print(f"device={device}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq = load_sequence(args.seq, device, args.image_size, args.skin_smooth)
    print(f"{seq.num_frames} frames, {seq.mesh.verts.shape[0]} vertices, {seq.rig.num_joints} joints, "
          f"{seq.tracks.face.shape[0]} tracks, skin_smooth={args.skin_smooth}")

    if args.stage1_from:
        saved = Path(args.stage1_from).with_name("stage1.json")
        if saved.exists():
            used = json.load(open(saved)).get("skin_smooth", 0)
            if used != args.skin_smooth:
                raise SystemExit(f"{args.stage1_from} was fitted with --skin_smooth {used}; pass the same value.")
        data = np.load(args.stage1_from)
        r6d, t = torch.from_numpy(data["r6d"]).to(device), torch.from_numpy(data["t"]).to(device)
        A_r, A_t = forward_kinematics(seq.rig, r6d, t)
        res1 = {"A_r": A_r, "A_t": A_t, "verts": torch.from_numpy(data["verts"]).to(device)}
    else:
        cfg1 = Stage1Config(iters=args.iters, batch_frames=args.batch_frames, depth_norm=args.depth_norm, w_depth=args.w_depth, depth_mode=args.depth_mode,
                            w_reg=args.w_reg, trans_max=args.trans_max, sigma=args.sigma, w_rigid=args.w_rigid)
        res1 = run_stage1(seq, cfg1, device, args.backend)
        np.savez(out / "stage1.npz", r6d=res1["r6d"].cpu().numpy(), t=res1["t"].cpu().numpy(), verts=res1["verts"].cpu().numpy())
        json.dump({"history": res1["history"], "config": res1["config"], "skin_smooth": args.skin_smooth},
                  open(out / "stage1.json", "w"), indent=2)
        save_preview(seq, res1["verts"], device, args.backend, out / "stage1_vs_video.mp4", args.sigma)

    if args.stage2:
        tet = np.load(Path(args.seq) / "tet.npz")
        cfg = Stage2Config(iters_per_level=args.stage2_iters, fps=args.fps, batch_frames=args.batch_frames,
                           depth_norm=args.depth_norm, w_depth=args.w_depth, sigma=args.sigma)
        E_fixed = np.load(args.stage2_E)["E"] if args.stage2_E else None
        if E_fixed is not None:
            cfg.iters_per_level = []
        res2 = run_stage2(seq, res1, tet["verts"], tet["tets"], cfg, device, args.backend, E_fixed=E_fixed)
        np.savez(out / "stage2.npz", E=res2["E"].cpu().numpy(), cluster=res2["cluster"], verts=res2["verts"].cpu().numpy(),
                 tet_x=res2["tet_x"].numpy())
        json.dump({"history": res2["history"], "config": res2["config"], "E_from": args.stage2_E}, open(out / "stage2.json", "w"), indent=2)
        save_preview(seq, res2["verts"], device, args.backend, out / "stage2_vs_video.mp4", args.sigma)


if __name__ == "__main__":
    main()
