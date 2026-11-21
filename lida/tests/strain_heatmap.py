"""Where a result stretches the surface: heat map at the worst frame, and which bones carry the stretch.

  python tests/strain_heatmap.py --seq data/paper_cat --result outputs/paper_cat_sharp_rigid/stage1.npz \
      --out outputs/paper_cat_sharp_rigid/strain

Strain per face = largest |edge length / rest length - 1| of its three edges, rest lengths floored at a quarter of
the median edge as in tests/surface_distortion.py. The worst frame is the one with the highest 99th percentile.
Writes <out>.png: for every --result, the textured mesh and the heat map (white 0 → red 30%+) from the input
view, +90° and -45° at that result's worst frame, and <out>.json with per-frame p99 and the bones whose faces
stretch most (each face goes to the joint with the largest raw skin weight).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from frame_qa_render import side_camera  # noqa: E402

from animamimic.baked import load_baked_mesh  # noqa: E402
from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import MeshData, RiggedGLB  # noqa: E402
from animamimic.render import RasterRenderer  # noqa: E402

FONT = next((f for f in ("/System/Library/Fonts/STHeiti Medium.ttc",
                         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc") if Path(f).exists()),
            "/System/Library/Fonts/STHeiti Medium.ttc")
VIEWS = [0, 90, -45]
CAP = 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--result", nargs="+", required=True)
    ap.add_argument("--label", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    seq, dev = Path(args.seq), torch.device(args.device)
    glb = RiggedGLB(seq / "rigged.glb")
    rest = glb.mesh()
    faces = rest.faces
    V0 = rest.verts.double()
    l0 = torch.stack([(V0[faces[:, a]] - V0[faces[:, b]]).norm(dim=-1) for a, b in [(0, 1), (1, 2), (2, 0)]])
    l0 = l0.clamp_min(0.25 * l0.median())  # (3, F)
    # dominant joint per face, from the raw skin weights of its first corner
    idx, w = glb.skin_idx, glb.skin_w
    dom = np.asarray(idx)[np.arange(len(idx)), np.asarray(w).argmax(1)][faces[:, 0].numpy()]
    try:
        textured, src = load_baked_mesh(seq)
    except FileNotFoundError:
        textured, src = rest, None
    cam = Camera.from_json(seq / "camera.json")
    cam.image_size = args.size
    labels = args.label or [Path(r).parent.name + "/" + Path(r).stem for r in args.result]
    font = ImageFont.truetype(FONT, 20)
    rows, report = [], {}
    for label, res in zip(labels, args.result):
        verts = torch.from_numpy(np.load(res)["verts"]).double()
        s = torch.stack([((verts[:, faces[:, a]] - verts[:, faces[:, b]]).norm(dim=-1) / l0[k] - 1).abs()
                         for k, (a, b) in enumerate([(0, 1), (1, 2), (2, 0)])], -1).amax(-1)  # (T, F)
        p99 = torch.quantile(s.float(), 0.99, dim=1)
        t = int(p99.argmax())
        st = s[t].float().numpy()
        per_joint = {}
        for j in np.unique(dom):
            m = dom == j
            per_joint[int(j)] = (float(np.percentile(st[m], 99)), float(st[m].mean()), int(m.sum()))
        top = sorted(per_joint.items(), key=lambda kv: -kv[1][0])[:6]
        report[label] = {"worst_frame": t, "p99_per_frame": [round(float(x), 4) for x in p99],
                         "top_joints_at_worst_frame": [{"joint": j, "name": glb.joint_names[j] if hasattr(glb, "joint_names") else str(j),
                                                         "p99": round(v[0], 4), "mean": round(v[1], 4), "faces": v[2]}
                                                        for j, v in top]}
        # heat colors: one texel per face; the faces get their own corners (some vertices are shared) so each
        # corner can carry its face's uv
        F = faces.shape[0]
        W = int(np.ceil(np.sqrt(F)))
        x = np.clip(st / CAP, 0, 1)
        col = np.stack([np.ones_like(x), 1 - x, 1 - x], -1).astype(np.float32)
        tex = np.ones((W * W, 3), np.float32)
        tex[:F] = col
        tex = torch.from_numpy(tex.reshape(W, W, 3))
        fi = torch.arange(F)
        uv_f = torch.stack([(fi % W + 0.5) / W, 1 - (fi // W + 0.5) / W], -1).float()
        uv = uv_f.repeat_interleave(3, 0)
        soup = torch.arange(3 * F).reshape(F, 3)
        heat = MeshData(rest.verts[faces].reshape(-1, 3), soup, uv, tex)
        vt = verts[t].float()
        cells = []
        for a in VIEWS:
            c = side_camera(cam, a)
            tv = vt[src] if src is not None else vt
            rgb, _ = RasterRenderer(textured, c, dev, sigma=1e-6)(tv[None].to(dev))
            hm, _ = RasterRenderer(heat, c, dev, sigma=1e-6, gamma=1e-6)(vt[faces].reshape(-1, 3)[None].to(dev))
            for im, kind in ((rgb, "貼圖"), (hm, "拉伸")):
                pil = Image.fromarray((im[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
                ImageDraw.Draw(pil).text((6, 4), f"{label} 第{t}幀 {kind} {a}°", fill=(0, 0, 160), font=font)
                cells.append(np.asarray(pil))
        rows.append(np.concatenate(cells, 1))
        print(f"{label}: worst frame {t}, p99 {float(p99[t]):.3f}; top joints " +
              ", ".join(f"{j}:{v[0]:.2f}" for j, v in top))
    Image.fromarray(np.concatenate(rows, 0)).save(f"{args.out}.png")
    json.dump(report, open(f"{args.out}.json", "w"), indent=1, ensure_ascii=False)
    print(f"wrote {args.out}.png and .json")


if __name__ == "__main__":
    main()
