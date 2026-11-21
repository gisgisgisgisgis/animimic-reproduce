"""Copy a Hunyuan3D-Paint texture (multi-view generated, all around the animal) into our baked atlas.

  python scripts/transfer_painted_texture.py --seq data/paper_wolf --painted outputs/paper_wolf_mv/painted.npz \
      --out outputs/paper_wolf_mv

The painted mesh is shape.obj re-indexed by xatlas, and shape.obj, input.glb, rigged.glb and our
texture_baked.npz all keep the same faces in the same order (prep_09b only moves and scales the vertices,
UniRig only splits them). So each texel of our atlas, a point in face f with barycentric b, is the same
point of face f in the painted mesh: its color is read from the painted texture at the painted UV, with no
nearest-point search. The script checks that correspondence and stops if it does not hold.

Writes three texture folders (each texture_baked.png + texture_baked.npz, usable with --texture_dir):
  mv/       the generated texture as is
  matched/  the same, with its colors fitted to frame 0 (the paint model removes lighting, frame 0 has it):
            a per-channel affine map fitted on the texels frame 0 sees directly
  hybrid/   frame 0's own colors where frame 0 sees the surface (green in texture_baked_source.png),
            the matched generated colors everywhere else, blended over a few texels at the border
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animamimic.gltf_rig import RiggedGLB  # noqa: E402


def rasterize_uv(uv, faces, W, H):
    """Face id (H, W) and barycentrics (H, W, 3) of every texel center covered by a UV triangle."""
    fid = np.full((H, W), -1, np.int64)
    bary = np.zeros((H, W, 3), np.float32)
    px = np.stack([uv[:, 0] * W - 0.5, (1.0 - uv[:, 1]) * H - 0.5], 1)  # texel-center coordinates
    for f, (a, b, c) in enumerate(faces):
        pa, pb, pc = px[a], px[b], px[c]
        x0, y0 = np.floor(np.minimum(np.minimum(pa, pb), pc)).astype(int)
        x1, y1 = np.ceil(np.maximum(np.maximum(pa, pb), pc)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(d) < 1e-12:
            continue
        w0 = ((pb[1] - pc[1]) * (gx - pc[0]) + (pc[0] - pb[0]) * (gy - pc[1])) / d
        w1 = ((pc[1] - pa[1]) * (gx - pc[0]) + (pa[0] - pc[0]) * (gy - pc[1])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        fid[gy[inside], gx[inside]] = f
        bary[gy[inside], gx[inside]] = np.stack([w0[inside], w1[inside], w2[inside]], -1)
    return fid, bary


def bilinear(img, u, v):
    """Sample img (H, W, 3) at UV (v up) with bilinear filtering."""
    H, W = img.shape[:2]
    x = np.clip(u * W - 0.5, 0, W - 1)
    y = np.clip((1.0 - v) * H - 0.5, 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--painted", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feather", type=float, default=4.0, help="hybrid: blend width in texels")
    args = ap.parse_args()
    seq, out = Path(args.seq), Path(args.out)

    p = np.load(args.painted)
    shape = trimesh.load(seq / "shape.obj", force="mesh", process=False)
    if len(p["faces"]) != len(shape.faces):
        raise SystemExit("painted mesh has a different face count; the face-order transfer does not apply")
    corner_err = np.abs(p["vertices"][p["faces"]] - np.asarray(shape.vertices)[shape.faces]).max()
    if corner_err > 1e-4:
        raise SystemExit(f"painted faces do not line up with shape.obj (max corner error {corner_err:.2e})")
    d = np.load(seq / "texture_baked.npz")
    rest = RiggedGLB(seq / "rigged.glb").mesh().verts.numpy()
    if len(d["faces"]) != len(p["faces"]):
        raise SystemExit("our atlas has a different face count from the painted mesh")

    base = np.asarray(Image.open(seq / "texture_baked.png").convert("RGB"), np.float32) / 255.0
    H, W = base.shape[:2]
    fid, bary = rasterize_uv(d["uv"], d["faces"], W, H)
    cov = fid >= 0
    f, b = fid[cov], bary[cov]
    puv = (p["uv"][p["faces"][f]] * b[..., None]).sum(1)
    ptex = p["texture"].astype(np.float32) / 255.0
    mv = base.copy()
    mv[cov] = bilinear(ptex, puv[:, 0], puv[:, 1])

    src = np.asarray(Image.open(seq / "texture_baked_source.png").convert("RGB"), np.float32) / 255.0
    if src.shape[:2] != (H, W):
        src = np.asarray(Image.fromarray((src * 255).astype(np.uint8)).resize((W, H), Image.NEAREST), np.float32) / 255.0
    seen = cov & (np.abs(src - np.array([0.1, 0.7, 0.1])).max(-1) < 0.06)

    # Per-channel affine fit, generated -> frame 0, on texels frame 0 sees.
    A = np.stack([mv[seen], np.ones((seen.sum(), 3), np.float32)], -1)  # (N, 3, 2)
    coef = [np.linalg.lstsq(A[:, c], base[seen][:, c], rcond=None)[0] for c in range(3)]
    matched = mv.copy()
    for c in range(3):
        matched[..., c] = np.where(cov, np.clip(mv[..., c] * coef[c][0] + coef[c][1], 0, 1), mv[..., c])

    dist = ndimage.distance_transform_edt(seen)  # texels from the border, inside the seen region
    w = np.clip(dist / max(args.feather, 1e-6), 0, 1)[..., None]
    hybrid = np.where(cov[..., None], w * base + (1 - w) * matched, base)

    for name, img in (("mv", mv), ("matched", matched), ("hybrid", hybrid)):
        o = out / name
        o.mkdir(parents=True, exist_ok=True)
        Image.fromarray((img * 255).round().astype(np.uint8)).save(o / "texture_baked.png")
        shutil.copy(seq / "texture_baked.npz", o / "texture_baked.npz")
    print(f"atlas {W}x{H}: {cov.sum()} texels, frame 0 sees {seen.sum() / cov.sum():.1%}; "
          f"colour fit gain {[round(float(c[0]), 3) for c in coef]} offset {[round(float(c[1]), 3) for c in coef]}; "
          f"face check {corner_err:.1e}; wrote {out}/{{mv,matched,hybrid}}")


if __name__ == "__main__":
    main()
