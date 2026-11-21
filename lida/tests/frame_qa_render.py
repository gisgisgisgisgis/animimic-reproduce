"""Per-frame check, step 1: render every frame of a result and score it with DINOv2. Run in the `animamimic` env.

  python tests/frame_qa_render.py --seq data/paper_dog --result outputs/paper_dog/stage1.npz --out outputs/paper_dog/frame_qa

Writes <out>/frames/<view>/<t>.png for each view (the input camera, +/-45 deg as in the paper's VBench
evaluation, and +90 deg, where broken geometry shows most), <out>/rest/<view>.png (the mesh without
motion, a known-normal reference for the same view) and <out>/video/<t>.png (the reference video).

DINOv2 (an image model unrelated to this pipeline) embeds every image. dino.json holds, per view and
frame, the cosine similarity to the previous frame (a sudden glitch shows as a dip; VBench's subject
consistency uses the same idea) and to the rest render of that view (how far the frame drifts from the
undeformed animal). The reference video gets the same numbers as a baseline for natural motion.
Step 2 is tests/frame_qa_vlm.py, step 3 tests/frame_qa_report.py.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.baked import load_baked_mesh  # noqa: E402
from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.render import RasterRenderer, render_in_chunks  # noqa: E402

VIEWS = {"input": 0.0, "plus45": 45.0, "minus45": -45.0, "side90": 90.0}


def side_camera(cam, degrees):
    a = np.radians(degrees)
    Y = torch.tensor([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]], dtype=cam.R.dtype, device=cam.R.device)
    return Camera(Y.T @ cam.R, cam.T, cam.fov_deg, cam.image_size)


def to_png(x):
    return Image.fromarray((x.clamp(0, 1).numpy() * 255).round().astype(np.uint8))


@torch.no_grad()
def embed(paths, model, proc, device, batch=16):
    out = []
    for s in range(0, len(paths), batch):
        ims = [Image.open(p).convert("RGB") for p in paths[s:s + batch]]
        x = proc(images=ims, return_tensors="pt").to(device)
        e = model(**x).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(e, dim=-1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--texture", choices=("rig", "baked"), default="baked")
    ap.add_argument("--sigma", type=float, default=1e-6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="for DINOv2; its bicubic position-embedding resize has no MPS kernel in torch 2.5")
    args = ap.parse_args()

    seq, out = Path(args.seq), Path(args.out)
    if args.texture == "baked":
        mesh, src = load_baked_mesh(seq)
    else:
        mesh, src = RiggedGLB(seq / "rigged.glb").mesh(), None
    verts = torch.from_numpy(np.load(args.result)["verts"]).float()
    if src is not None:
        verts = verts[:, src]
    cam = Camera.from_json(seq / "camera.json")
    cam.image_size = args.size

    paths = {}
    for view, deg in VIEWS.items():
        c = side_camera(cam, deg)
        (out / "frames" / view).mkdir(parents=True, exist_ok=True)
        (out / "rest").mkdir(parents=True, exist_ok=True)
        rgb, _ = render_in_chunks(RasterRenderer(mesh, c, torch.device("cpu"), sigma=args.sigma), verts)
        paths[view] = []
        for t, im in enumerate(rgb):
            p = out / "frames" / view / f"{t:02d}.png"
            to_png(im).save(p)
            paths[view].append(p)
        rest, _ = RasterRenderer(mesh, c, torch.device("cpu"), sigma=args.sigma)(mesh.verts[None])
        to_png(rest[0]).save(out / "rest" / f"{view}.png")
    (out / "video").mkdir(parents=True, exist_ok=True)
    paths["video"] = []
    for t, f in enumerate(sorted((seq / "frames").glob("*.png"))):
        p = out / "video" / f"{t:02d}.png"
        Image.open(f).convert("RGB").resize((args.size, args.size), Image.LANCZOS).save(p)
        paths["video"].append(p)
    print(f"rendered {len(verts)} frames x {len(VIEWS)} views to {out}")

    from transformers import AutoImageProcessor, AutoModel
    device = torch.device(args.device)
    proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
    report = {"model": "facebook/dinov2-base, CLS token", "views": {}}
    for view, ps in paths.items():
        e = embed(ps, model, proc, device)
        ref = e[0] if view == "video" else embed([out / "rest" / f"{view}.png"], model, proc, device)[0]
        report["views"][view] = {
            "sim_prev": [None] + [round(float(x), 4) for x in (e[1:] * e[:-1]).sum(-1)],
            "sim_ref": [round(float(x), 4) for x in e @ ref],
            "ref": "frame 0 of the video" if view == "video" else "rest render, same view",
        }
    json.dump(report, open(out / "dino.json", "w"), indent=1)
    for view, r in report["views"].items():
        sp = [x for x in r["sim_prev"] if x is not None]
        print(f"{view:8s} sim_prev min {min(sp):.3f} median {np.median(sp):.3f} | sim_ref min {min(r['sim_ref']):.3f}")


if __name__ == "__main__":
    main()
