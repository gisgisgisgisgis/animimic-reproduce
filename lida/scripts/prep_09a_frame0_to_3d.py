"""Reconstruct a mesh from the first frame of an existing video with Hunyuan3D-2 (shape only).

For videos that come without their 3D asset, such as the paper's reference videos. The paper's assets
were themselves image-to-3D outputs (Meshy, Tripo), so this follows the same route with an open model.
Run in the `hunyuan3d` env:

  python scripts/prep_09a_frame0_to_3d.py --video data/paper_refs/ref_4.mp4 --seq data/paper_dog

Writes <seq>/frame0.png, <seq>/frame0_mask.png (white-background threshold) and <seq>/shape.obj.
Hunyuan3D's texture stage needs CUDA, so color comes from the frame in prep_09b_fit_frame0.py.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402


def white_background_mask(frame, threshold=15.0):
    """Largest connected region that differs from the border color by more than `threshold`."""
    f = frame.astype(np.float32)
    border = np.concatenate([f[:20].reshape(-1, 3), f[-20:].reshape(-1, 3), f[:, :20].reshape(-1, 3), f[:, -20:].reshape(-1, 3)])
    m = np.abs(f - np.median(border, 0)).max(-1) > threshold
    m = ndimage.binary_opening(m, iterations=2)
    lab, k = ndimage.label(m)
    if k == 0:
        raise ValueError("No foreground found on frame 0.")
    m = lab == (np.argmax(ndimage.sum(m, lab, range(1, k + 1))) + 1)
    return ndimage.binary_fill_holes(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--repo", default="third_party/Hunyuan3D-2")
    ap.add_argument("--model", default="tencent/Hunyuan3D-2mini")
    ap.add_argument("--subfolder", default="hunyuan3d-dit-v2-mini")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--octree_resolution", type=int, default=380)
    ap.add_argument("--faces", type=int, default=40000, help="decimate to this many faces")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seq = Path(args.seq)
    seq.mkdir(parents=True, exist_ok=True)
    frame = next(iter(imageio.get_reader(args.video).iter_data()))[..., :3]
    mask = white_background_mask(frame)
    imageio.imwrite(seq / "frame0.png", frame)
    imageio.imwrite(seq / "frame0_mask.png", (mask * 255).astype(np.uint8))

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline

    rgba = Image.fromarray(np.dstack([frame, (mask * 255).astype(np.uint8)]), "RGBA")
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model, subfolder=args.subfolder, use_safetensors=True, device=args.device, dtype=dtype)
    mesh = pipe(image=rgba, num_inference_steps=args.steps, octree_resolution=args.octree_resolution,
                generator=torch.Generator().manual_seed(args.seed), output_type="trimesh")[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=args.faces)
    mesh.export(seq / "shape.obj")
    print(f"Wrote {seq}/shape.obj: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")


if __name__ == "__main__":
    main()
