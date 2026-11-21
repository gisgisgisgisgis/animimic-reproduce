"""Monocular depth for every frame with VGGT. Run in the `animamimic-prep` env."""
import argparse
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animamimic.device import pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    # On the fox, correlation with true depth was 0.97 for "single" and 0.56 for "multi":
    # VGGT's multi-view mode assumes a static scene, and the object here is moving.
    ap.add_argument("--mode", default="single", choices=["multi", "single"],
                    help="single: each frame on its own. multi: all frames together, as a static scene.")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device={device}")
    seq = Path(args.seq)
    paths = [str(p) for p in sorted((seq / "frames").glob("*.png"))]
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    groups = [paths] if args.mode == "multi" else [[p] for p in paths]
    parts = []
    with torch.no_grad(), amp:
        for group in groups:
            images = load_and_preprocess_images(group).to(device)
            parts.append(model(images[None])["depth"][0, ..., 0].float())  # (T, H, W)
    depth = torch.cat(parts)
    depth = F.interpolate(depth[:, None], size=(args.image_size, args.image_size), mode="bilinear", align_corners=False)
    np.savez(seq / "depth.npz", depth=depth[:, 0].cpu().numpy().astype(np.float32))
    print(f"Wrote depth for {depth.shape[0]} frames")


if __name__ == "__main__":
    main()
