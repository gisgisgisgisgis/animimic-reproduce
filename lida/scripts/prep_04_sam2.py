"""Propagate the rendered first-frame mask through the video with SAM2. Run in the `animamimic-prep` env."""
import argparse
import contextlib
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # SAM2 uses a few ops MPS lacks

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sam2.sam2_video_predictor import SAM2VideoPredictor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animamimic.device import pick_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--model", default="facebook/sam2.1-hiera-large")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    seq = Path(args.seq)
    paths = sorted((seq / "frames").glob("*.png"))
    first = np.asarray(Image.open(seq / "reference_mask.png").convert("L").resize(Image.open(paths[0]).size)) > 127
    out = seq / "masks"
    out.mkdir(exist_ok=True)

    device = pick_device(args.device)
    print(f"device={device}")
    predictor = SAM2VideoPredictor.from_pretrained(args.model, device=str(device))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode(), amp:
        # SAM2 reads a folder of integer-named JPEGs.
        for i, p in enumerate(paths):
            Image.open(p).convert("RGB").save(Path(tmp) / f"{i:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=tmp)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=first)
        count = 0
        for frame, _, logits in predictor.propagate_in_video(state):
            mask = (logits[0, 0] > 0.0).cpu().numpy()
            imageio.imwrite(out / f"{frame:05d}.png", (mask * 255).astype(np.uint8))
            count += 1
    print(f"Wrote {count} masks to {out}")


if __name__ == "__main__":
    main()
