"""Split a generated video into <seq>/frames/%05d.png at the camera's resolution."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--num_frames", type=int, default=32)
    args = ap.parse_args()

    seq = Path(args.seq)
    size = json.load(open(seq / "camera.json"))["image_size"]
    # The ffmpeg reader reports an infinite length, so list(reader) raises MemoryError.
    frames = [f for f in imageio.get_reader(args.video).iter_data()]
    pick = np.linspace(0, len(frames) - 1, min(args.num_frames, len(frames))).round().astype(int)
    out = seq / "frames"
    out.mkdir(exist_ok=True)
    for i, f in enumerate(pick):
        im = Image.fromarray(frames[f])
        if im.width != im.height:
            print(f"[warn] frame is {im.width}x{im.height}; center-cropping to a square.")
            s = min(im.size)
            left, top = (im.width - s) // 2, (im.height - s) // 2
            im = im.crop((left, top, left + s, top + s))
        im.resize((size, size), Image.LANCZOS).save(out / f"{i:05d}.png")
    print(f"Wrote {len(pick)} frames from {len(frames)} to {out}")


if __name__ == "__main__":
    main()
