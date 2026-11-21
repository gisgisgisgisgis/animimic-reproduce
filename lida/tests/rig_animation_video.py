"""Side-by-side video for an asset animated from a generated video (no paper result to compare with).

  python tests/rig_animation_video.py --qa outputs/hf_cow/frame_qa --out outputs/hf_cow/cow_vs_video

Reads the frames tests/frame_qa_render.py wrote (run it with --texture rig). One row per frame:
the driving video | our fitted mesh from the input camera | +45 deg | the side (90 deg) | -45 deg.
The video and input-view tiles share one crop box so they line up; each other view has its own.
Writes <out>.mp4 (played twice) and <out>.png (first, middle and last frame).
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_versions_video import FONT, crop_box  # noqa: E402

COLS = [("video", "生成的影片"), ("input", "原始 mesh｜輸入視角"), ("plus45", "+45°"), ("side90", "側面 90°"),
        ("minus45", "−45°")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="frame_qa_render.py output folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=360)
    ap.add_argument("--fps", type=float, default=7.75)
    args = ap.parse_args()

    qa, S = Path(args.qa), args.tile
    T = len(list((qa / "video").glob("*.png")))

    def load(view):
        d = qa / ("video" if view == "video" else f"frames/{view}")
        return np.stack([np.asarray(Image.open(d / f"{t:02d}.png").convert("RGB")) for t in range(T)])

    stacks = {v: load(v) for v, _ in COLS}
    boxes = {v: crop_box([stacks[v]]) for v, _ in COLS}
    boxes["video"] = boxes["input"] = crop_box([stacks["video"], stacks["input"]])

    big, small = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 20)
    head = Image.new("RGB", (len(COLS) * S, 44), "white")
    dr = ImageDraw.Draw(head)
    for i, (_, name) in enumerate(COLS):
        w = dr.textlength(name, font=big)
        dr.text((i * S + (S - w) / 2, 9), name, fill=(30, 30, 30), font=big)
    head = np.asarray(head)

    frames = []
    for t in range(T):
        cells = []
        for v, _ in COLS:
            im = Image.fromarray(stacks[v][t]).crop(boxes[v]).resize((S, S), Image.LANCZOS)
            if v == "video":
                ImageDraw.Draw(im).text((8, 6), f"frame {t}", fill=(150, 150, 150), font=small)
            cells.append(np.asarray(im))
        frames.append(np.concatenate([head, np.concatenate(cells, 1)], 0))
    imageio.mimsave(f"{args.out}.mp4", frames * 2, fps=args.fps, macro_block_size=2, quality=8)
    Image.fromarray(np.concatenate([frames[0], frames[T // 2], frames[-1]], 0)).save(f"{args.out}.png")
    print(f"Wrote {args.out}.mp4 ({T} frames, {frames[0].shape[1]}x{frames[0].shape[0]}) and {args.out}.png")


if __name__ == "__main__":
    main()
