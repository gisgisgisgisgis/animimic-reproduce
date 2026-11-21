"""Stack several compare_with_paper.py results into one overview video, one row per sequence.

  python tests/paper_overview_video.py --out outputs/paper_overview \
      --item "白狼=outputs/paper_wolf_sharp_rigid/compare_with_paper_baked_512" \
      --item "貓=outputs/paper_cat_sharp_rigid/compare_with_paper_baked_512" ...

Each <prefix>.mp4 is the 5-panel video compare_with_paper.py writes (reference | paper | ours | ours +90 | ours -45)
and <prefix>.json its scores; the row label shows the silhouette IoU against the video, ours vs the paper's.
Writes <out>.mp4 and <out>.png (middle frame).
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = next((f for f in ("/System/Library/Fonts/STHeiti Medium.ttc",
                         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc") if Path(f).exists()),
            "/System/Library/Fonts/STHeiti Medium.ttc")
COLUMNS = ["參考影片", "論文的結果", "我們（輸入視角）", "我們 +90°", "我們 −45°"]


def ours_key(scores):
    return next(k for k in scores if not k.startswith("paper") and not k.startswith("ours, rest"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--item", action="append", required=True, help="label=prefix of compare_with_paper output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell", type=int, default=256)
    args = ap.parse_args()
    S, label_w, head_h = args.cell, 230, 44
    font, small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 19)
    rows = []
    for item in args.item:
        label, prefix = item.split("=", 1)
        s = json.load(open(f"{prefix}.json"))
        ours, paper = s[ours_key(s)], s["paper (their mesh and texture)"]
        frames = [np.asarray(Image.fromarray(f).resize((5 * S, S), Image.LANCZOS))
                  for f in imageio.get_reader(f"{prefix}.mp4").iter_data()]
        frames = frames[: len(frames) // 2]  # compare_with_paper writes the clip twice
        tag = Image.new("RGB", (label_w, S), "white")
        d = ImageDraw.Draw(tag)
        d.text((14, S // 2 - 52), label, fill=(0, 0, 0), font=font)
        d.text((14, S // 2 - 8), f"IoU 我們 {ours['mask_iou']:.3f}", fill=(40, 40, 40), font=small)
        d.text((14, S // 2 + 20), f"IoU 論文 {paper['mask_iou']:.3f}", fill=(40, 40, 40), font=small)
        rows.append((np.asarray(tag), frames))
    T = min(len(f) for _, f in rows)
    head = Image.new("RGB", (label_w + 5 * S, head_h), "white")
    d = ImageDraw.Draw(head)
    for i, c in enumerate(COLUMNS):
        d.text((label_w + i * S + 10, 8), c, fill=(0, 0, 0), font=small)
    head = np.asarray(head)
    out = []
    for t in range(T):
        out.append(np.concatenate([head] + [np.concatenate([tag, f[t]], 1) for tag, f in rows], 0))
    imageio.mimsave(f"{args.out}.mp4", out * 2, fps=8, macro_block_size=2, quality=8)
    Image.fromarray(out[T // 2]).save(f"{args.out}.png")
    print(f"wrote {args.out}.mp4 ({len(rows)} rows, {T} frames) and {args.out}.png")


if __name__ == "__main__":
    main()
