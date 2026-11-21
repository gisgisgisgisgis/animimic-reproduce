"""Side-by-side video of several Stage 1 versions of one sequence, with each frame's VLM score.

  python scripts/compare_versions_video.py --paper data/paper_refs/ours_4.mp4 \
      --version "舊設定=outputs/paper_dog/frame_qa" --version "新設定=outputs/paper_dog_sharp_rigid/frame_qa" \
      --out outputs/paper_dog_sharp_rigid/versions

Reads the frames tests/frame_qa_render.py wrote and the scores in report.json (run frame_qa_vlm.py and
frame_qa_report.py first). Row 1 is the input view: reference video | paper's result | each version.
Rows 2 and 3 are each version at +45 deg and at the side (90 deg). Each row is cropped to the box that
holds the animal in every frame and column of that row, so the animal fills the tile. Each tile shows the
VLM score of that frame (green = judged normal, red = not); the score of the undeformed mesh from the
same view is in brackets. Writes <out>.mp4 and <out>.png (first, middle and last frame).
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
ROWS = [("input", "輸入視角"), ("plus45", "+45°"), ("side90", "側面 90°")]


def crop_box(stacks, pad=0.08):
    """Square box around the non-white pixels of every image in `stacks` (each T x H x W x 3, same size)."""
    fg = np.zeros(stacks[0].shape[1:3], bool)
    for s in stacks:
        fg |= (s.min(-1) < 235).any(0)
    ys, xs = np.nonzero(fg)
    H, W = fg.shape
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + pad)
    half = min(half, H / 2, W / 2)
    cy, cx = np.clip(cy, half, H - half), np.clip(cx, half, W - half)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, help="the paper's result video for this sequence")
    ap.add_argument("--version", action="append", required=True, help="label=frame_qa dir, in display order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    versions = []
    for v in args.version:
        label, d = v.split("=", 1)
        versions.append((label, Path(d), json.load(open(Path(d) / "report.json"))))
    S = args.tile
    big, small = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT, 22)
    ref_paths = sorted((versions[0][1] / "video").glob("*.png"))
    T = len(ref_paths)

    def load(paths, size):
        return np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)) for p in paths])

    size = np.asarray(Image.open(versions[0][1] / "frames" / "input" / "00.png")).shape[0]
    ref = load(ref_paths, size)
    paper_all = [f for f in imageio.get_reader(args.paper).iter_data()]
    pick = np.linspace(0, len(paper_all) - 1, T).round().astype(int)
    paper = np.stack([np.asarray(Image.fromarray(paper_all[i][..., :3]).resize((size, size), Image.LANCZOS)) for i in pick])
    renders = {view: [load([d / "frames" / view / f"{t:02d}.png" for t in range(T)], size) for _, d, _ in versions]
               for view, _ in ROWS}
    boxes = {view: crop_box(renders[view] + ([ref, paper] if view == "input" else [])) for view, _ in ROWS}

    def tile(arr, box, score=None, normal=None, rest=None):
        im = Image.fromarray(arr).crop(box).resize((S, S), Image.LANCZOS)
        if score is not None:
            dr = ImageDraw.Draw(im)
            txt = f"VLM {score}" + (f"（靜止 {rest}）" if rest is not None else "")
            w = dr.textlength(txt, font=small)
            dr.rectangle((S - w - 16, 6, S - 6, 38), fill=(40, 140, 80) if normal == "yes" else (190, 50, 60))
            dr.text((S - w - 11, 9), txt, fill="white", font=small)
        return np.asarray(im)

    def label_tile(text):
        im = Image.new("RGB", (S, S), (245, 245, 245))
        dr = ImageDraw.Draw(im)
        w = dr.textlength(text, font=big)
        dr.text(((S - w) / 2, S / 2 - 16), text, fill=(90, 90, 90), font=big)
        return np.asarray(im)

    ncol = 2 + len(versions)
    head = Image.new("RGB", (ncol * S, 48), "white")
    dr = ImageDraw.Draw(head)
    for i, t in enumerate(["參考影片", "論文的結果"] + [v[0] for v in versions]):
        w = dr.textlength(t, font=big)
        dr.text((i * S + (S - w) / 2, 10), t, fill=(30, 30, 30), font=big)
    head = np.asarray(head)

    out = []
    for t in range(T):
        rows = []
        for r, (view, name) in enumerate(ROWS):
            box = boxes[view]
            cells = [tile(ref[t], box), tile(paper[t], box)] if r == 0 else [label_tile(name), label_tile("")]
            for k, (_, _, rep) in enumerate(versions):
                f = rep["views"][view]["frames"][t]
                cells.append(tile(renders[view][k][t], box, f["vlm_score"], f["vlm_normal"],
                                  rep["views"][view]["rest_vlm"]["score"]))
            rows.append(np.concatenate(cells, 1))
        out.append(np.concatenate([head] + rows, 0))
    imageio.mimsave(f"{args.out}.mp4", out * 2, fps=args.fps, macro_block_size=2, quality=8)
    Image.fromarray(np.concatenate([out[0], out[T // 2], out[-1]], 0)).save(f"{args.out}.png")
    print(f"Wrote {args.out}.mp4 ({T} frames, {out[0].shape[1]}x{out[0].shape[0]}) and {args.out}.png")


if __name__ == "__main__":
    main()
