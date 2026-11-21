"""Per-frame check, step 3: combine the DINOv2 and VLM scores and flag abnormal frames. Run in the `animamimic` env.

  python tests/frame_qa_report.py --dir outputs/paper_dog/frame_qa

A frame is flagged when any of these holds:
- VLM: the model answers "not normal", or its score is 2 or more below its score for the undeformed mesh
  seen from the same view (the reconstructed mesh is imperfect, so the rest render is the fair baseline).
- jump: DINOv2 similarity to the previous frame falls far below the run's own median (median - max(0.03,
  5 x MAD)); a one-frame glitch shows this way.
- drift: DINOv2 similarity to the rest render falls more than 0.05 below the lowest value the reference video
  reaches against its own frame 0, i.e. the frame moved further from the animal than the real motion does.
Writes report.json, report.png (scores per frame) and flagged.png (the flagged frames, or the 8 lowest
VLM scores when nothing is flagged).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

VIEWS = ["input", "plus45", "minus45", "side90"]
LABEL = {"input": "輸入視角", "plus45": "+45°", "minus45": "−45°", "side90": "側面 90°", "video": "參考影片"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    dino = json.load(open(d / "dino.json"))["views"]
    vlm = json.load(open(d / "vlm.json"))["images"]

    video_scores = [vlm[k]["score"] for k in sorted(vlm) if k.startswith("video/") and vlm[k]["score"] is not None]
    drift_floor = min(dino["video"]["sim_ref"]) - 0.05
    report = {"video_vlm": {"mean": round(float(np.mean(video_scores)), 2), "min": int(min(video_scores)),
                            "not_normal": sum(vlm[k]["normal"] == "no" for k in vlm if k.startswith("video/"))},
              "drift_floor": round(drift_floor, 4), "views": {}, "flagged": []}
    for view in VIEWS:
        rest = vlm[f"rest/{view}.png"]
        sp = np.array([x for x in dino[view]["sim_prev"] if x is not None])
        med = float(np.median(sp))
        mad = float(np.median(np.abs(sp - med)))
        jump_floor = med - max(0.03, 5 * mad)
        frames = []
        for t, (p, r) in enumerate(zip(dino[view]["sim_prev"], dino[view]["sim_ref"])):
            v = vlm[f"frames/{view}/{t:02d}.png"]
            why = []
            if v["normal"] == "no" or (v["score"] is not None and rest["score"] is not None and v["score"] <= rest["score"] - 2):
                why.append("VLM")
            if p is not None and p < jump_floor:
                why.append("jump")
            if r < drift_floor:
                why.append("drift")
            frames.append({"t": t, "vlm_score": v["score"], "vlm_normal": v["normal"], "vlm_problem": v["problem"],
                           "dino_prev": p, "dino_rest": r, "flags": why})
            if why:
                report["flagged"].append({"view": view, **frames[-1]})
        scores = [f["vlm_score"] for f in frames if f["vlm_score"] is not None]
        report["views"][view] = {
            "rest_vlm": {"score": rest["score"], "normal": rest["normal"], "problem": rest["problem"]},
            "vlm_mean": round(float(np.mean(scores)), 2), "vlm_min": int(min(scores)),
            "not_normal": sum(f["vlm_normal"] == "no" for f in frames),
            "dino_prev_min": round(float(sp.min()), 4), "jump_floor": round(jump_floor, 4),
            "dino_rest_min": round(min(f["dino_rest"] for f in frames), 4),
            "flagged": sum(bool(f["flags"]) for f in frames), "frames": frames,
        }
    json.dump(report, open(d / "report.json", "w"), indent=1, ensure_ascii=False)

    print(f"reference video: VLM mean {report['video_vlm']['mean']}, min {report['video_vlm']['min']}, "
          f"not normal {report['video_vlm']['not_normal']}/{len(video_scores)}")
    for view, r in report["views"].items():
        print(f"{view:8s} rest VLM {r['rest_vlm']['score']} ({r['rest_vlm']['normal']}) | frames VLM mean {r['vlm_mean']} "
              f"min {r['vlm_min']} not-normal {r['not_normal']} | DINO prev min {r['dino_prev_min']} "
              f"rest min {r['dino_rest_min']} | flagged {r['flagged']}/{len(r['frames'])}")

    plt.rcParams["font.family"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2), dpi=110)
    colors = {"input": "#23695a", "plus45": "#c27c1a", "minus45": "#7a4fb0", "side90": "#b03a48", "video": "#666666"}
    T = len(report["views"]["input"]["frames"])
    for view in VIEWS:
        fr = report["views"][view]["frames"]
        ax[0].plot([f["vlm_score"] for f in fr], "-o", ms=3, color=colors[view], label=LABEL[view])
        ax[1].plot([f["dino_prev"] for f in fr], "-o", ms=3, color=colors[view], label=LABEL[view])
        ax[2].plot([f["dino_rest"] for f in fr], "-o", ms=3, color=colors[view], label=LABEL[view])
    ax[0].plot([vlm[f"video/{t:02d}.png"]["score"] for t in range(T)], "--", color=colors["video"], label=LABEL["video"])
    ax[1].plot(dino["video"]["sim_prev"], "--", color=colors["video"], label=LABEL["video"])
    ax[2].plot(dino["video"]["sim_ref"], "--", color=colors["video"], label="參考影片（對第 0 幀）")
    ax[2].axhline(drift_floor, color="#999", lw=0.8, ls=":")
    ax[0].set_ylim(0.5, 10.5)
    ax[0].set_title("VLM 分數（Qwen3-VL，1–10）")
    ax[1].set_title("DINOv2：和前一幀的相似度")
    ax[2].set_title("DINOv2：和不動的模型（同視角）的相似度")
    for a in ax:
        a.set_xlabel("幀")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(d / "report.png")

    picks = report["flagged"][:12]
    title = "被標記的幀"
    if not picks:
        allf = [{"view": v, **f} for v in VIEWS for f in report["views"][v]["frames"]]
        picks = sorted(allf, key=lambda f: (f["vlm_score"] if f["vlm_score"] is not None else 99, f["dino_prev"] or 1))[:8]
        title = "沒有幀被標記；這是 VLM 分數最低的 8 幀"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/STHeiti Medium.ttc", 18)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for f in picks:
        im = Image.open(d / "frames" / f["view"] / f"{f['t']:02d}.png").convert("RGB").resize((320, 320))
        canvas = Image.new("RGB", (320, 388), "white")
        canvas.paste(im, (0, 0))
        dr = ImageDraw.Draw(canvas)
        flags = ",".join(f["flags"]) or "—"
        dr.text((6, 324), f"{LABEL[f['view']]} 第 {f['t']} 幀  VLM {f['vlm_score']}  標記 {flags}", fill="black", font=font)
        dr.text((6, 350), str(f["vlm_problem"])[:34], fill="#555", font=font)
        tiles.append(np.asarray(canvas))
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    head = Image.new("RGB", (grid.shape[1], 40), "white")
    ImageDraw.Draw(head).text((8, 8), title, fill="black", font=font)
    Image.fromarray(np.concatenate([np.asarray(head), grid], 0)).save(d / "flagged.png")
    print(f"Wrote {d}/report.json, report.png, flagged.png")


if __name__ == "__main__":
    main()
