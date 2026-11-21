"""Summarize tests/frame_sim_vlm.py results. Run in any env:

  python tests/frame_sim_report.py --vqa "paper=outputs/x/paper_frames/vlm_vqascore.json" \\
      "ours=outputs/x/frame_qa/vlm_vqascore.json" --out outputs/x/similarity.json

--vqa: mean P(Yes) per view and question over the frames, and how many frames are above 0.5.
--sim / --pref: the 1-10 scores and forced choices (kept for reference; they did not separate versions).
"""
import argparse
import json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--vqa", nargs="*", default=[], help="label=vlm_vqascore.json")
ap.add_argument("--sim", nargs="*", default=[], help="label=vlm_sim.json")
ap.add_argument("--pref", nargs="*", default=[], help="vlm_pref_<label>.json")
ap.add_argument("--out", default=None, help="write the --vqa table as json")
args = ap.parse_args()

VIEWS = ["input", "plus45", "minus45", "side90"]
table = {}
if args.vqa:
    cols = [("input", "pose"), ("input", "look"), ("plus45", "identity"), ("minus45", "identity"), ("side90", "identity"),
            ("plus45", "pose"), ("minus45", "pose"), ("side90", "pose")]
    print("mean P(Yes) over frames [frames above 0.5]")
    print(f"{'':24s}" + "".join(f"{v + ' ' + q:>18s}" for v, q in cols))
    for item in args.vqa:
        label, path = item.split("=", 1)
        pairs = json.load(open(path))["pairs"]
        acc = defaultdict(list)
        for k, v in pairs.items():
            for q, x in v.items():
                acc[(k.split("/")[0], q)].append(x)
        table[label] = {f"{v}/{q}": {"mean": round(sum(acc[(v, q)]) / len(acc[(v, q)]), 4),
                                     "above_half": sum(x > 0.5 for x in acc[(v, q)]), "frames": len(acc[(v, q)])}
                        for v, q in cols if acc.get((v, q))}
        cells = [f"{table[label][f'{v}/{q}']['mean']:.3f} [{table[label][f'{v}/{q}']['above_half']:2d}/{table[label][f'{v}/{q}']['frames']}]"
                 if f"{v}/{q}" in table[label] else "-" for v, q in cols]
        print(f"{label:24s}" + "".join(f"{c:>18s}" for c in cells))
    if args.out:
        json.dump(table, open(args.out, "w"), indent=1)
for item in args.sim:
    label, path = item.split("=", 1)
    pairs = json.load(open(path))["pairs"]
    acc = defaultdict(list)
    for k, v in pairs.items():
        if v.get("score") is not None:
            acc[k.split("/")[0]].append(v["score"])
    print(f"{label:24s} 1-10 score: " + ", ".join(f"{v} {sum(acc[v]) / len(acc[v]):.2f}" for v in VIEWS if acc[v]))
for path in args.pref:
    d = json.load(open(path))
    out = []
    for view in VIEWS:
        r = [v["result"] for k, v in d["frames"].items() if k.startswith(view + "/")]
        if r:
            out.append(f"{view}: win {r.count('win')} / loss {r.count('loss')} / tie {r.count('tie')}")
    print(f"{d['dir']} vs {d['versus']}: " + "; ".join(out))
