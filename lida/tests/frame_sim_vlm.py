"""Per-frame similarity to the original video, judged by a vision-language model. Run in the `vlm` env:

  third_party/venvs/vlm/bin/python tests/frame_sim_vlm.py --dir outputs/paper_wolf_sharp_rigid/frame_qa --animal wolf

For every frame t the model sees two images, frame t of the original video (<ref>/NN.png, by default <dir>/video/)
and our render of frame t (<dir>/frames/<view>/NN.png), and is asked yes/no questions. The score is the model's
probability of answering Yes (VQAScore, Lin et al. 2024), read from the first output token, so it is continuous:
- Input view (same camera as the video): "same pose?" and "same face, fur colors and markings?".
- Novel views (+/-45 degrees, side 90): the video has no frame from there, so "the same animal, seen from that
  angle?" and "the same pose, seen from that angle?".
Writes <dir>/vlm_vqascore.json (vlm_vqascore_rest.json with --rest). Same model as frame_qa_vlm.py
(Qwen3-VL-8B, 4-bit MLX).

Why probabilities: asked for a 1-10 score (--mode score, writes vlm_sim.json), the model gave the wolf video against
itself 10, our wolf render 9 and our dog render 8, and nearly every frame of every version 9. Asked to pick the
closer of two renders (--versus, writes vlm_pref_<label>.json), it picked the second image 32 times out of 32 in
both orders. P(Yes) on frame 16 of the wolf, "same pose?": video itself 0.999, the paper's result 0.78, ours 0.27,
undeformed mesh 0.10, ours with the old settings 0.03, our dog 0.01.

Baselines, laid out as render dirs:
  --extract_paper data/paper_refs/ours_5.mp4 --dir <new dir>   writes the paper's result as frames/input/NN.png,
      sampled like compare_with_paper.py (run once in `animamimic`, which has imageio), then scores it.
  --rest   scores the undeformed mesh (<dir>/rest/<view>.png) against every video frame.
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

INPUT_PROMPT = (
    "Image 1 is a frame of a video of a {animal}. Image 2 is a 3D model of the same {animal}, rendered from the "
    "same camera at the same moment, made to reproduce Image 1. Compare Image 2 with Image 1: the pose (where the "
    "head, ears, legs and tail are), the shape of the body and head, and the appearance (face, fur colors and "
    "markings). Ignore small differences in lighting and image sharpness. "
    'Reply with JSON only, no other text: {{"pose": <integer 1-10>, "appearance": <integer 1-10>, '
    '"score": <integer 1-10, 10 = the same picture>, "difference": "<the most visible difference in a few words, or none>"}}'
)
NOVEL_PROMPT = (
    "Image 1 is a frame of a video of a {animal}. Image 2 is a 3D model of that {animal} at the same moment, "
    "seen from a different camera, {deg} degrees around it, so the view itself is expected to differ. Judge "
    "whether Image 2 shows the same {animal} as Image 1, as it would look from that angle: the same face, fur "
    "colors and markings, body proportions and pose. Ignore small differences in lighting and image sharpness. "
    'Reply with JSON only, no other text: {{"identity": <integer 1-10, 10 = clearly the same animal>, '
    '"pose": <integer 1-10, 10 = the same pose seen from another angle>, "score": <integer 1-10 overall>, '
    '"difference": "<the most visible difference in a few words, or none>"}}'
)
CHOICE_PROMPT = (
    "Image 1 is a frame of a video of a {animal}. Images 2 and 3 are two 3D renders that try to reproduce Image 1 "
    "{how}. Which of Image 2 and Image 3 matches Image 1 better: the same pose (head, ears, legs, tail) and the same "
    "appearance (face, fur colors and markings){extra}? Look carefully at the differences before answering. "
    'Reply with JSON only: {{"better": 2 or 3, "reason": "<a few words>"}}'
)
DEGREES = {"plus45": "45", "minus45": "45", "side90": "90"}
QUESTIONS = {
    "input": {
        "pose": "Image 1 is a frame of a video of a {animal}. Image 2 is a 3D render made to reproduce it from the same "
                "camera. Is the {animal} in Image 2 in exactly the same pose as in Image 1, with the head, ears, legs "
                "and tail in the same places? Answer Yes or No.",
        "look": "Image 1 is a frame of a video of a {animal}. Image 2 is a 3D render made to reproduce it from the same "
                "camera. Does the {animal} in Image 2 have the same face, fur colors and markings as in Image 1? "
                "Answer Yes or No.",
    },
    "novel": {
        "identity": "Image 1 is a frame of a video of a {animal}. Image 2 shows a 3D model of that {animal} at the same "
                    "moment from a camera {deg} degrees around it. Is Image 2 the same {animal} as Image 1, with the "
                    "same face, fur colors and markings, as it would look from that angle? Answer Yes or No.",
        "pose": "Image 1 is a frame of a video of a {animal}. Image 2 shows a 3D model of that {animal} at the same "
                "moment from a camera {deg} degrees around it. Is the {animal} in Image 2 in the same pose as in "
                "Image 1, as it would look from that angle? Answer Yes or No.",
    },
}


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            out = {k: int(d[k]) for k in ("score", "pose", "appearance", "identity") if d.get(k) is not None}
            out["difference"] = str(d.get("difference", ""))
            return out
        except (ValueError, TypeError):
            pass
    out = {}
    for k in ("score", "pose", "appearance", "identity"):
        s = re.search(rf'"?{k}"?\s*[:=]\s*(\d+)', text)
        if s:
            out[k] = int(s.group(1))
    out["difference"] = text.strip()[:200]
    return out


def extract_paper(mp4, dst, ref_dir, size):
    T = len(sorted(Path(ref_dir).glob("*.png")))
    if len(list((dst / "frames" / "input").glob("*.png"))) == T:
        return  # already extracted (the vlm env has no imageio; run this step once in `animamimic`)
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image

    frames = [f for f in imageio.get_reader(mp4).iter_data()]
    pick = np.linspace(0, len(frames) - 1, T).round().astype(int)
    (dst / "frames" / "input").mkdir(parents=True, exist_ok=True)
    for t, i in enumerate(pick):
        Image.fromarray(frames[i]).convert("RGB").resize((size, size), Image.LANCZOS).save(
            dst / "frames" / "input" / f"{t:02d}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="render dir in the frame_qa_render.py layout")
    ap.add_argument("--ref", default=None, help="original video frames NN.png (default <dir>/video)")
    ap.add_argument("--animal", required=True)
    ap.add_argument("--views", default="input,plus45,minus45,side90")
    ap.add_argument("--rest", action="store_true", help="score the undeformed mesh against every frame")
    ap.add_argument("--extract_paper", default=None, help="paper result .mp4 to write into <dir> first")
    ap.add_argument("--mode", choices=("vqascore", "score"), default="vqascore")
    ap.add_argument("--versus", default=None, help="second render dir for the forced choice")
    ap.add_argument("--label", default=None, help="name of the forced-choice comparison")
    ap.add_argument("--out", default=None, help="default <dir>/vlm_sim.json (vlm_sim_rest.json with --rest)")
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    ap.add_argument("--max_tokens", type=int, default=100)
    args = ap.parse_args()

    d = Path(args.dir)
    ref = Path(args.ref) if args.ref else d / "video"
    if args.extract_paper:
        extract_paper(args.extract_paper, d, ref, 512)
        if not (d / "video").exists() and args.ref:
            shutil.copytree(ref, d / "video")
    if args.versus:
        return choose(args, d, ref)
    if args.mode == "vqascore":
        return vqascore(args, d, ref)
    out_path = Path(args.out) if args.out else d / ("vlm_sim_rest.json" if args.rest else "vlm_sim.json")
    refs = sorted(ref.glob("*.png"))
    jobs = []
    for view in args.views.split(","):
        for t, r in enumerate(refs):
            img = d / "rest" / f"{view}.png" if args.rest else d / "frames" / view / f"{t:02d}.png"
            if img.exists():
                jobs.append((f"{view}/{t:02d}", view, r, img))

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    done = json.load(open(out_path))["pairs"] if out_path.exists() else {}
    model, processor = load(args.model)
    prompts = {}
    for view in {j[1] for j in jobs}:
        text = INPUT_PROMPT.format(animal=args.animal) if view == "input" else \
            NOVEL_PROMPT.format(animal=args.animal, deg=DEGREES.get(view, "some"))
        prompts[view] = (text, apply_chat_template(processor, model.config, text, num_images=2))
    t0 = time.time()

    def save():
        json.dump({"model": args.model, "ref": str(ref), "rest": args.rest,
                   "prompts": {v: p[0] for v, p in prompts.items()}, "pairs": done}, open(out_path, "w"), indent=1)

    for i, (key, view, r, img) in enumerate(jobs):
        if key in done:
            continue
        res = generate(model, processor, prompts[view][1], [str(r), str(img)], max_tokens=args.max_tokens,
                       temperature=0.0, verbose=False)
        text = res.text if hasattr(res, "text") else str(res)
        done[key] = {**parse(text), "raw": text.strip()}
        if i % 10 == 0 or i == len(jobs) - 1:
            save()
            print(f"{i + 1}/{len(jobs)} {key}: {done[key].get('score')} | {time.time() - t0:.0f}s", flush=True)
    save()


def vqascore(args, d, ref):
    import math

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.generate import stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    out_path = Path(args.out) if args.out else d / ("vlm_vqascore_rest.json" if args.rest else "vlm_vqascore.json")
    done = json.load(open(out_path))["pairs"] if out_path.exists() else {}
    model, processor = load(args.model)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    yes = mx.array(sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")}))
    no = mx.array(sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")}))
    refs = sorted(ref.glob("*.png"))
    texts, prompts = {}, {}
    t0 = time.time()

    def save():
        json.dump({"model": args.model, "ref": str(ref), "rest": args.rest, "score": "P(Yes) of the first token",
                   "questions": texts, "pairs": done}, open(out_path, "w"), indent=1)

    for view in args.views.split(","):
        qs = QUESTIONS["input" if view == "input" else "novel"]
        for name, q in qs.items():
            texts[f"{view}/{name}"] = q.format(animal=args.animal, deg=DEGREES.get(view, "some"))
            prompts[f"{view}/{name}"] = apply_chat_template(processor, model.config, texts[f"{view}/{name}"], num_images=2)
        for t, r in enumerate(refs):
            key = f"{view}/{t:02d}"
            img = d / "rest" / f"{view}.png" if args.rest else d / "frames" / view / f"{t:02d}.png"
            if key in done or not img.exists():
                continue
            done[key] = {}
            for name in qs:
                for res in stream_generate(model, processor, prompts[f"{view}/{name}"], [str(r), str(img)],
                                           max_tokens=1, temperature=0.0):
                    ly, ln = mx.logsumexp(res.logprobs[yes]).item(), mx.logsumexp(res.logprobs[no]).item()
                    done[key][name] = round(1.0 / (1.0 + math.exp(ln - ly)), 4)
                    break
        save()
        vals = {name: [v[name] for k, v in done.items() if k.startswith(view + "/") and name in v] for name in qs}
        print(f"{view}: " + ", ".join(f"{n} {sum(x) / max(1, len(x)):.3f}" for n, x in vals.items())
              + f" | {time.time() - t0:.0f}s", flush=True)
    save()


def choose(args, d, ref):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    other = Path(args.versus)
    out_path = Path(args.out) if args.out else d / f"vlm_pref_{args.label or other.name}.json"
    done = json.load(open(out_path))["frames"] if out_path.exists() else {}
    model, processor = load(args.model)
    refs = sorted(ref.glob("*.png"))
    t0 = time.time()
    prompts = {}
    for view in args.views.split(","):
        if view == "input":
            text = CHOICE_PROMPT.format(animal=args.animal, how="from the same camera at the same moment", extra="")
        else:
            text = CHOICE_PROMPT.format(animal=args.animal, extra=", as the animal would look from that angle",
                                        how=f"at the same moment, both seen from the same different camera angle "
                                            f"({DEGREES.get(view, 'some')} degrees around it)")
        prompts[view] = (text, apply_chat_template(processor, model.config, text, num_images=3))
        for t, r in enumerate(refs):
            key = f"{view}/{t:02d}"
            a, b = d / "frames" / view / f"{t:02d}.png", other / "frames" / view / f"{t:02d}.png"
            if key in done or not (a.exists() and b.exists()):
                continue
            picks = []
            for first, second in ((a, b), (b, a)):
                res = generate(model, processor, prompts[view][1], [str(r), str(first), str(second)], max_tokens=60,
                               temperature=0.0, verbose=False)
                text_out = res.text if hasattr(res, "text") else str(res)
                m = re.search(r'"better"\s*:\s*"?(\d)', text_out)
                picks.append((int(m.group(1)) if m else None, text_out.strip()))
            ours = (picks[0][0] == 2) + (picks[1][0] == 3)
            theirs = (picks[0][0] == 3) + (picks[1][0] == 2)
            done[key] = {"result": "win" if ours == 2 else "loss" if theirs == 2 else "tie",
                         "order_ab": picks[0][1], "order_ba": picks[1][1]}
            json.dump({"model": args.model, "dir": str(d), "versus": str(other), "prompts": {v: p[0] for v, p in prompts.items()},
                       "frames": done}, open(out_path, "w"), indent=1)
        n = [v["result"] for k, v in done.items() if k.startswith(view + "/")]
        print(f"{view}: {d.name} wins {n.count('win')}, loses {n.count('loss')}, ties {n.count('tie')} "
              f"| {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
