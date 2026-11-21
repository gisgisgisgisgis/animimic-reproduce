"""Per-frame check, step 2: a vision-language model looks at every rendered frame and says whether the
animal looks normal. Run in the `vlm` env (third_party/venvs/vlm, MLX on Apple Silicon):

  third_party/venvs/vlm/bin/python tests/frame_qa_vlm.py --dir outputs/paper_dog/frame_qa --animal dog

Reads the PNGs that tests/frame_qa_render.py wrote (frames/<view>/, rest/, video/) and writes vlm.json:
for every image a score from 1 to 10, a yes/no verdict and the problem the model names. The model is
Qwen3-VL-8B-Instruct (4-bit MLX), which has nothing to do with the pipeline that made the frames.
Greedy decoding, so the same image always gets the same answer.
"""
import argparse
import json
import re
import time
from pathlib import Path

PROMPT = (
    "This image is one frame of a 3D animation of a {animal}, rendered on a white background. "
    "Judge only whether the {animal} itself looks normal and intact. Look carefully for: torn, stretched or "
    "pinched surfaces; holes or cracks; parts sticking out; missing, extra, twisted or broken legs, ears or tail; "
    "melted or blobby shapes; smeared or streaked texture; wrong colors. "
    'Reply with JSON only, no other text: {{"score": <integer 1-10, 10 = a normal, intact {animal}>, '
    '"normal": "yes" or "no", "problem": "<the most visible problem in a few words, or none>"}}'
)


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"score": int(d.get("score")), "normal": str(d.get("normal", "")).lower().strip(),
                    "problem": str(d.get("problem", ""))}
        except (ValueError, TypeError):
            pass
    s = re.search(r'"?score"?\s*[:=]\s*(\d+)', text)
    n = re.search(r'"?normal"?\s*[:=]\s*"?(yes|no)', text, re.I)
    return {"score": int(s.group(1)) if s else None, "normal": n.group(1).lower() if n else None, "problem": text.strip()[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--animal", required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    ap.add_argument("--max_tokens", type=int, default=80)
    args = ap.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    d = Path(args.dir)
    images = sorted(d.glob("frames/*/*.png")) + sorted(d.glob("rest/*.png")) + sorted(d.glob("video/*.png"))
    out_path = d / "vlm.json"
    done = json.load(open(out_path))["images"] if out_path.exists() else {}
    model, processor = load(args.model)
    prompt = apply_chat_template(processor, model.config, PROMPT.format(animal=args.animal), num_images=1)
    t0 = time.time()
    for i, p in enumerate(images):
        key = str(p.relative_to(d))
        if key in done:
            continue
        r = generate(model, processor, prompt, [str(p)], max_tokens=args.max_tokens, temperature=0.0, verbose=False)
        text = r.text if hasattr(r, "text") else str(r)
        done[key] = {**parse(text), "raw": text.strip()}
        if i % 10 == 0 or i == len(images) - 1:
            json.dump({"model": args.model, "prompt": PROMPT.format(animal=args.animal), "images": done},
                      open(out_path, "w"), indent=1)
            print(f"{i + 1}/{len(images)} {key}: {done[key]['score']} {done[key]['normal']} | {time.time() - t0:.0f}s", flush=True)
    json.dump({"model": args.model, "prompt": PROMPT.format(animal=args.animal), "images": done}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
