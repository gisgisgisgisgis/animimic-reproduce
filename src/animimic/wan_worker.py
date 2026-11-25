"""Isolated Wan TI2V inference with a configurable pixel budget.

Uses the official Python API because its CLI only lists 704p for the 5B model.
Run by models.py so model allocations disappear when this process exits.
"""
import json
import logging
import sys
from pathlib import Path


def generate(repo, checkpoint, image, output, config):
    sys.path.insert(0, str(repo))
    import torch
    import wan
    from PIL import Image
    from wan.configs import WAN_CONFIGS
    from wan.utils.utils import save_video

    logging.basicConfig(level=logging.INFO)
    cfg = WAN_CONFIGS["ti2v-5B"]
    width, height = config.get("width", 1280), config.get("height", 704)
    count = config.get("num_frames", 121)
    print(f"[video] Wan 5B | area budget {width}x{height} | {count} frames | "
          "T5 on CPU, model offload enabled", flush=True)
    with Image.open(image) as source:
        conditioning = source.convert("RGB")
    try:
        pipeline = wan.WanTI2V(config=cfg, checkpoint_dir=str(checkpoint), device_id=0,
                              t5_cpu=True, init_on_cpu=True, convert_model_dtype=True)
        with torch.inference_mode():
            video = pipeline.generate(
                config["prompt"], img=conditioning, size=(width, height), max_area=width*height,
                frame_num=count, shift=config.get("sample_shift", cfg.sample_shift),
                sampling_steps=config.get("num_inference_steps", cfg.sample_steps),
                guide_scale=config.get("guidance_scale", cfg.sample_guide_scale),
                n_prompt=config.get("negative_prompt", "") if config.get("use_negative_prompt", True) else "",
                seed=config.get("seed", 2026),
                offload_model=True,
            )
        # Export on CPU; save_video otherwise duplicates full-resolution GPU frames.
        video = video.cpu()
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        save_video(tensor=video[None], save_file=str(output), fps=config.get("fps", 24),
                   nrow=1, normalize=True, value_range=(-1, 1))
    except torch.cuda.OutOfMemoryError:
        print("[video] Wan ran out of GPU memory. Reduce video.width/height or num_frames; "
              "check nvidia-smi for other GPU users. No automatic quality fallback was applied.",
              file=sys.stderr, flush=True)
        raise
    if not Path(output).is_file():
        raise RuntimeError("Wan did not produce reference video")


if __name__ == "__main__":
    generate(*sys.argv[1:5], json.load(sys.stdin))
