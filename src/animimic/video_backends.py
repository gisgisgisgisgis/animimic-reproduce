"""Selectable I2V backends. Existing Wan implementation and cache stay intact."""
from pathlib import Path

BACKENDS = ("wan", "ltx")
LTX_MODEL = "Lightricks/LTX-Video"


def backend_name(config):
    backend = config.get("backend", "wan")
    if backend not in BACKENDS:
        raise ValueError(f"Unknown video backend {backend!r}; choose {BACKENDS}")
    return backend


def validate_video(config):
    backend = backend_name(config)
    count = config.get("num_frames", 49 if backend == "ltx" else 121)
    divisor = 8 if backend == "ltx" else 4
    if not isinstance(count, int) or count < divisor+1 or (count-1) % divisor:
        raise ValueError(f"{backend} frame count must be {divisor}k+1 and at least {divisor+1}")
    if config.get("fps", 24) <= 0:
        raise ValueError("video.fps must be positive")
    if backend == "ltx" or "width" in config or "height" in config:
        defaults = (("width", 512), ("height", 288)) if backend == "ltx" else (("width", 1280), ("height", 704))
        for name, default in defaults:
            value = config.get(name, default)
            if not isinstance(value, int) or value < 32 or value % 32:
                raise ValueError(f"{backend} {name} must be a positive multiple of 32")
    if config.get("num_inference_steps", 30 if backend == "ltx" else 50) < 1:
        raise ValueError("Video inference steps must be positive")
    return backend


def generate_ltx(image, output, config):
    """Original 2B DiT, using Diffusers; CPU offload avoids keeping T5 on GPU."""
    import torch
    from PIL import Image
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video
    from .models import release

    validate_video(config)
    if not torch.cuda.is_available():
        raise RuntimeError("LTX local generation requires an accessible CUDA GPU")
    pipeline = None
    try:
        pipeline = LTXImageToVideoPipeline.from_pretrained(
            config.get("model", LTX_MODEL),
            revision=config.get("revision", "main"),
            torch_dtype=torch.bfloat16,
        )
        # Confirm the official base architecture; avoid silently loading a 13B model.
        if (pipeline.transformer.config.num_layers != 28
                or pipeline.transformer.config.num_attention_heads != 32):
            raise ValueError("Expected original LTX 2B architecture (28 layers, 32 heads)")
        pipeline.vae.enable_tiling()
        if config.get("cpu_offload", True):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to("cuda")
        width, height = config.get("width", 512), config.get("height", 288)
        with Image.open(image) as source:
            conditioning = source.convert("RGB").resize((width, height))
        generator = torch.Generator(device="cpu").manual_seed(config.get("seed", 2026))
        with torch.inference_mode():
            result = pipeline(
                image=conditioning, prompt=config["prompt"],
                negative_prompt=config.get("negative_prompt",
                    "camera movement, camera cuts, zoom, extra limbs, distorted anatomy, flicker"),
                width=width, height=height, num_frames=config.get("num_frames", 49),
                frame_rate=config.get("fps", 24),
                num_inference_steps=config.get("num_inference_steps", 30),
                guidance_scale=config.get("guidance_scale", 3.0),
                generator=generator,
            )
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        export_to_video(result.frames[0], str(output), fps=config.get("fps", 24))
    finally:
        del pipeline
        release()


def generate_video(image, output, config):
    backend = validate_video(config)
    if backend == "wan":
        from .models import generate_video as generate_wan
        return generate_wan(image, output, config)
    return generate_ltx(image, output, config)
