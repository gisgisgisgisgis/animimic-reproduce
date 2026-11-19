"""Frozen pretrained inference adapters; model results are cached by the pipeline."""
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_video(image, output, config):
    """Wan 2.2 TI2V-5B image-to-video through the official `wan` package.

    Calls WanTI2V directly rather than generate.py, which has no negative
    prompt option and fixes the remaining sampling settings. Output size
    follows the input image's aspect ratio within `max_area` pixels.
    """
    repo = Path(config.get("repo", "third_party/Wan2.2")).resolve()
    checkpoint = Path(config.get("checkpoint", "weights/Wan2.2-TI2V-5B")).resolve()
    if not (repo/"wan"/"textimage2video.py").is_file():
        raise RuntimeError(f"Install official Wan2.2 repository at {repo}")
    if not checkpoint.is_dir():
        raise RuntimeError(f"Download Wan2.2-TI2V-5B weights to {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("Wan local generation requires an accessible CUDA GPU")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import imageio
    from wan import WanTI2V
    from wan.configs import WAN_CONFIGS

    settings = WAN_CONFIGS["ti2v-5B"]
    pipeline = None
    try:
        pipeline = WanTI2V(config=settings, checkpoint_dir=str(checkpoint), device_id=0,
                           t5_cpu=config.get("t5_cpu", True),
                           convert_model_dtype=config.get("convert_model_dtype", True))
        with Image.open(image) as source:
            conditioning = source.convert("RGB")
        video = pipeline.generate(
            config["prompt"], img=conditioning,
            max_area=config.get("max_area", 704*1280),
            frame_num=config.get("num_frames", settings.frame_num),
            shift=config.get("shift", settings.sample_shift),
            sample_solver=config.get("solver", "unipc"),
            sampling_steps=config.get("num_inference_steps", settings.sample_steps),
            guide_scale=config.get("guidance_scale", settings.sample_guide_scale),
            # An empty negative prompt selects Wan's built-in default.
            n_prompt=config.get("negative_prompt", ""),
            seed=config.get("seed", 2026),
            offload_model=config.get("offload", True))
        frames = ((video.clamp(-1, 1)+1)*127.5).round().to(torch.uint8)
        frames = restore_wan_framing(frames.permute(1, 2, 3, 0).cpu().numpy(), conditioning)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(output), fps=config.get("fps", settings.sample_fps),
                                codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
    finally:
        del pipeline
        release()
    if not Path(output).is_file():
        raise RuntimeError("Wan did not produce reference video")


def restore_wan_framing(frames, image):
    """Undo Wan's center crop so frames cover the conditioning image's view.

    Wan scales the image by max(ow/iw, oh/ih) and center-crops it to its
    latent grid (e.g. 768x448 -> 1216x736 drops ~14 input pixels per side).
    Resizing such frames back to the render size would stretch the subject.
    The cropped margins are filled from the equally scaled conditioning
    image, which is exact for the static background framing the subject.
    """
    height, width = frames.shape[1:3]
    scale = max(width/image.width, height/image.height)
    canvas = np.asarray(image.convert("RGB").resize(
        (round(image.width*scale), round(image.height*scale)), Image.LANCZOS))
    x1, y1 = (canvas.shape[1]-width)//2, (canvas.shape[0]-height)//2
    restored = np.broadcast_to(canvas, (len(frames),)+canvas.shape).copy()
    restored[:, y1:y1+height, x1:x1+width] = frames
    return restored


def sample_video(path, output, count=25, size=(512, 288)):
    import imageio.v3 as iio
    frames = list(iio.imiter(path))
    if len(frames) < 2:
        raise ValueError("Reference video requires at least two frames")
    indices = np.linspace(0, len(frames)-1, min(count, len(frames))).astype(int)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sampled = []
    for t, index in enumerate(indices):
        image = Image.fromarray(frames[index]).resize(size)
        image.save(output/f"{t:05d}.jpg")
        sampled.append(np.asarray(image))
    return np.stack(sampled), indices


def segment_frames(directory, first_mask, config):
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    predictor = SAM2VideoPredictor.from_pretrained(
        config.get("sam_model", "facebook/sam2.1-hiera-large"))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(directory), offload_video_to_cpu=True)
        predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=first_mask)
        masks = {}
        for t, _, logits in predictor.propagate_in_video(state):
            masks[t] = (logits[0, 0] > 0).cpu().numpy()
    del predictor, state
    release()
    return np.stack([masks[t] for t in sorted(masks)])


def track_frames(frames, points, config):
    repo = str(Path(config.get("alltracker_repo", "third_party/alltracker")).resolve())
    sys.path.insert(0, repo)
    from nets.alltracker import Net
    model = Net(16)
    state = torch.hub.load_state_dict_from_url(
        config.get("tracker_url",
                   "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"),
        map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model = model.eval().cuda().requires_grad_(False)
    video = torch.tensor(frames, device="cuda").permute(0, 3, 1, 2)[None].float()
    with torch.inference_mode():
        flow, visconf, _, _ = model.forward_sliding(
            video, iters=4, sw=None, is_training=False)
        flow = flow.to(points.device)
        visconf = visconf.to(points.device)
        x, y = points.long().T
        trajectories = flow[0, :, :, y, x].permute(0, 2, 1) + points[None]
        vc = visconf[0, :, :, y, x].permute(0, 2, 1)
        result = (trajectories.cpu().numpy(), vc[..., 0].cpu().numpy(),
                  vc[..., 1].cpu().numpy())
    del model, state, video, flow, visconf
    release()
    return result


def estimate_depth(directory, config):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    model = VGGT.from_pretrained(config.get("depth_model", "facebook/VGGT-1B"))
    model = model.eval().cuda().requires_grad_(False)
    paths = sorted(str(p) for p in Path(directory).glob("*.jpg"))
    # Per-frame inference bounds memory and avoids treating moving objects as a rigid scene.
    maps = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for path in paths:
            images = load_and_preprocess_images([path], mode="pad").cuda()
            output = model(images)
            depth = output["depth"].reshape(1, 1, *output["depth"].shape[-3:-1])
            # Invert VGGT's exact 14-pixel-rounded resize and center padding.
            h, w = Image.open(path).size[::-1]
            dh, dw = depth.shape[-2:]
            new_w = 518 if w >= h else round(w*(518/h)/14)*14
            new_h = round(h*(518/w)/14)*14 if w >= h else 518
            top, left = (dh-new_h)//2, (dw-new_w)//2
            depth = depth[..., top:top+new_h, left:left+new_w]
            maps.append(torch.nn.functional.interpolate(
                depth.float(), size=(h, w), mode="bilinear",
                align_corners=False)[0, 0].cpu().numpy())
    del model
    release()
    return np.stack(maps)
