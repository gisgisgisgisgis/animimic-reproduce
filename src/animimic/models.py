"""Frozen pretrained inference adapters; model results are cached by the pipeline."""
import gc
import json
import os
import subprocess
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
    """Use official Wan TI2V entrypoint (5B does not use CLIPVisionModel)."""
    repo = Path(config.get("repo", "third_party/Wan2.2")).resolve()
    checkpoint = Path(config.get("checkpoint", "weights/Wan2.2-TI2V-5B")).resolve()
    if not (repo/"generate.py").is_file():
        raise RuntimeError(f"Install official Wan2.2 repository at {repo}")
    from .video_backends import validate_video
    validate_video(config)
    release()
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # The official CLI restricts 5B to 704p. Its Python API supports max_area,
    # allowing a smaller latent grid without patching the upstream repository.
    if "width" in config or "height" in config or config.get("cpu_export", False):
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("wan_worker.py")),
            str(repo), str(checkpoint), str(Path(image).resolve()), str(Path(output).resolve()),
        ], input=json.dumps(config), text=True, cwd=repo, env=environment, check=True)
    else:
        command = [
            sys.executable, str(repo/"generate.py"), "--task", "ti2v-5B",
            "--size", "1280*704", "--ckpt_dir", str(checkpoint),
            "--offload_model", "True", "--convert_model_dtype", "--t5_cpu",
            "--image", str(Path(image).resolve()), "--prompt", config["prompt"],
            "--base_seed", str(config.get("seed", 2026)),
            "--frame_num", str(config.get("num_frames", 121)),
            "--save_file", str(Path(output).resolve()),
        ]
        for key, flag in (("num_inference_steps", "--sample_steps"), ("guidance_scale", "--sample_guide_scale")):
            if key in config:
                command.extend([flag, str(config[key])])
        subprocess.run(command, cwd=repo, env=environment, check=True)
    if not Path(output).is_file():
        raise RuntimeError("Wan did not produce reference video")


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
        state = predictor.init_state(str(directory), offload_video_to_cpu=True, offload_state_to_cpu=True)
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
