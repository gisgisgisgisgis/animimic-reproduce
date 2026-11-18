"""Resumable stage orchestration with independent upstream cache keys."""
import json
import subprocess
import time
import shutil
from pathlib import Path

import numpy as np

from .core import Cache, digest, file_hash, save_json
from .progress import duration


STAGES = ("prepare", "render", "video", "supervision", "joints",
          "tetrahedralize", "physics", "export")


def repo_revision(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-installed"


def run(config, force_stage=None, from_stage=None, until_stage=None):
    if until_stage not in (None, "render", "video"):
        raise ValueError("until_stage must be render or video")
    if from_stage and until_stage and STAGES.index(from_stage) > STAGES.index(until_stage):
        raise ValueError("--from-stage cannot be after --until-stage")
    print("[pipeline] Loading dependencies...", flush=True)
    import torch
    from PIL import Image
    from .cli import fetch_asset
    from .glb import load_rig, rotate_rig, load_uv_material
    from .models import sample_video, segment_frames, track_frames, estimate_depth
    from .video_backends import generate_video, backend_name
    from .optimize import tensors, fit_joints
    from .render import Renderer
    from .export import export_glb, previews
    from .evaluate import metrics

    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. GPU stages require an NVIDIA runtime.")
    asset = Path(config["asset_path"]) if "asset_path" in config else fetch_asset(
        config["asset"], "assets/downloads")
    cache = Cache(config.get("cache_root", ".cache/animimic"))
    run_id = digest({"config": config, "asset_sha256": file_hash(asset)})[:12]
    run_dir = Path(config.get("output_root", "runs"))/config["asset"]/run_id
    manifest = {"config": config, "asset_sha256": file_hash(asset), "stages": {},
                "versions": {"torch": torch.__version__,
                             "cuda": torch.version.cuda}}
    directories = {}
    source = Path(__file__).parent
    # Each stage includes only its own settings and ancestors, so optimizer changes
    # do not invalidate a saved reference video.
    def stage(name, settings, upstream, producer, modules):
        label = f"stage {STAGES.index(name)+1}/{len(STAGES)} {name}"
        print(f"[{label}] Checking cache...", flush=True)
        started = time.monotonic()
        produced = False
        def produce(out):
            nonlocal produced
            produced = True
            print(f"[{label}] Running", flush=True)
            return producer(out)
        key_inputs = {"settings": settings, "upstream": upstream,
                      "code": {m: file_hash(source/m) for m in modules}}
        must_exist = from_stage and STAGES.index(name) < STAGES.index(from_stage)
        key = digest({"stage": name, "inputs": key_inputs, "schema": 1})
        if must_exist and not (cache.root/name/key/"stage.json").is_file():
            raise RuntimeError(f"--from-stage requires cached predecessor {name}")
        manifest["stages"][name] = {"status": "running"}
        save_json(run_dir/"manifest.json", manifest)
        try:
            directory = cache.run(name, key_inputs, produce, name == force_stage)
        except Exception as exc:
            manifest["stages"][name] = {"status": "failed", "error": str(exc)}
            save_json(run_dir/"manifest.json", manifest)
            raise
        print(f"[{label}] {'Complete' if produced else 'Cache hit'} "
              f"| elapsed {duration(time.monotonic()-started)}", flush=True)
        directories[name] = directory
        record = json.loads((directory/"stage.json").read_text())
        manifest["stages"][name] = {"status": "complete", "directory": str(directory),
                                    "artifacts": record["files"]}
        save_json(run_dir/"manifest.json", manifest)
        return directory

    def prepare(out):
        data = load_rig(asset)
        if config.get("camera", {}).get("use_texture"):
            data.update(load_uv_material(asset))
        if config.get("asset_rotation"):
            data = rotate_rig(data, config["asset_rotation"])
        np.savez_compressed(out/"rig.npz", **data)
        return ["rig.npz"]
    settings = {"rotation": config["asset_rotation"]} if config.get("asset_rotation") else {}
    if config.get("camera", {}).get("use_texture"):
        settings["uv_material"] = True
    prepared = stage("prepare", settings, {"asset": file_hash(asset)}, prepare, ["glb.py"])
    raw = dict(np.load(prepared/"rig.npz"))
    rig = tensors(raw, device)
    camera = config.get("camera", {})
    renderer = Renderer(device, **camera)
    renderer.bind_material(rig)
    def render(out):
        with torch.no_grad():
            image, mask = renderer(rig["vertices"], rig["faces"], rig["colors"])
        picture = Image.fromarray((image.clamp(0, 1).cpu().numpy()*255).astype(np.uint8))
        # Keep the legacy Wan image size; LTX uses the native render without a
        # redundant enlargement and resize back down.
        if backend_name(config["video"]) == "wan":
            picture = picture.resize((1280, 704))
        picture.save(out/"canonical.png")
        np.save(out/"mask.npy", mask.cpu().numpy())
        return ["canonical.png", "mask.npy"]
    render_settings = camera if backend_name(config["video"]) == "wan" else {**camera, "native_image": True}
    rendered = stage("render", render_settings, {"rig": file_hash(prepared/"rig.npz")},
                     render, ["render.py"])
    shutil.copy2(rendered/"canonical.png", run_dir/"canonical.png")
    if until_stage == "render":
        print(f"[review] Input image: {run_dir/'canonical.png'}", flush=True)
        return run_dir
    video_config = config["video"]
    def video(out):
        if config.get("reference_video"):
            shutil.copyfile(config["reference_video"], out/"reference.mp4")
        else:
            generate_video(rendered/"canonical.png", out/"reference.mp4", video_config)
        return ["reference.mp4"]
    video_inputs = {"image": file_hash(rendered/"canonical.png")}
    backend = backend_name(video_config)
    if backend == "wan":
        video_inputs["wan_revision"] = repo_revision(video_config.get("repo", "third_party/Wan2.2"))
    else:
        video_inputs["backend"] = backend
    if config.get("reference_video"):
        video_inputs["external_video"] = file_hash(config["reference_video"])
    generated = stage("video", video_config, video_inputs, video,
                      ["models.py"] if backend == "wan" else ["video_backends.py"])
    if until_stage == "video":
        from .video_review import write_review
        write_review(rendered/"canonical.png", generated/"reference.mp4", run_dir, video_config)
        print(f"[review] Video: {run_dir/'reference.mp4'}", flush=True)
        print(f"[review] Contact sheet: {run_dir/'video-review.jpg'}", flush=True)
        print("[review] Stopped after video. Check the result before running later stages.", flush=True)
        return run_dir
    from .physics import tetrahedralize, fit_material
    def supervise(out):
        frames, indices = sample_video(generated/"reference.mp4", out/"frames",
            video_config.get("sample_frames", 25), renderer.size[::-1])
        settings = config.get("supervision", {})
        print(f"[supervision] SAM2 masks | {len(frames)} frames", flush=True)
        masks = segment_frames(out/"frames", np.load(rendered/"mask.npy")>.5, settings)
        with torch.no_grad():
            tri, beta, pixels = renderer.surface_samples(
                rig["vertices"], rig["faces"], rig["colors"], settings.get("max_tracks", 512))
        print("[supervision] AllTracker tracks", flush=True)
        tracks, visibility, confidence = track_frames(frames, pixels, settings)
        print("[supervision] VGGT depth", flush=True)
        depth_maps = estimate_depth(out/"frames", settings)
        xy = np.rint(tracks).astype(int)
        h, w = renderer.size
        in_bounds = (xy[..., 0]>=0)&(xy[..., 0]<w)&(xy[..., 1]>=0)&(xy[..., 1]<h)
        xy[..., 0] = xy[..., 0].clip(0, w-1)
        xy[..., 1] = xy[..., 1].clip(0, h-1)
        depth = depth_maps[np.arange(len(frames))[:, None], xy[..., 1], xy[..., 0]]
        visibility *= in_bounds
        if masks.reshape(len(masks), -1).sum(-1).min() == 0:
            raise ValueError("SAM2 lost the foreground; inspect/reject reference video")
        np.savez_compressed(out/"supervision.npz", frames=frames/255., masks=masks.astype(float),
            triangles=tri.cpu(), barycentric=beta.cpu(), tracks=tracks,
            visibility=visibility, confidence=confidence, depth=depth,
            depth_maps=depth_maps, indices=indices)
        return ["supervision.npz"]+[str(p.relative_to(out)) for p in sorted((out/"frames").glob("*.jpg"))]
    supervised = stage("supervision", config.get("supervision", {}),
        {"video": file_hash(generated/"reference.mp4"), "mask": file_hash(rendered/"mask.npy"),
         "alltracker_revision": repo_revision("third_party/alltracker"),
         "sam_revision": repo_revision("third_party/sam2"),
         "vggt_revision": repo_revision("third_party/vggt")},
        supervise, ["models.py", "render.py"])
    target_raw = dict(np.load(supervised/"supervision.npz"))
    target = {k: torch.as_tensor(v, device=device,
              dtype=torch.long if k in ("triangles", "indices") else torch.float32)
              for k, v in target_raw.items() if k != "depth_maps"}
    joint_dir = stage("joints", config.get("joints", {}),
        {"rig": file_hash(prepared/"rig.npz"), "supervision": file_hash(supervised/"supervision.npz")},
        lambda out: fit_joints(rig, target, renderer, out, config.get("joints", {})),
        ["rig.py", "optimize.py", "losses.py", "render.py"])
    tet_dir = stage("tetrahedralize", config.get("physics", {}),
        {"rig": file_hash(prepared/"rig.npz")},
        lambda out: tetrahedralize(raw, out, config.get("physics", {})), ["physics.py"])
    joint = dict(np.load(joint_dir/"joint_trajectory.npz"))
    indices = target_raw["indices"]
    dt = float(np.diff(indices).mean())/video_config.get("fps", 24)
    physics_dir = stage("physics", config.get("physics", {}),
        {"tet": file_hash(tet_dir/"tetrahedral.npz"), "joint": file_hash(joint_dir/"joint_trajectory.npz"),
         "supervision": file_hash(supervised/"supervision.npz")},
        lambda out: fit_material(rig, dict(np.load(tet_dir/"tetrahedral.npz")),
            torch.tensor(joint["world"], device=device), target, renderer, out,
            config.get("physics", {}), dt), ["physics.py", "fem.py", "optimize.py", "losses.py"])
    def export(out):
        export_glb(out/"joint.glb", raw, joint["local"], 1/dt)
        physical = np.load(physics_dir/"physics_trajectory.npz")["vertices"]
        trajectories = {
            "initial": rig["vertices"][None].expand(len(joint["vertices"]), -1, -1),
            "joint": torch.tensor(joint["vertices"], device=device),
            "physics": torch.tensor(physical, device=device)}
        report = {name: metrics(value, rig, target, renderer)
                  for name, value in trajectories.items()}
        save_json(out/"metrics.json", report)
        previews(out, rig, torch.tensor(joint["vertices"], device=device),
                 torch.tensor(physical, device=device), renderer,
                 (target_raw["frames"]*255).astype(np.uint8), 1/dt)
        return ["joint.glb", "metrics.json"]+[p.name for p in out.glob("*.mp4")]
    stage("export", {}, {"joint": file_hash(joint_dir/"joint_trajectory.npz"),
        "physics": file_hash(physics_dir/"physics_trajectory.npz")}, export,
        ["export.py", "evaluate.py", "render.py"])
    return run_dir
