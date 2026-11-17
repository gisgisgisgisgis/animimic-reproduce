import argparse
import importlib.util
import json
import shutil
import urllib.request
from pathlib import Path

from .core import file_hash, load_config, save_json
from .glb import inspect_skin


ASSETS = {
    "fox": ("Fox", "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Fox/glTF-Binary/Fox.glb",
            "CC BY 4.0", "PixelMannen, tomkranis, AsoboStudio and scurest"),
    "humanoid": ("CesiumMan", "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CesiumMan/glTF-Binary/CesiumMan.glb",
                 "CC BY 4.0; Cesium logo trademark reserved", "Cesium"),
    "robot": ("RobotExpressive", "https://raw.githubusercontent.com/mrdoob/three.js/dev/examples/models/gltf/RobotExpressive/RobotExpressive.glb",
              "CC0 1.0", "Tomas Laulhe (Quaternius), Don McCurdy"),
}


ASSET_HASHES = {
    "fox": "d97044e701822bac5a62696459b27d7b375aada5de8574ed4362edbba94771f7",
    "humanoid": "b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9",
    "robot": "047f5e5fb3bb6d378bd1df16ca6137f2a596c99b3a1b5690b4020c05aaf6f319",
}

ANIMAL_ASSETS = json.loads(Path(__file__).with_name("animal_assets.json").read_text())
for _name, _spec in ANIMAL_ASSETS.items():
    ASSETS[_name] = (_spec["name"], _spec["url"], _spec["license"], _spec["credit"])
    ASSET_HASHES[_name] = _spec["sha256"]


def fetch_asset(name, root):
    display, url, license, credit = ASSETS[name]
    directory = Path(root)/name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/"model.glb"
    if not path.exists():
        temporary = path.with_suffix(".partial")
        request = urllib.request.Request(url, headers={"User-Agent": "animimic-reproduce/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        inspect_skin(temporary)
        if file_hash(temporary) != ASSET_HASHES[name]:
            raise ValueError(f"{name}: upstream asset changed; review its checksum before use")
        temporary.replace(path)
    if file_hash(path) != ASSET_HASHES[name]:
        raise ValueError(f"{name}: cached asset checksum mismatch")
    save_json(directory/"asset.json", {
        "name": display, "url": url, "sha256": file_hash(path),
        "license": license,
        "credit": f"{display} by {credit}, {license}",
        "skin": inspect_skin(path),
        **({"source": ANIMAL_ASSETS[name]["source"],
            "metadata_source": ANIMAL_ASSETS[name]["metadata_source"],
            "style": ANIMAL_ASSETS[name]["style"]} if name in ANIMAL_ASSETS else {}),
    })
    return path


def main():
    parser = argparse.ArgumentParser(description="AniMimic reproduction tools")
    sub = parser.add_subparsers(dest="command", required=True)
    assets = sub.add_parser("assets")
    assets.add_argument("action", choices=["list", "fetch", "preview"])
    assets.add_argument("--asset", choices=ASSETS)
    assets.add_argument("--root", default="assets/downloads")
    assets.add_argument("--output", default="runs/asset-previews")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("path")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--from-stage")
    run.add_argument("--force-stage")
    run.add_argument("--video-backend", choices=("wan", "ltx"))
    videos = sub.add_parser("video")
    videos.add_argument("--config", required=True)
    videos.add_argument("--image", required=True)
    videos.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "assets":
        if args.action == "list":
            for name in ASSETS:
                if name in ANIMAL_ASSETS:
                    spec = ANIMAL_ASSETS[name]
                    print(f"{name:12} animal   {spec['faces']:6} triangles  {spec['style']}")
                else:
                    style = "low-poly animal debug" if name == "fox" else "human/robot debug"
                    print(f"{name:12} debug           {style}")
        elif not args.asset:
            parser.error("assets fetch/preview requires --asset")
        elif args.action == "preview":
            from .asset_preview import preview_asset
            print(preview_asset(fetch_asset(args.asset, args.root), args.output, args.asset))
        else:
            print(fetch_asset(args.asset, args.root))
    elif args.command == "inspect":
        path = Path(args.path)
        result = json.loads((path/"manifest.json").read_text()) if path.is_dir() else inspect_skin(path)
        print(json.dumps(result, indent=2))
    elif args.command == "run":
        from .pipeline import run, STAGES
        for name in (args.from_stage, args.force_stage):
            if name and name not in STAGES:
                parser.error(f"Unknown stage {name}")
        config = load_config(args.config)
        if args.video_backend:
            current = config.setdefault("video", {}).get("backend", "wan")
            if args.video_backend != current:
                # Backend-specific model IDs and generation settings cannot be reused.
                keep = ("prompt", "negative_prompt", "seed", "sample_frames")
                original = config["video"]
                config["video"] = {k: original[k] for k in keep if k in original}
                config["video"]["backend"] = args.video_backend
        print(run(config, args.force_stage, args.from_stage))
    elif args.command == "video":
        from .video_backends import generate_video
        from .core import Cache
        config = load_config(args.config)
        from .pipeline import repo_revision
        from .video_backends import backend_name
        video = config["video"]
        backend = backend_name(video)
        inputs = {"image": file_hash(args.image), "config": video, "backend": backend,
                  "code": file_hash(Path(__file__).with_name("video_backends.py"))}
        if backend == "wan":
            inputs["revision"] = repo_revision(video.get("repo", "third_party/Wan2.2"))
        def produce(directory):
            generate_video(args.image, directory/"reference.mp4", video)
            return ["reference.mp4"]
        directory = Cache(config.get("cache_root", ".cache/animimic")).run(
            "standalone-video", inputs, produce)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.resolve() != (directory/"reference.mp4").resolve():
            shutil.copyfile(directory/"reference.mp4", output)
        print(output)
    else:
        missing = []
        config = load_config(args.config)
        for module in ["torch", "pytorch3d", "sam2", "vggt", "meshio", "imageio"]:
            ok = importlib.util.find_spec(module) is not None
            print(f"{module}: {'installed' if ok else 'missing'}")
            if not ok:
                missing.append(module)
        print(f"fTetWild: {shutil.which('FloatTetwild_bin') or 'missing'}")
        if not shutil.which("FloatTetwild_bin"):
            missing.append("fTetWild")
        from .video_backends import backend_name
        entries = [("alltracker", Path("third_party/alltracker/nets/alltracker.py"))]
        if backend_name(config.get("video", {})) == "wan" and not config.get("reference_video"):
            entries.append(("Wan2.2", Path(config.get("video", {}).get(
                "repo", "third_party/Wan2.2"))/"generate.py"))
        elif backend_name(config.get("video", {})) == "ltx" and not config.get("reference_video"):
            for module in ("diffusers", "transformers", "accelerate", "sentencepiece"):
                ok = importlib.util.find_spec(module) is not None
                print(f"{module}: {'installed' if ok else 'missing'}")
                if not ok:
                    missing.append(module)
        for name, repo in entries:
            print(f"{name}: {'present' if repo.exists() else 'missing'}")
            if not repo.exists():
                missing.append(name)
        try:
            import torch
            print(f"CUDA: {torch.cuda.is_available()}")
            if not torch.cuda.is_available():
                missing.append("CUDA")
        except ImportError:
            pass
        if missing:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
