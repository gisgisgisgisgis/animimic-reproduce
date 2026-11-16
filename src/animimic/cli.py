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
    })
    return path


def main():
    parser = argparse.ArgumentParser(description="AniMimic reproduction tools")
    sub = parser.add_subparsers(dest="command", required=True)
    assets = sub.add_parser("assets")
    assets.add_argument("action", choices=["list", "fetch"])
    assets.add_argument("--asset", choices=ASSETS)
    assets.add_argument("--root", default="assets/downloads")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("path")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--from-stage")
    run.add_argument("--force-stage")
    args = parser.parse_args()
    if args.command == "assets":
        if args.action == "list":
            print("\n".join(ASSETS))
        elif not args.asset:
            parser.error("assets fetch requires --asset")
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
        print(run(load_config(args.config), args.force_stage, args.from_stage))
    else:
        load_config(args.config)
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
        for name, entry in [("Wan2.2", "generate.py"), ("alltracker", "nets/alltracker.py")]:
            repo = Path("third_party")/name/entry
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
