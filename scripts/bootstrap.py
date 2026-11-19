"""Install official inference repositories into an isolated environment.

Run with the same Python used for animimic. The setup is explicit because each
upstream CUDA package has its own system/compiler requirements.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOS = {
    "Wan2.2": "https://github.com/Wan-Video/Wan2.2.git",
    "alltracker": "https://github.com/aharley/alltracker.git",
    "sam2": "https://github.com/facebookresearch/sam2.git",
    "vggt": "https://github.com/facebookresearch/vggt.git",
    "pytorch3d": "https://github.com/facebookresearch/pytorch3d.git",
    "fTetWild": "https://github.com/wildmeshing/fTetWild.git",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clone-only", action="store_true")
    parser.add_argument("--video-backend", choices=("wan", "ltx"), default="wan",
                        help="ltx installs the Diffusers I2V extra and skips cloning Wan")
    args = parser.parse_args()
    root = Path("third_party")
    root.mkdir(exist_ok=True)
    revision_file = root/"revisions.json"
    revisions = json.loads(revision_file.read_text()) if revision_file.exists() else {}
    if args.video_backend == "ltx" and not args.clone_only:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[video]"], check=True)
    for name, url in REPOS.items():
        if name == "Wan2.2" and args.video_backend != "wan":
            continue
        directory = root/name
        if not directory.exists():
            subprocess.run(["git", "clone", "--recursive", url, str(directory)], check=True)
            if name == "pytorch3d":
                subprocess.run(["git", "-C", str(directory), "checkout", "--detach",
                                "3eac7c62fed39a09768cd08119c5e1bf4daef559"], check=True)
        revisions[name] = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
        if args.clone_only or name == "fTetWild":
            continue
        if name == "Wan2.2":
            # flash_attn imports torch while preparing its build metadata. Use
            # the active CUDA PyTorch instead of pip's temporary isolated env.
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "setuptools>=69", "wheel", "packaging", "psutil", "ninja"], check=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "-r",
                            str(directory/"requirements.txt")], check=True)
        elif name == "alltracker":
            subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                            str(directory/"requirements.txt")], check=True)
        else:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e",
                            str(directory), "--no-build-isolation"], check=True)
    revision_file.write_text(json.dumps(revisions, indent=2)+"\n")
    print("Build fTetWild with CMake; add FloatTetwild_bin to PATH.")
    if args.video_backend == "wan":
        print("Download Wan-AI/Wan2.2-TI2V-5B to weights/Wan2.2-TI2V-5B.")
    else:
        print("LTX 2B weights download automatically on first use.")


if __name__ == "__main__":
    main()
