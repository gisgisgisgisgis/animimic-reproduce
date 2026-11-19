"""Install official inference repositories into an isolated environment.

Run with the same Python used for animimic. The setup is explicit because each
upstream CUDA package has its own system/compiler requirements.
"""
import argparse
import json
import platform
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


def flash_attn_wheel_url():
    """Match the official wheel tags to this Python and CUDA PyTorch build."""
    import torch
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The FlashAttention wheel installer requires Linux x86_64")
    if torch.version.cuda is None:
        raise RuntimeError("Install CUDA PyTorch before FlashAttention")
    version = "2.8.3.post2"
    cuda = torch.version.cuda.split(".")[0]
    pytorch = ".".join(torch.__version__.split(".")[:2])
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    wheel = f"flash_attn-{version}+cu{cuda}torch{pytorch}cxx11abi{abi}-{python}-{python}-linux_x86_64.whl"
    return f"https://github.com/Dao-AILab/flash-attention/releases/download/v{version}/{wheel}"


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
            command = [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-r",
                       str(directory/"requirements.txt")]
            available = subprocess.run([sys.executable, "-c", "import flash_attn"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if available.returncode:
                # Supply a wheel directly so pip never invokes flash_attn's
                # setup.py downloader/os.rename across temporary/cache mounts.
                url = flash_attn_wheel_url()
                print(f"Installing prebuilt FlashAttention: {url}", flush=True)
                command.append(url)
            else:
                print("Using installed FlashAttention", flush=True)
            subprocess.run(command, check=True)
            subprocess.run([sys.executable, "-c",
                            "import flash_attn; print('FlashAttention:', flash_attn.__version__)"], check=True)
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
