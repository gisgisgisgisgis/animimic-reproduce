"""Track foreground pixels from frame 0 with AllTracker. Run in the AllTracker repo's env.

Follows the repo's demo.py: forward_sliding returns flow (B, T, 2, H, W) and
visconf (B, T, 2, H, W), and the demo thresholds channel 1 of visconf.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from animamimic.device import pick_device  # noqa: E402

WEIGHTS = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--repo", required=True, help="path to a clone of aharley/alltracker")
    ap.add_argument("--num_points", type=int, default=4000)
    ap.add_argument("--conf_thr", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--window_len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device={device}")
    if device.type != "cuda":
        # forward_sliding calls .cuda() directly on its image chunks.
        torch.Tensor.cuda = lambda self, *a, **k: self.to(device)

    sys.path.insert(0, str(Path(args.repo).resolve()))
    from nets.alltracker import Net

    seq = Path(args.seq)
    paths = sorted((seq / "frames").glob("*.png"))
    frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    H, W = frames.shape[1:3]
    if H % 8 or W % 8:
        raise ValueError(f"AllTracker needs sides divisible by 8, got {W}x{H}.")
    first = np.asarray(Image.open(seq / "masks" / "00000.png").convert("L")) > 127

    model = Net(args.window_len)
    state = torch.hub.load_state_dict_from_url(WEIGHTS, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    torch.set_grad_enabled(False)

    rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    flows, visconf, _, _ = model.forward_sliding(rgbs, iters=args.iters, sw=None, is_training=False)

    ys, xs = np.where(first)
    pick = np.random.default_rng(args.seed).choice(len(xs), min(args.num_points, len(xs)), replace=False)
    xq, yq = torch.from_numpy(xs[pick]).to(flows.device), torch.from_numpy(ys[pick]).to(flows.device)
    flow_at = flows[0, :, :, yq, xq].permute(0, 2, 1)  # (T, N, 2)
    query = torch.stack((xq, yq), -1).float()[None]
    # The tracker indexes pixel centers from 0. PyTorch3D screen space puts them at +0.5.
    xy = flow_at + query + 0.5
    visible = visconf[0, :, 1, yq, xq] > args.conf_thr
    visible[0] = True

    np.savez(
        seq / "tracks.npz", xy=xy.cpu().numpy().astype(np.float32), visible=visible.cpu().numpy(),
        channel0=visconf[0, :, 0, yq, xq].cpu().numpy().astype(np.float32),
        channel1=visconf[0, :, 1, yq, xq].cpu().numpy().astype(np.float32),
    )
    print(f"Wrote {xy.shape[1]} tracks over {xy.shape[0]} frames; visible fraction {float(visible.float().mean()):.2f}")


if __name__ == "__main__":
    main()
