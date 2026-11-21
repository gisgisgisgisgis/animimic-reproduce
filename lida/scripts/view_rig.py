"""Draw a rig for checking: skin weights as colors on the mesh, skeleton on top, from the input camera
and from the side. Run in the `animamimic` env:

  python scripts/view_rig.py --seq data/paper_dog --out outputs/rig_views/paper_dog.png

Each vertex is colored by its skin weights (a blend of its joints' colors), so a sharp color edge is a
place where the weights jump from one joint to another across one edge. The weights are smoothed the
same way Stage 1 smooths them (`--skin_smooth`, default 30, as in the runs). The title gives the median
vertex-to-bone distance (prep_01 refuses a rig above 0.15; correct rigs measured about 0.045).
"""
import argparse
import colorsys
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.render import RasterRenderer  # noqa: E402
from animamimic.rig import bone_distance, smooth_skin_weights  # noqa: E402


def joint_colors(k):
    """Well-separated hues; consecutive joints (usually neighbors in the chain) land far apart."""
    return np.array([colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.75, 0.95) for i in range(k)])


def side_camera(cam, degrees):
    a = np.radians(degrees)
    Y = torch.tensor([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]], dtype=cam.R.dtype)
    return Camera(Y.T @ cam.R, cam.T, cam.fov_deg, cam.image_size)


def render_colors(mesh, cam, colors):
    """Rasterize per-vertex colors with the Stage 1 renderer: the UV slot carries the color and the
    texture lookup is skipped, so colors are interpolated across each face and lightly shaded."""
    r = RasterRenderer(mesh, cam, torch.device("cpu"))
    r.uv = torch.from_numpy(colors).float()
    r.wrap_uv = False
    r._sample_texture = lambda c: c
    rgb, _ = r(mesh.verts[None])
    return rgb[0].clamp(0, 1).numpy()


def draw(ax, cam, mesh, colors, joints, parents, title):
    ax.imshow(render_colors(mesh, cam, colors), extent=(0, cam.image_size, cam.image_size, 0))
    jp, _ = cam.project(joints)
    jp = jp.numpy()
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot(jp[[j, p], 0], jp[[j, p], 1], color="white", lw=3.2, solid_capstyle="round")
            ax.plot(jp[[j, p], 0], jp[[j, p], 1], color="#111", lw=1.4, solid_capstyle="round")
    ax.scatter(jp[:, 0], jp[:, 1], s=14, c="#111", edgecolors="white", linewidths=0.8, zorder=5)
    ax.set_xlim(0, cam.image_size)
    ax.set_ylim(cam.image_size, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skin_smooth", type=int, default=30)
    ap.add_argument("--label", default=None)
    ap.add_argument("--size", type=int, default=400)
    args = ap.parse_args()

    seq = Path(args.seq)
    glb = RiggedGLB(seq / "rigged.glb")
    mesh, rig = glb.mesh(), glb.rig()
    rig = smooth_skin_weights(rig, mesh.verts, mesh.faces, args.skin_smooth)
    K = rig.num_joints
    dense = torch.zeros(len(mesh.verts), K).scatter_add_(1, rig.skin_idx.long(), rig.skin_w.float())
    colors = np.clip(dense.numpy() @ joint_colors(K), 0, 1)
    offset = float(bone_distance(rig, mesh.verts).median())

    cam = Camera.from_json(seq / "camera.json", torch.device("cpu"))
    cam.image_size = args.size
    views = [("輸入視角", cam), ("側面 90°", side_camera(cam, 90.0)), ("−45°", side_camera(cam, -45.0))]
    plt.rcParams["font.family"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
    fig, axes = plt.subplots(1, len(views), figsize=(4.2 * len(views), 4.6), dpi=110)
    for ax, (name, c) in zip(axes, views):
        draw(ax, c, mesh, colors, rig.joints, rig.parents, name)
    label = args.label or seq.name
    fig.suptitle(f"{label}：{K} 個關節，{len(mesh.verts):,} 個頂點，頂點到骨頭距離中位數 {offset:.3f}", fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"{label}: joints {K}, verts {len(mesh.verts)}, median bone distance {offset:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
