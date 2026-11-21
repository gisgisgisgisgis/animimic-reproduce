"""Bake a proper texture for a mesh that was textured by projecting frame 0 (prep_09b_fit_frame0.py).

Run in the `animamimic` env after rigging:

  python scripts/bake_texture.py --seq data/paper_dog

The projection in prep_09b gives every surface point the color of the frame-0 pixel it lands on, so the
far side of the animal copies the near side (the back of the head gets the face) and surfaces seen at a
grazing angle are smeared into streaks. Here:
1. The rigged mesh gets its own UV atlas (xatlas), so every surface point has its own texel.
2. A texel takes the frame-0 color only if it is visible in frame 0 (z-buffer), faces the camera
   (cosine above --cos_min) and lands inside the video's silhouette, away from its rim and from
   occlusion edges (--edge_px), where rim shading and small misalignments put the wrong color.
   Thin bands of texels left out this way take the nearest trusted color (--band_radius).
3. The rest is filled from the mirror image across the animal's left-right symmetry plane, which is
   found by search. Whatever the mirror does not cover keeps the plain frame-0 projection, as before.

Only frame 0 is used, the same information as before, so the paper-metric comparison stays fair.
Writes <seq>/texture_baked.png, <seq>/texture_baked.npz (uv, faces, src_idx into the rigged vertices),
<seq>/textured_rest.glb for viewing, and <seq>/texture_baked_check.png (old vs new at three views).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import xatlas
from PIL import Image
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from animamimic.baked import load_baked_mesh, rasterize, weld  # noqa: E402
from animamimic.camera import Camera  # noqa: E402
from animamimic.gltf_rig import RiggedGLB  # noqa: E402
from animamimic.render import RasterRenderer  # noqa: E402


def vertex_normals(v, f):
    fn = torch.linalg.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = torch.zeros_like(v)
    for k in range(3):
        n.index_add_(0, f[:, k], fn)
    n = torch.nn.functional.normalize(n, dim=-1)
    if ((v - v.mean(0)) * n).sum(-1).mean() < 0:  # make them point outward
        n = -n
    return n


def symmetry_plane(points, sample=4000, seed=0):
    """Vertical plane n.x = d (n horizontal) that best maps the point set onto itself."""
    tree = cKDTree(points)
    sub = points[np.random.default_rng(seed).permutation(len(points))[:sample]]

    def score(theta, d):
        n = np.array([math.cos(theta), 0.0, math.sin(theta)])
        r = sub - 2.0 * (sub @ n - d)[:, None] * n
        return tree.query(r)[0].mean()

    best = None
    for deg in range(180):
        th = math.radians(deg)
        d = float(np.median(points @ np.array([math.cos(th), 0.0, math.sin(th)])))
        s = score(th, d)
        if best is None or s < best[0]:
            best = (s, th, d)
    _, th0, d0 = best
    for th in np.radians(np.arange(-1.5, 1.51, 0.1)) + th0:
        for d in np.arange(-0.04, 0.0401, 0.002) + d0:
            s = score(th, d)
            if s < best[0]:
                best = (s, th, d)
    s, th, d = best
    return np.array([math.cos(th), 0.0, math.sin(th)]), float(d), float(s)


def tilted_symmetry_plane(points, max_tilt=30, sample=3000, seed=0):
    """Plane n.x = d that best maps the point set onto itself, with n up to max_tilt degrees out of the
    horizontal, for one body part (a head that is turned and tilted relative to the body)."""
    tree = cKDTree(points)
    sub = points[np.random.default_rng(seed).permutation(len(points))[:sample]]

    def normal(yaw, tilt):
        y, t = math.radians(yaw), math.radians(tilt)
        return np.array([math.cos(t) * math.cos(y), math.sin(t), math.cos(t) * math.sin(y)])

    def score(n, d):
        return tree.query(sub - 2.0 * (sub @ n - d)[:, None] * n)[0].mean()

    best = None
    for yaw in range(0, 180, 2):
        for tilt in range(-max_tilt, max_tilt + 1, 5):
            n = normal(yaw, tilt)
            d = float(np.median(points @ n))
            s = score(n, d)
            if best is None or s < best[0]:
                best = (s, yaw, tilt, d)
    for step_a, step_d in ((1.0, 0.004), (0.25, 0.001)):
        _, yaw0, tilt0, d0 = best
        for yaw in yaw0 + step_a * np.arange(-4, 5):
            for tilt in tilt0 + step_a * np.arange(-4, 5):
                for d in d0 + step_d * np.arange(-5, 6):
                    s = score(normal(yaw, tilt), d)
                    if s < best[0]:
                        best = (s, yaw, tilt, d)
    s, yaw, tilt, d = best
    return normal(yaw, tilt), float(d), float(s)


def midline_planes(W, dom, parents, n_g, d_g, min_frac=0.02, gain=0.7, max_angle=45.0):
    """Own symmetry planes for skeleton subtrees that sit on the midline (head, neck, tail).

    A single plane for the whole animal does not fit a head that is turned or tilted in frame 0: on the wolf
    the head's own plane is 32 degrees off the body's, the far half of the face found no mirror partner and was
    left flat gray. Every bone whose subtree straddles the global plane (25-75% of its vertices on each side)
    gets its own plane if that maps the subtree onto itself clearly better (error below `gain` x the global
    plane's) and is still a left-right plane (within `max_angle` of the global one). Thin parts such as a tail
    segment or the snout tip are nearly cylinders, whose best plane can be any plane, front-to-back included.
    Returns [(name, bone set, normal, offset, error, global error)]."""
    J = len(parents)
    kids = [[] for _ in range(J)]
    for j, p in enumerate(parents):
        if p >= 0:
            kids[p].append(j)

    def subtree(j):
        out, stack = [], [j]
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(kids[k])
        return out

    side = W @ n_g - d_g
    planes = []
    for j in range(J):
        bones = subtree(j)
        inside = np.isin(dom, bones)
        cnt = int(inside.sum())
        if cnt < min_frac * len(W) or cnt == len(W):
            continue
        frac = float((side[inside] > 0).mean())
        if not 0.25 < frac < 0.75:
            continue
        Pj = W[inside]
        n, d, err = tilted_symmetry_plane(Pj)
        g_err = float(cKDTree(Pj).query(Pj - 2.0 * (Pj @ n_g - d_g)[:, None] * n_g)[0].mean())
        angle = math.degrees(math.acos(min(1.0, abs(float(n @ n_g)))))
        if err < gain * g_err and angle <= max_angle:
            planes.append((f"bone {j}", np.array(bones), n, d, err, g_err))
    return planes


def sample_image(img, x, y):
    """Bilinear lookup in an (H, W, 3) image at pixel-center coordinates (integers are centers)."""
    return np.stack([ndimage.map_coordinates(img[..., c], [y, x], order=1, mode="nearest") for c in range(3)], -1)


def side_camera(cam, degrees):
    a = np.radians(degrees)
    Y = torch.tensor([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]], dtype=cam.R.dtype)
    return Camera(Y.T @ cam.R, cam.T, cam.fov_deg, cam.image_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--tex_size", type=int, default=2048)
    ap.add_argument("--cos_min", type=float, default=0.25, help="texels seen more obliquely than this are refilled")
    ap.add_argument("--depth_tol", type=float, default=0.005, help="z-buffer tolerance, mesh longest side = 1")
    ap.add_argument("--edge_px", type=int, default=0,
                    help="texels landing this close (frame-0 pixels) to the silhouette or to an occlusion edge are "
                         "refilled; 0 = image height / 160")
    ap.add_argument("--depth_jump", type=float, default=0.02, help="depth step that counts as an occlusion edge")
    ap.add_argument("--band_radius", type=float, default=0.025,
                    help="untrusted texels this close to a colored texel (3D) take its color")
    ap.add_argument("--mirror_radius", type=float, default=0.03)
    ap.add_argument("--pose_mirror_radius", type=float, default=0.015,
                    help="mirror radius in the straightened pose, where the plane fits about twice as well")
    ap.add_argument("--midline", type=float, default=0.015, help="texels this close to the plane are not mirrored")
    ap.add_argument("--oblique_cos", type=float, default=0.5, help="seen texels below this cosine count as oblique")
    ap.add_argument("--dark_oblique", type=float, default=0.0,
                    help="oblique texels darker than this fraction of the median around them take that median; 0 = off")
    ap.add_argument("--band_median", type=int, default=50,
                    help="band texels take the median color of every Nth colored texel nearby; 0 = nearest texel")
    ap.add_argument("--global_mirror_only", action="store_true",
                    help="one symmetry plane for the whole animal (the first version), no per-part planes")
    args = ap.parse_args()

    seq = Path(args.seq)
    T = args.tex_size
    glb = RiggedGLB(seq / "rigged.glb")
    mesh = glb.mesh()

    # 1. Weld the triangle soup and unwrap it.
    group, first = weld(mesh.verts)
    W = mesh.verts[first].float()
    Fw = group[mesh.faces.long()]
    atlas = xatlas.Atlas()
    atlas.add_mesh(W.numpy(), Fw.numpy().astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution, pack.padding, pack.bilinear = T, 4, True
    atlas.generate(pack_options=pack)
    vmap, faces, uv = atlas[0]
    vmap, faces = torch.from_numpy(vmap.astype(np.int64)), torch.from_numpy(faces.astype(np.int64))
    uv = torch.from_numpy(uv).float()
    A = W[vmap]
    src_idx = first[vmap]
    normals = vertex_normals(W, Fw)[vmap]
    print(f"atlas: {len(A)} vertices, {len(faces)} faces, {atlas.chart_count} charts")

    # 2. Texel -> surface point. Texel (row y, col x) of the sampling-space image S sits at uv = (x, y) / (T - 1),
    # matching RasterRenderer._sample_texture; the stored PNG is S flipped upside down (v points up).
    tface, tbary, _ = rasterize(uv[faces] * (T - 1), torch.zeros(len(faces), 3), T, T)
    covered = tface >= 0
    f = tface[covered]
    b = tbary[covered]
    P = (A[faces[f]] * b[..., None]).sum(1)
    N = torch.nn.functional.normalize((normals[faces[f]] * b[..., None]).sum(1), dim=-1)

    # 3. Frame-0 visibility and color.
    frame = np.asarray(Image.open(seq / "frame0.png").convert("RGB"), np.float32) / 255.0
    H = frame.shape[0]
    mask = np.asarray(Image.open(seq / "frame0_mask.png").convert("L")) > 127
    inner = ndimage.binary_erosion(mask, iterations=max(2, H // 256))
    _, (iy, ix) = ndimage.distance_transform_edt(~inner, return_indices=True)
    frame_filled = frame[iy, ix]  # background replaced by the nearest foreground color
    cam = Camera.from_json(seq / "camera.json")
    cam.image_size = H
    xy, z = cam.project(A)
    _, _, zbuf = rasterize(xy[faces] - 0.5, z[faces], H, H)

    # Pixels near the video's silhouette or near an occlusion edge of the mesh hold rim shading and, after
    # small misalignments, the color of the wrong surface (the shadow under the chin lands on the neck).
    edge_px = args.edge_px or max(3, H // 160)
    zb = zbuf.numpy()
    jump = np.zeros_like(mask)
    for axis in (0, 1):
        d = np.abs(np.diff(np.where(np.isfinite(zb), zb, 1e3), axis=axis)) > args.depth_jump
        if axis == 0:
            jump[1:] |= d
            jump[:-1] |= d
        else:
            jump[:, 1:] |= d
            jump[:, :-1] |= d
    trust = ndimage.binary_erosion(mask, iterations=edge_px) & ~ndimage.binary_dilation(jump, iterations=edge_px)
    pxy, pz = cam.project(P)
    px, py = (pxy[:, 0] - 0.5).numpy(), (pxy[:, 1] - 0.5).numpy()
    rx, ry = np.clip(np.round(px).astype(int), 0, H - 1), np.clip(np.round(py).astype(int), 0, H - 1)
    eye = -(cam.R @ cam.T)
    view = torch.nn.functional.normalize(eye[None] - P, dim=-1)
    cos = (N * view).sum(-1).numpy()
    visible = pz.numpy() <= zbuf.numpy()[ry, rx] + args.depth_tol
    seen = visible & (cos > args.cos_min) & trust[ry, rx]
    color = np.zeros((len(P), 3), np.float32)
    color[seen] = sample_image(frame_filled, px[seen], py[seen])
    stretched = np.zeros(len(P), bool)
    if args.dark_oblique:
        # A thin dark feature in frame 0 (the rim of the wolf's eye) that lands, after a small misfit between mesh
        # and image, on a surface frame 0 sees obliquely (the brow) is stretched over it, and from the front it
        # reads as a hole; the mirror then copies it to the other eye. Obliquely seen texels much darker than the
        # trusted texels around them take the median color around them instead.
        Pn0, Nn0 = P.numpy(), N.numpy()
        sid = np.flatnonzero(seen)
        sub = sid[np.random.default_rng(1).random(len(sid)) < 1.0 / 50]
        cand = np.flatnonzero(seen & (cos < args.oblique_cos))
        k = 32
        dist, idx = cKDTree(Pn0[sub]).query(Pn0[cand], k=k, distance_upper_bound=args.band_radius)
        ok = np.isfinite(dist)
        agree = np.zeros_like(ok)
        agree[ok] = (Nn0[sub[np.minimum(idx[ok], len(sub) - 1)]] * np.repeat(Nn0[cand][:, None], k, 1)[ok]).sum(-1) > 0.5
        count = agree.sum(1)
        cols = np.sort(np.where(agree[..., None], color[sub[np.minimum(idx, len(sub) - 1)]], np.inf), axis=1)
        enough = count >= 5
        mid = np.broadcast_to(((count[enough] - 1) // 2)[:, None, None], (int(enough.sum()), 1, 3))
        med = np.take_along_axis(cols[enough], mid, axis=1)[:, 0]
        c_e = cand[enough]
        dark = color[c_e].mean(1) < args.dark_oblique * med.mean(1)
        color[c_e[dark]] = med[dark]
        stretched[c_e[dark]] = True

    # 4. Mirror fill across the symmetry plane, then nearest colored texel. One plane fits the body, but a
    # midline part that is turned in frame 0 (the wolf's head, 30 degrees) does not map onto itself across it,
    # and the far half of the face found no mirror partner. So the mesh is first straightened with its own rig:
    # every such part is rotated (with linear-blend skinning, so the neck follows smoothly) until its own
    # symmetry plane lines up with the body's, and the mirror is taken in that symmetric pose. Texels whose
    # mirror image misses there fall back to the plane of the mesh as it is.
    Wn = W.numpy().astype(np.float64)
    n_sym, d_sym, sym_err = symmetry_plane(Wn)
    dom = glb.skin_idx[np.arange(len(glb.skin_idx)), glb.skin_w.argmax(1)][first.numpy()]  # per welded vertex
    planes = [] if args.global_mirror_only else midline_planes(Wn, dom, glb.parents, n_sym, d_sym)
    sk_i, sk_w = glb.skin_idx[first.numpy()], glb.skin_w[first.numpy()]
    joints = glb._normalize(glb.rest_joints).astype(np.float64)
    Wc = Wn.copy()
    for name, bones, n, d, err, g_err in sorted(planes, key=lambda q: -len(q[1])):
        print(f"{name}: own plane error {err:.4f} vs global {g_err:.4f}, "
              f"{math.degrees(math.acos(min(1.0, abs(float(n @ n_sym))))):.1f} deg off the global plane")
        n, d, _ = tilted_symmetry_plane(Wc[np.isin(dom, bones)])  # again, after outer parts were straightened
        if n @ n_sym < 0:
            n, d = -n, -d
        axis = np.cross(n, n_sym)
        sin, cos_a = np.linalg.norm(axis), float(n @ n_sym)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / max(sin, 1e-12)
        R = np.eye(3) + sin * K + (1 - cos_a) * K @ K  # rotates n onto n_sym
        c = joints[bones[0]]  # the part's root joint is the pivot
        shift = (d_sym - (d - n @ c + n_sym @ c)) * n_sym  # then slide the rotated plane onto the body's
        w = (np.isin(sk_i, bones) * sk_w).sum(1)[:, None]
        Wc = (1 - w) * Wc + w * ((Wc - c) @ R.T + c + shift)
        joints[bones] = (joints[bones] - c) @ R.T + c + shift
    Pn, Nn = P.numpy(), N.numpy()
    if planes:
        n_c, d_c, err_c = symmetry_plane(Wc)
        print(f"symmetric pose: plane error {err_c:.4f} (mesh as it is: {sym_err:.4f})")
        bw = b.numpy()[..., None]
        fv = vmap.numpy()[faces.numpy()[f.numpy()]]
        Pc = (Wc[fv] * bw).sum(1)
        Nw = vertex_normals(torch.from_numpy(Wc).float(), Fw).numpy()
        Nc = (Nw[fv] * bw).sum(1)
        Nc /= np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-12)
        # Texels mostly on a straightened part fall back to the plain plane only if they really are body.
        on_parts = (np.isin(sk_i, np.concatenate([q[1] for q in planes])) * sk_w).sum(1)
        part_w = (on_parts[fv] * bw[..., 0]).sum(1)
        del fv
        spaces = [(Pc, Nc, n_c, d_c, args.pose_mirror_radius, np.ones(len(P), bool)),
                  (Pn, Nn, n_sym, d_sym, args.mirror_radius, part_w < 0.5)]
    else:
        spaces = [(Pn, Nn, n_sym, d_sym, args.mirror_radius, np.ones(len(P), bool))]
    src = np.flatnonzero(seen)
    todo = np.flatnonzero(~seen)
    best_src = np.full(len(todo), -1)
    use_local = np.zeros(len(todo), bool)
    for k, (Ps, Ns, n, d, radius, allowed) in enumerate(spaces):
        # A texel on the midline strip mirrors almost onto itself: the mirror adds nothing and only picks the
        # nearest seen texel, between the wolf's eyes the dark inner corner of the eye. Those take the band fill.
        off_mid = np.abs(Ps[todo] @ n - d) > args.midline
        ii = np.flatnonzero((best_src < 0) & allowed[todo] & off_mid)
        Pr = Ps[todo[ii]] - 2.0 * (Ps[todo[ii]] @ n - d)[:, None] * n
        Nr = Ns[todo[ii]] - 2.0 * (Ns[todo[ii]] @ n)[:, None] * n
        dist, idx = cKDTree(Ps[seen]).query(Pr, k=6, distance_upper_bound=radius)
        ok = np.isfinite(dist)
        agree = np.zeros_like(ok)
        agree[ok] = (Ns[src[idx[ok]]] * np.repeat(Nr[:, None], 6, 1)[ok]).sum(-1) > 0.3
        pick = agree.argmax(1)
        got = agree[np.arange(len(ii)), pick]
        best_src[ii[got]] = src[idx[np.arange(len(ii)), pick][got]]
        use_local[ii[got]] = k == 0 and len(spaces) > 1
    hit = best_src >= 0
    mirrored = todo[hit]
    mirrored_by_parts = mirrored[use_local[hit]]
    by_parts = len(mirrored_by_parts)
    color[mirrored] = color[best_src[hit]]
    filled = seen.copy()
    filled[mirrored] = True

    # Neither seen nor mirrored (mostly surfaces seen edge-on, such as the face and chest from the side):
    # keep the plain projection, as prep_09b did. Its streaks run along the viewing direction, which for an
    # animal filmed from the side is the left-right axis, so they read as a symmetric extension.
    # Thin untrusted bands next to colored texels (silhouette rims, occlusion edges) take the nearest color.
    rest = np.flatnonzero(~filled)
    src2 = np.flatnonzero(filled)
    dist, idx = cKDTree(Pn[filled]).query(Pn[rest], k=8, distance_upper_bound=args.band_radius)
    ok = np.isfinite(dist)
    agree = np.zeros_like(ok)
    agree[ok] = (Nn[src2[idx[ok]]] * np.repeat(Nn[rest][:, None], 8, 1)[ok]).sum(-1) > 0.5
    pick = agree.argmax(1)
    hit = agree[np.arange(len(rest)), pick]
    banded = rest[hit]
    near = color[src2[idx[hit, pick[hit]]]]
    if args.band_median:
        # The nearest colored texel copies whatever thin feature borders the band: above the wolf's eye it was
        # the dark eye rim, which became a dark bar across the brow. Use the median color of a coarser sample
        # of colored texels around it instead (a patch about the band radius wide), when enough of them agree.
        sub = src2[np.random.default_rng(0).random(len(src2)) < 1.0 / args.band_median]
        k = 32
        dist, idx = cKDTree(Pn[sub]).query(Pn[banded], k=k, distance_upper_bound=args.band_radius)
        ok = np.isfinite(dist)
        agree = np.zeros_like(ok)
        agree[ok] = (Nn[sub[idx[ok]]] * np.repeat(Nn[banded][:, None], k, 1)[ok]).sum(-1) > 0.5
        cols = np.sort(np.where(agree[..., None], color[sub[np.minimum(idx, len(sub) - 1)]], np.inf), axis=1)
        count = agree.sum(1)
        enough = count >= 5
        mid = np.broadcast_to(((count[enough] - 1) // 2)[:, None, None], (int(enough.sum()), 1, 3))
        near[enough] = np.take_along_axis(cols[enough], mid, axis=1)[:, 0]  # per-channel median of the agreeing
    color[banded] = near
    rest = rest[~hit]
    # Still uncolored but visible in frame 0, only edge-on (the face and chest seen from the side): keep the
    # plain projection, whose streaks run along the viewing direction, i.e. roughly left-right.
    in_view = visible[rest] & (cos[rest] > 0.0) & mask[ry[rest], rx[rest]]
    projected = rest[in_view]
    color[projected] = sample_image(frame_filled, px[projected], py[projected])
    # Hidden in frame 0 and not mirrored (back of the head, inner legs): smooth fill over the surface.
    # Vertex colors come from known texels at the vertices' UVs, unknown vertices are solved by harmonic
    # (Jacobi) diffusion over the mesh edges, and hidden texels interpolate the vertex colors.
    hidden = rest[~in_view]
    known = np.ones(len(P), bool)
    known[hidden] = False
    row = torch.full((T, T), -1, dtype=torch.long)
    row[covered] = torch.arange(len(P))
    tx = (uv * (T - 1)).round().long().clamp(0, T - 1)
    k_row = row[tx[:, 1], tx[:, 0]].numpy()
    k_ok = k_row >= 0
    k_ok[k_ok] = known[k_row[k_ok]]
    G = len(W)
    vsum = np.zeros((G, 3))
    vcnt = np.zeros(G)
    np.add.at(vsum, vmap.numpy()[k_ok], color[k_row[k_ok]])
    np.add.at(vcnt, vmap.numpy()[k_ok], 1)
    fixed = vcnt > 0
    vcol = np.where(fixed[:, None], vsum / np.maximum(vcnt, 1)[:, None], vsum[fixed].sum(0) / vcnt[fixed].sum())
    e = Fw.numpy()
    ea = np.concatenate([e[:, 0], e[:, 1], e[:, 2], e[:, 1], e[:, 2], e[:, 0]])
    eb = np.concatenate([e[:, 1], e[:, 2], e[:, 0], e[:, 0], e[:, 1], e[:, 2]])
    adj = sparse.csr_matrix((np.ones(len(ea)), (ea, eb)), shape=(G, G))
    adj.data[:] = 1.0  # duplicate edges count once
    deg = np.maximum(np.asarray(adj.sum(1)).ravel(), 1.0)
    for _ in range(400):
        vcol = np.where(fixed[:, None], vcol, (adj @ vcol) / deg[:, None])
    fv = vmap.numpy()[faces.numpy()[f.numpy()[hidden]]]  # welded vertices of each hidden texel's face
    color[hidden] = (vcol[fv] * b.numpy()[hidden][..., None]).sum(1)

    # 5. Write the texture with a gutter around every chart for bilinear lookups.
    S = np.zeros((T, T, 3), np.float32)
    S[covered.numpy()] = color
    _, (gy, gx) = ndimage.distance_transform_edt(~covered.numpy(), return_indices=True)
    S = S[gy, gx]
    stored = (np.clip(S[::-1], 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(stored).save(seq / "texture_baked.png")
    # Where each texel's color came from, in the same atlas, for looking at problem spots:
    # green seen (white: seen obliquely and recolored), blue mirrored (mesh as it is), cyan mirrored (symmetric pose), yellow band, magenta edge-on
    # projection, red diffused.
    source = np.zeros((len(P), 3), np.float32)
    source[seen] = (0.1, 0.7, 0.1)
    source[stretched] = (1.0, 1.0, 1.0)
    source[mirrored] = (0.1, 0.2, 0.9)
    source[mirrored_by_parts] = (0.1, 0.8, 0.9)
    source[banded] = (0.95, 0.85, 0.1)
    source[projected] = (0.9, 0.1, 0.8)
    source[hidden] = (0.9, 0.15, 0.1)
    S2 = np.zeros((T, T, 3), np.float32)
    S2[covered.numpy()] = source
    Image.fromarray((S2[gy, gx][::-1] * 255).round().astype(np.uint8)).save(seq / "texture_baked_source.png")
    np.savez(seq / "texture_baked.npz", uv=uv.numpy(), faces=faces.numpy(), src_idx=src_idx.numpy(),
             mirror_normal=n_sym, mirror_offset=d_sym)
    tm = trimesh.Trimesh(A.numpy(), faces.numpy(), process=False,
                         visual=trimesh.visual.TextureVisuals(uv=uv.numpy(), image=Image.fromarray(stored)))
    tm.export(seq / "textured_rest.glb")

    stats = {"texels": int(covered.sum()), "seen_in_frame0": round(float(seen.mean()), 4),
             "mirror_filled": round(len(mirrored) / len(P), 4), "band_filled": round(len(banded) / len(P), 4),
             "projected_edge_on": round(len(projected) / len(P), 4), "diffused_hidden": round(len(hidden) / len(P), 4),
             "dark_oblique_recolored": round(float(stretched.mean()), 4),
             "symmetry_error": round(sym_err, 4), "mirror_normal": n_sym.round(4).tolist(),
             "mirror_filled_in_symmetric_pose": round(by_parts / len(P), 4),
             "part_planes": [{"part": name, "bones": len(bones), "normal": n.round(4).tolist(),
                              "error": round(err, 4), "global_error": round(g_err, 4)}
                             for name, bones, n, d, err, g_err in planes]}
    json.dump(stats, open(seq / "texture_baked.json", "w"), indent=1)
    print(json.dumps(stats))

    # 6. Check image: old texture (top) vs baked (bottom), input view / side 90 / -45, sharp edges, 512 px.
    baked, _ = load_baked_mesh(seq)
    cam512 = Camera.from_json(seq / "camera.json")
    cam512.image_size = 512
    rows = []
    for m in (mesh, baked):
        row = []
        for c in (cam512, side_camera(cam512, 90.0), side_camera(cam512, -45.0)):
            rgb, _ = RasterRenderer(m, c, torch.device("cpu"), sigma=1e-6)(m.verts[None])
            row.append((rgb[0].clamp(0, 1).numpy() * 255).astype(np.uint8))
        rows.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(seq / "texture_baked_check.png")
    print(f"Wrote {seq}/texture_baked.png, texture_baked.npz, textured_rest.glb, texture_baked_check.png "
          "(top: frame-0 projection, bottom: baked)")


if __name__ == "__main__":
    main()
