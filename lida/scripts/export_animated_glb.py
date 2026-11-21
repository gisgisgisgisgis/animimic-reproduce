"""Write the Stage 1 joint motion back into the original rigged .glb as a glTF animation.

The output is the input asset (mesh, texture, skin, skeleton unchanged) plus one animation that keys every
joint's local translation, rotation and scale at each video frame, so any glTF viewer or Blender plays it.

Only Stage 1 can be exported this way: Stage 2 moves vertices with the FEM simulation, which a skeleton
cannot express.

Usage:
  .venv/bin/python scripts/export_animated_glb.py --seq data/hf_cow --stage1 outputs/hf_cow/stage1.npz \
      --duration 4.0 --out outputs/hf_cow/cow_animated.glb
"""
import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from pygltflib import GLTF2, Accessor, Animation, AnimationChannel, AnimationChannelTarget, AnimationSampler, BufferView

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animamimic.gltf_rig import RiggedGLB, _node_local  # noqa: E402
from animamimic.rig import forward_kinematics  # noqa: E402


def mat_to_quat(R):
    """Rotation matrices (..., 3, 3) to quaternions (..., 4) in glTF order (x, y, z, w)."""
    m = R
    w = np.sqrt(np.clip(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], 0, None)) / 2
    x = np.sqrt(np.clip(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 0, None)) / 2
    y = np.sqrt(np.clip(1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 0, None)) / 2
    z = np.sqrt(np.clip(1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], 0, None)) / 2
    x = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    q = np.stack([x, y, z, w], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def decompose(M):
    """4x4 matrices (T, 4, 4) to translation (T, 3), quaternion (T, 4) and scale (T, 3)."""
    t = M[:, :3, 3]
    A = M[:, :3, :3]
    s = np.linalg.norm(A, axis=1)  # column norms
    R = A / s[:, None, :]
    U, _, Vh = np.linalg.svd(R)
    R = U @ Vh
    q = mat_to_quat(R)
    for i in range(1, len(q)):  # keep the sign continuous so LINEAR slerp takes the short way
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return t, q, s


def joint_world_matrices(glb: RiggedGLB, r6d, t):
    """World matrices G (T, K, 4, 4) of the joint nodes that reproduce the fitted pose in a glTF viewer.

    Our rig works in normalized coordinates N(x) = (x - c) * s, and a joint's skinning map there is
    (A_r, A_t). In the mesh's own coordinates that map is B = N^-1 A N, and glTF skins a vertex by
    G_j @ IBM_j, so G_j = W @ B_j @ inv(IBM_j), where W is the bind-pose skinning matrix (the mesh's world
    placement). W must be the same for every joint; we take it from the bind pose recorded in the IBMs.
    """
    rig = glb.rig()
    with torch.no_grad():
        A_r, A_t = forward_kinematics(rig, torch.from_numpy(r6d).float(), torch.from_numpy(t).float())
    A_r, A_t = A_r.double().numpy(), A_t.double().numpy()
    c, s = glb.center.astype(np.float64), glb.scale
    T, K = A_r.shape[:2]
    B = np.tile(np.eye(4), (T, K, 1, 1))
    B[..., :3, :3] = A_r
    B[..., :3, 3] = A_t / s + c - (A_r @ c[:, None])[..., 0]
    W = mesh_world(glb)
    return W @ B @ np.linalg.inv(glb.ibm)[None]


def node_world(glb: RiggedGLB, n):
    m = _node_local(glb.g.nodes[n])
    p = glb.parent_node.get(n)
    return m if p is None else node_world(glb, p) @ m


def mesh_world(glb: RiggedGLB):
    skinned = [i for i, n in enumerate(glb.g.nodes) if n.skin == 0 and n.mesh is not None]
    return node_world(glb, skinned[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True, help="sequence folder with the original rigged.glb")
    ap.add_argument("--stage1", required=True, help="stage1.npz with r6d and t")
    ap.add_argument("--duration", type=float, required=True, help="seconds from the first to the last fitted frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="animamimic")
    ap.add_argument("--keep_original", action="store_true", help="keep the asset's own animations after ours")
    args = ap.parse_args()

    glb = RiggedGLB(Path(args.seq) / "rigged.glb")
    g, blob = glb.g, bytearray(glb.blob)
    data = np.load(args.stage1)
    r6d, t = data["r6d"], data["t"]
    T = r6d.shape[0]
    times = np.linspace(0.0, args.duration, T).astype(np.float32)

    G = joint_world_matrices(glb, r6d, t)  # (T, K, 4, 4)
    joint_of_node = {n: j for j, n in enumerate(glb.joint_nodes)}
    W = mesh_world(glb)

    def world_at(n, i):
        """World matrix of node n at frame i: joints follow the fit, other nodes keep their rest transform."""
        if n in joint_of_node:
            return G[i, joint_of_node[n]]
        return node_world(glb, n)

    for n in glb.joint_nodes:  # a non-joint node between two joints would need its own keys
        p = glb.parent_node.get(n)
        while p is not None and p not in joint_of_node:
            assert not any(q in joint_of_node for q in _ancestors(glb, p)), f"node {p} sits between joints"
            p = glb.parent_node.get(p)

    def add_accessor(arr, kind, minmax=False):
        nonlocal blob
        arr = np.ascontiguousarray(arr, np.float32)
        while len(blob) % 4:
            blob += b"\x00"
        view = BufferView(buffer=0, byteOffset=len(blob), byteLength=arr.nbytes)
        blob += arr.tobytes()
        g.bufferViews.append(view)
        acc = Accessor(bufferView=len(g.bufferViews) - 1, componentType=5126, count=len(arr), type=kind)
        if minmax:
            acc.min, acc.max = [float(arr.min())], [float(arr.max())]
        g.accessors.append(acc)
        return len(g.accessors) - 1

    time_acc = add_accessor(times[:, None], "SCALAR", minmax=True)
    anim = Animation(name=args.name, samplers=[], channels=[])
    max_scale_dev = 0.0
    for n in glb.joint_nodes:
        p = glb.parent_node.get(n)
        local = np.stack([
            (np.linalg.inv(world_at(p, i)) if p is not None else np.eye(4)) @ world_at(n, i) for i in range(T)
        ])
        tr, q, sc = decompose(local)
        max_scale_dev = max(max_scale_dev, float(np.abs(sc - sc.mean(0)).max()))
        node = g.nodes[n]
        node.matrix = None
        node.translation, node.rotation, node.scale = tr[0].tolist(), q[0].tolist(), sc[0].tolist()
        for path, vals, kind in (("translation", tr, "VEC3"), ("rotation", q, "VEC4"), ("scale", sc, "VEC3")):
            anim.samplers.append(AnimationSampler(input=time_acc, output=add_accessor(vals, kind), interpolation="LINEAR"))
            anim.channels.append(AnimationChannel(sampler=len(anim.samplers) - 1,
                                                  target=AnimationChannelTarget(node=n, path=path)))

    g.animations = [anim] + (list(g.animations or []) if args.keep_original else [])
    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    g.save_binary(str(out))

    # Round trip: play the written animation back with our own glTF reader and compare to the fitted vertices.
    back = RiggedGLB(out)
    posed = back.sample_animation(args.name, times).numpy()  # normalized world coordinates
    Winv = np.linalg.inv(W)
    if not np.allclose(W, np.eye(4)):
        raw = posed / back.scale + back.center
        raw = raw @ Winv[:3, :3].T + Winv[:3, 3]
        posed = (raw - back.center) * back.scale
    err = np.abs(posed - data["verts"]).max() if "verts" in data else float("nan")
    print(f"wrote {out} | {T} keys over {args.duration:.2f} s, {len(glb.joint_nodes)} joints | "
          f"round-trip max vertex error {err:.2e} (normalized units) | scale drift {max_scale_dev:.1e}")


def _ancestors(glb, n):
    out, p = [], glb.parent_node.get(n)
    while p is not None:
        out.append(p)
        p = glb.parent_node.get(p)
    return out


if __name__ == "__main__":
    main()
