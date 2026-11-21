"""Read a skinned .glb: UniRig's merged output, or a rigged asset with animation."""
import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from pygltflib import GLTF2

from .rig import Rig

_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass
class MeshData:
    verts: torch.Tensor  # (V, 3)
    faces: torch.Tensor  # (F, 3)
    uv: torch.Tensor  # (V, 2), v axis points up (PyTorch3D / OBJ convention)
    texture: torch.Tensor  # (H, W, 3) in [0, 1]

    def to(self, device) -> "MeshData":
        return MeshData(self.verts.to(device), self.faces.to(device), self.uv.to(device), self.texture.to(device))


def _accessor(g, blob, index):
    a = g.accessors[index]
    bv = g.bufferViews[a.bufferView]
    dtype, n = np.dtype(_DTYPE[a.componentType]), _COUNT[a.type]
    item = dtype.itemsize * n
    start = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or item
    if stride == item:
        arr = np.frombuffer(blob, dtype, a.count * n, start).reshape(a.count, n)
    else:
        arr = np.stack([np.frombuffer(blob, dtype, n, start + i * stride) for i in range(a.count)])
    if a.normalized and np.issubdtype(dtype, np.integer):
        arr = arr.astype(np.float32) / np.iinfo(dtype).max
    return arr.copy()


def _texture(g, blob, material_index):
    fallback = np.ones((4, 4, 3), dtype=np.float32) * 0.7
    if material_index is None:
        return fallback
    material = g.materials[material_index]
    # Sketchfab exports often keep the colour in KHR_materials_pbrSpecularGlossiness (diffuseTexture).
    specgloss = (material.extensions or {}).get("KHR_materials_pbrSpecularGlossiness") or {}
    if (specgloss.get("diffuseTexture") or {}).get("index") is not None:
        image = g.images[g.textures[specgloss["diffuseTexture"]["index"]].source]
        bv = g.bufferViews[image.bufferView]
        data = blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.float32) / 255.0
    pbr = material.pbrMetallicRoughness
    if pbr is None or pbr.baseColorTexture is None:
        if pbr is not None and pbr.baseColorFactor:
            return np.ones((4, 4, 3), dtype=np.float32) * np.asarray(pbr.baseColorFactor[:3], np.float32)
        return fallback
    image = g.images[g.textures[pbr.baseColorTexture.index].source]
    bv = g.bufferViews[image.bufferView]
    data = blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.float32) / 255.0


def _trs(t, r, s):
    """TRS to a 4x4 matrix; r is a quaternion (x, y, z, w)."""
    x, y, z, w = r
    m = np.eye(4)
    m[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]) * np.asarray(s)[None, :]
    m[:3, 3] = t
    return m


def _node_local(node):
    """A node's local 4x4 transform, from its TRS or its raw column-major matrix."""
    if node.matrix:
        return np.array(node.matrix, np.float64).reshape(4, 4).T
    return _trs(*_node_trs(node))


def _node_trs(node):
    if node.matrix:
        raise ValueError("Nodes with a raw matrix are not supported; re-export as TRS.")
    return (
        np.array(node.translation or [0.0, 0.0, 0.0]),
        np.array(node.rotation or [0.0, 0.0, 0.0, 1.0]),
        np.array(node.scale or [1.0, 1.0, 1.0]),
    )


def _slerp(a, b, u):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)
    return q / np.linalg.norm(q)


class RiggedGLB:
    """Mesh, rig and optional animation of the first skin in a .glb file."""

    def __init__(self, path):
        g = GLTF2().load(str(path))
        blob = g.binary_blob()
        self.g, self.blob = g, blob
        skin = g.skins[0]
        skinned = [n for n in g.nodes if n.skin == 0 and n.mesh is not None]
        if not skinned:
            raise ValueError(f"{path}: no skinned mesh node")

        verts, faces, uvs, idx, wts, base, materials = [], [], [], [], [], 0, []
        for node in skinned:
            for prim in g.meshes[node.mesh].primitives:
                a = prim.attributes
                pos = _accessor(g, blob, a.POSITION).astype(np.float32)
                verts.append(pos)
                uvs.append(
                    _accessor(g, blob, a.TEXCOORD_0).astype(np.float32)
                    if a.TEXCOORD_0 is not None else np.zeros((len(pos), 2), np.float32)
                )
                idx.append(_accessor(g, blob, a.JOINTS_0).astype(np.int64))
                wts.append(_accessor(g, blob, a.WEIGHTS_0).astype(np.float32))
                tri = (
                    _accessor(g, blob, prim.indices).reshape(-1, 3).astype(np.int64)
                    if prim.indices is not None else np.arange(len(pos)).reshape(-1, 3)
                )
                faces.append(tri + base)
                base += len(pos)
                materials.append(prim.material)
        if len(set(materials)) > 1:
            print(f"[warn] {len(set(materials))} materials found; only the first texture is used.")

        self.joint_nodes = list(skin.joints)
        ibm = _accessor(g, blob, skin.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.parent_node = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}

        # Reorder joints so every parent precedes its children.
        node_to_old = {n: i for i, n in enumerate(self.joint_nodes)}
        order, seen = [], set()

        def visit(old):
            if old in seen:
                return
            p = node_to_old.get(self.parent_node.get(self.joint_nodes[old]))
            if p is not None:
                visit(p)
            seen.add(old)
            order.append(old)

        for old in range(len(self.joint_nodes)):
            visit(old)
        new_of_old = {old: new for new, old in enumerate(order)}
        self.joint_nodes = [self.joint_nodes[o] for o in order]
        self.ibm = ibm[order].astype(np.float64)
        node_to_new = {n: i for i, n in enumerate(self.joint_nodes)}
        self.parents = [node_to_new.get(self.parent_node.get(n), -1) for n in self.joint_nodes]

        remap = np.array([new_of_old[i] for i in range(len(order))])
        skin_idx = remap[np.concatenate(idx)]
        skin_w = np.concatenate(wts)
        skin_w = skin_w / np.clip(skin_w.sum(1, keepdims=True), 1e-8, None)

        uv = np.concatenate(uvs).copy()
        uv[:, 1] = 1.0 - uv[:, 1]  # glTF v points down, PyTorch3D v points up

        self.verts = np.concatenate(verts)
        self.faces = np.concatenate(faces)
        self.uv = uv
        self.skin_idx, self.skin_w = skin_idx, skin_w
        self.texture = _texture(g, blob, materials[0])
        self.rest_joints = np.linalg.inv(self.ibm)[:, :3, 3].astype(np.float32)
        self.center = 0.5 * (self.verts.min(0) + self.verts.max(0))
        self.scale = 1.0 / float((self.verts.max(0) - self.verts.min(0)).max())

    def _normalize(self, pts):
        return (pts - self.center) * self.scale

    def mesh(self) -> MeshData:
        return MeshData(
            torch.from_numpy(self._normalize(self.verts)).float(),
            torch.from_numpy(self.faces).long(),
            torch.from_numpy(self.uv).float(),
            torch.from_numpy(self.texture).float(),
        )

    def rig(self) -> Rig:
        return Rig(
            torch.from_numpy(self._normalize(self.rest_joints)).float(),
            list(self.parents),
            torch.from_numpy(self.skin_idx).long(),
            torch.from_numpy(self.skin_w).float(),
        )

    def animation_names(self):
        return [a.name for a in (self.g.animations or [])]

    def animation_duration(self, name) -> float:
        anim = next(a for a in self.g.animations if a.name == name)
        return max(float(_accessor(self.g, self.blob, s.input).max()) for s in anim.samplers)

    def sample_skin_transforms(self, name, times):
        """Per-joint rest-to-posed transforms in normalized coordinates: A_r (T, K, 3, 3), A_t (T, K, 3)."""
        skins = self._skin_matrices(name, times)
        c = self.center.astype(np.float64)
        A = skins[..., :3, :3]
        U, _, Vh = np.linalg.svd(A)
        A_r = U @ Vh
        A_t = self.scale * ((A @ c[None, None, ..., None])[..., 0] + skins[..., :3, 3] - c)
        return torch.from_numpy(A_r).float(), torch.from_numpy(A_t).float()

    def _skin_matrices(self, name, times):
        g, blob = self.g, self.blob
        anim = next(a for a in g.animations if a.name == name)
        tracks = {}
        for c in anim.channels:
            s = anim.samplers[c.sampler]
            tracks[(c.target.node, c.target.path)] = (
                _accessor(g, blob, s.input)[:, 0],
                _accessor(g, blob, s.output).astype(np.float64),
            )

        def value(node, path, t, default):
            if (node, path) not in tracks:
                return default
            ts, vs = tracks[(node, path)]
            t = np.clip(t, ts[0], ts[-1])
            i = int(np.clip(np.searchsorted(ts, t) - 1, 0, len(ts) - 2))
            u = float((t - ts[i]) / max(ts[i + 1] - ts[i], 1e-9))
            return _slerp(vs[i], vs[i + 1], u) if path == "rotation" else (1 - u) * vs[i] + u * vs[i + 1]

        out = []
        for t in times:
            world = {}

            def world_of(n):
                if n not in world:
                    if g.nodes[n].matrix and not any((n, p) in tracks for p in ("translation", "rotation", "scale")):
                        m = _node_local(g.nodes[n])  # static node stored as a matrix (e.g. Sketchfab's root)
                    else:
                        tn, rn, sn = _node_trs(g.nodes[n])
                        m = _trs(value(n, "translation", t, tn), value(n, "rotation", t, rn), value(n, "scale", t, sn))
                    p = self.parent_node.get(n)
                    world[n] = m if p is None else world_of(p) @ m
                return world[n]

            out.append(np.stack([world_of(n) @ self.ibm[j] for j, n in enumerate(self.joint_nodes)]))
        return np.stack(out)

    def sample_animation(self, name, times):
        """Ground-truth posed vertices (T, V, 3), in the normalized coordinates."""
        verts_h = np.c_[self.verts.astype(np.float64), np.ones(len(self.verts))]
        out = []
        for skin in self._skin_matrices(name, times):
            per = skin[self.skin_idx]  # (V, M, 4, 4)
            posed = (per @ verts_h[:, None, :, None])[..., :3, 0]
            out.append((posed * self.skin_w[..., None]).sum(1))
        return torch.from_numpy(self._normalize(np.stack(out))).float()
