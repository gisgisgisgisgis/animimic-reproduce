"""Read standard, uncompressed glTF binary skins without Blender."""
import json
import struct
from pathlib import Path

import numpy as np


def rotate_rig(rig, degrees):
    """Rotate normalized mesh and bind hierarchy together (extrinsic XYZ degrees)."""
    x, y, z = np.deg2rad(degrees)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rotation = rz @ ry @ rx
    transform = np.eye(4)
    transform[:3, :3] = rotation
    result = {k: v.copy() for k, v in rig.items()}
    result["vertices"] = result["vertices"] @ rotation.T
    roots = result["parents"] < 0
    result["local"][roots] = transform @ result["local"][roots]
    result["inverse_bind"] = result["inverse_bind"] @ np.linalg.inv(transform)
    return result


def load_glb(path):
    raw = Path(path).read_bytes()
    magic, version, size = struct.unpack_from("<III", raw)
    if magic != 0x46546C67 or version != 2 or size != len(raw):
        raise ValueError("Invalid GLB 2.0")
    offset, binary, document = 12, None, None
    while offset < size:
        length, kind = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset+8:offset+8+length]
        if kind == 0x4E4F534A:
            document = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8+length
    if document is None or binary is None:
        raise ValueError("GLB requires JSON and binary chunks")
    return document, binary


def accessor(document, binary, index):
    entry = document["accessors"][index]
    if "sparse" in entry:
        raise ValueError("Sparse accessors require conversion before import")
    view = document["bufferViews"][entry["bufferView"]]
    types = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2",
             5125: "<u4", 5126: "<f4"}
    dimensions = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    dtype = np.dtype(types[entry["componentType"]])
    width = dimensions[entry["type"]]
    start = view.get("byteOffset", 0) + entry.get("byteOffset", 0)
    stride = view.get("byteStride", width*dtype.itemsize)
    value = np.ndarray((entry["count"], width), dtype=dtype, buffer=binary,
                       offset=start, strides=(stride, dtype.itemsize)).copy()
    if entry.get("normalized") and dtype.kind in "iu":
        value = value.astype(np.float32)/np.iinfo(dtype).max
        if dtype.kind == "i":
            value = np.maximum(value, -1)
    if entry["type"] == "MAT4":
        return value.reshape(-1, 4, 4).transpose(0, 2, 1)
    return value


def inspect_skin(path):
    document, binary = load_glb(path)
    if not document.get("skins"):
        raise ValueError("Asset has no skeleton/skin")
    summary = []
    for skin in document["skins"]:
        if "inverseBindMatrices" not in skin:
            inverse = np.tile(np.eye(4), (len(skin["joints"]), 1, 1))
        else:
            inverse = accessor(document, binary, skin["inverseBindMatrices"])
        summary.append({"joints": len(skin["joints"]),
                        "inverse_bind_shape": list(inverse.shape)})
    return {"skins": summary, "meshes": len(document.get("meshes", [])),
            "animations": len(document.get("animations", []))}


def load_uv_material(path):
    """Preserve the full base-color map for a single-primitive textured asset."""
    from io import BytesIO
    from PIL import Image
    document, binary = load_glb(path)
    primitives = [p for mesh in document["meshes"] for p in mesh["primitives"]]
    if len(primitives) != 1:
        raise ValueError("UV rendering currently supports a single mesh primitive")
    primitive = primitives[0]
    pbr = document["materials"][primitive.get("material", 0)]["pbrMetallicRoughness"]
    reference = pbr["baseColorTexture"]
    uv = accessor(document, binary, primitive["attributes"][f"TEXCOORD_{reference.get('texCoord', 0)}"]).copy()
    if (uv < 0).any() or (uv > 1).any():
        raise ValueError("UV rendering currently requires coordinates in [0, 1]")
    uv[:, 1] = 1-uv[:, 1]  # glTF image origin is at the top; TexturesUV uses bottom.
    image = document["images"][document["textures"][reference["index"]]["source"]]
    view = document["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    with Image.open(BytesIO(binary[start:start+view["byteLength"]])) as picture:
        srgb = np.asarray(picture.convert("RGB"), dtype=np.float32)/255
    linear = np.where(srgb <= .04045, srgb/12.92, ((srgb+.055)/1.055)**2.4)
    linear *= np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3])
    return {"uv": uv, "texture": linear}


def load_rig(path):
    """Bake mesh-node transforms and retain all ancestor nodes in the rig."""
    document, binary = load_glb(path)
    nodes = document["nodes"]
    parents = [-1] * len(nodes)
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = i
    order, active = [], set()
    def visit(i):
        if i in active:
            raise ValueError("Cyclic node hierarchy")
        active.add(i)
        order.append(i)
        for child in nodes[i].get("children", []):
            visit(child)
    for i, parent in enumerate(parents):
        if parent < 0:
            visit(i)
    remap = {old: new for new, old in enumerate(order)}
    local, world = [], []
    for i in order:
        node = nodes[i]
        if "matrix" in node:
            matrix = np.array(node["matrix"]).reshape(4, 4).T
        else:
            x, y, z, w = node.get("rotation", [0, 0, 0, 1])
            r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            matrix = np.eye(4)
            matrix[:3, :3] = r @ np.diag(node.get("scale", [1, 1, 1]))
            matrix[:3, 3] = node.get("translation", [0, 0, 0])
        local.append(matrix)
        world.append(matrix if parents[i] < 0 else world[remap[parents[i]]] @ matrix)
    verts, faces, weights, colors = [], [], [], []
    count = 0
    for i, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        mesh_world = world[remap[i]]
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            if primitive.get("mode", 4) != 4:
                raise ValueError("Only triangle primitives supported")
            attrs = primitive["attributes"]
            positions = accessor(document, binary, attrs["POSITION"])
            transformed = np.c_[positions, np.ones(len(positions))] @ mesh_world.T
            verts.append(transformed[:, :3])
            indices = (accessor(document, binary, primitive["indices"]).ravel()
                       if "indices" in primitive else np.arange(len(positions)))
            faces.append(indices.reshape(-1, 3) + count)
            weight = np.zeros((len(positions), len(nodes)))
            if "skin" in node:
                skin = document["skins"][node["skin"]]
                for suffix in ("0", "1"):
                    if "JOINTS_"+suffix not in attrs:
                        continue
                    ids = accessor(document, binary, attrs["JOINTS_"+suffix]).astype(int)
                    values = accessor(document, binary, attrs["WEIGHTS_"+suffix])
                    for column in range(ids.shape[1]):
                        mapped = np.array([remap[skin["joints"][j]] for j in ids[:, column]])
                        np.add.at(weight, (np.arange(len(positions)), mapped), values[:, column])
            else:
                weight[:, remap[i]] = 1
            if (weight.sum(-1) <= 0).any() or (weight < 0).any():
                raise ValueError("Every vertex must have positive, nonnegative skin weights")
            weight /= weight.sum(-1, keepdims=True)
            weights.append(weight)
            material = document.get("materials", [{}])[primitive.get("material", 0)]
            pbr = material.get("pbrMetallicRoughness", {})
            legacy = material.get("extensions", {}).get("KHR_materials_pbrSpecularGlossiness")
            if legacy:
                pbr = {**pbr, "baseColorFactor": legacy.get("diffuseFactor", [1, 1, 1, 1])}
                if "diffuseTexture" in legacy:
                    pbr["baseColorTexture"] = legacy["diffuseTexture"]
            color = np.tile(np.asarray(pbr.get("baseColorFactor", [0.7, 0.7, 0.7, 1])[:3],
                                       dtype=float), (len(positions), 1))
            if "COLOR_0" in attrs:
                color *= accessor(document, binary, attrs["COLOR_0"])[:, :3]
            if "baseColorTexture" in pbr and "TEXCOORD_0" in attrs:
                from PIL import Image
                from io import BytesIO
                tex = document["textures"][pbr["baseColorTexture"]["index"]]
                image = document["images"][tex["source"]]
                if "bufferView" in image:
                    view = document["bufferViews"][image["bufferView"]]
                    start = view.get("byteOffset", 0)
                    pixels = np.array(Image.open(BytesIO(
                        binary[start:start+view["byteLength"]])).convert("RGB")) / 255
                    uv = accessor(document, binary, attrs["TEXCOORD_0"])
                    xy = np.clip(uv, 0, 1)
                    color *= pixels[(xy[:, 1]*(pixels.shape[0]-1)).astype(int),
                                    (xy[:, 0]*(pixels.shape[1]-1)).astype(int)]
            colors.append(color)
            count += len(positions)
    if not verts:
        raise ValueError("No triangle meshes")
    vertices = np.concatenate(verts)
    center = (vertices.min(0)+vertices.max(0))/2
    scale = np.linalg.norm(vertices.max(0)-vertices.min(0))
    normal = np.eye(4)
    normal[:3, :3] /= scale
    normal[:3, 3] = -center/scale
    for j, old in enumerate(order):
        if parents[old] < 0:
            local[j] = normal @ local[j]
    world = np.array([normal @ m for m in world])
    # Recover the actual bind skeleton from inverse-bind matrices. Many assets
    # store an animation pose in default node transforms, not the bind pose.
    assigned = {}
    for mesh_index, node in enumerate(nodes):
        if "skin" not in node or "mesh" not in node:
            continue
        skin = document["skins"][node["skin"]]
        if "inverseBindMatrices" in skin:
            ibm = accessor(document, binary, skin["inverseBindMatrices"])
            mesh_global = np.linalg.inv(normal) @ world[remap[mesh_index]]
            for k, joint in enumerate(skin["joints"]):
                bind_world = normal @ mesh_global @ np.linalg.inv(ibm[k])
                mapped = remap[joint]
                if mapped in assigned and not np.allclose(assigned[mapped], bind_world, atol=2e-3):
                    raise ValueError("Multiple skins have incompatible bind frames; bake them before import")
                assigned[mapped] = bind_world
    for joint, matrix in assigned.items():
        world[joint] = matrix
    for j, old in enumerate(order):
        parent = parents[old]
        local[j] = world[j] if parent < 0 else np.linalg.solve(world[remap[parent]], world[j])
    return dict(vertices=(vertices-center)/scale, faces=np.concatenate(faces),
                weights=np.concatenate(weights), colors=np.concatenate(colors),
                local=np.array(local), inverse_bind=np.linalg.inv(world),
                parents=np.array([-1 if parents[i] < 0 else remap[parents[i]] for i in order]))
