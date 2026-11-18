"""Portable skeletal GLB and research previews."""
import json
import struct

import numpy as np


def export_glb(path, rig, local, fps):
    from scipy.spatial.transform import Rotation
    payload = bytearray()
    views, accessors = [], []
    def add(value, kind, component=5126):
        nonlocal payload
        value = np.asarray(value, dtype="<f4" if component == 5126 else "<u4")
        while len(payload) % 4:
            payload.append(0)
        start = len(payload)
        payload.extend(value.tobytes())
        views.append({"buffer": 0, "byteOffset": start, "byteLength": value.nbytes})
        entry = {"bufferView": len(views)-1, "componentType": component,
                 "count": len(value), "type": kind}
        if kind == "VEC3":
            entry.update(min=value.min(0).tolist(), max=value.max(0).tolist())
        accessors.append(entry)
        return len(accessors)-1
    vertices = add(rig["vertices"], "VEC3")
    colors = add(rig["colors"], "VEC3")
    faces = add(rig["faces"].reshape(-1), "SCALAR", 5125)
    # glTF joints use unsigned short; add manually to avoid lossy index casts.
    ids = np.argsort(rig["weights"], axis=-1)[:, -4:]
    if ids.shape[1] < 4:
        ids = np.pad(ids, ((0, 0), (0, 4-ids.shape[1])), mode="edge")
    weights = np.take_along_axis(rig["weights"], ids, -1)
    weights /= weights.sum(-1, keepdims=True)
    while len(payload) % 4:
        payload.append(0)
    views.append({"buffer": 0, "byteOffset": len(payload), "byteLength": ids.size*2})
    payload.extend(ids.astype("<u2").tobytes())
    accessors.append({"bufferView": len(views)-1, "componentType": 5123,
                      "count": len(ids), "type": "VEC4"})
    joints = len(accessors)-1
    weight_accessor = add(weights, "VEC4")
    inverse = add(rig["inverse_bind"].transpose(0, 2, 1).reshape(-1, 16), "MAT4")
    times = add(np.arange(len(local))/fps, "SCALAR")
    accessors[times]["min"] = [0.]
    accessors[times]["max"] = [(len(local)-1)/fps]
    nodes, samplers, channels = [], [], []
    for j, parent in enumerate(rig["parents"]):
        node = {"name": f"joint_{j}"}
        children = np.flatnonzero(rig["parents"] == j).tolist()
        if children:
            node["children"] = children
        matrices = local[:, j, :3, :3]
        scale = np.linalg.norm(matrices, axis=1)
        rotation = matrices / scale[:, None, :]
        for key, data, kind in (
            ("translation", local[:, j, :3, 3], "VEC3"),
            ("rotation", Rotation.from_matrix(rotation).as_quat(), "VEC4"),
            ("scale", scale, "VEC3"),
        ):
            node[key] = data[0].tolist()
            index = add(data, kind)
            samplers.append({"input": times, "output": index, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers)-1, "target": {"node": j, "path": key}})
        nodes.append(node)
    nodes.append({"mesh": 0, "skin": 0})
    roots = np.flatnonzero(rig["parents"] < 0).tolist()+[len(nodes)-1]
    doc = {"asset": {"version": "2.0", "generator": "animimic-reproduce"},
           "buffers": [{"byteLength": len(payload)}], "bufferViews": views,
           "accessors": accessors, "nodes": nodes, "scenes": [{"nodes": roots}], "scene": 0,
           "meshes": [{"primitives": [{"attributes": {"POSITION": vertices,
               "COLOR_0": colors, "JOINTS_0": joints, "WEIGHTS_0": weight_accessor},
               "indices": faces, "material": 0}]}],
           "materials": [{"doubleSided": True, "pbrMetallicRoughness":
                          {"metallicFactor": 0, "roughnessFactor": 1}}],
           "skins": [{"joints": list(range(len(nodes)-1)), "inverseBindMatrices": inverse}],
           "animations": [{"samplers": samplers, "channels": channels}]}
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    payload.extend(b"\0" * (-len(payload) % 4))
    length = 12+8+len(encoded)+8+len(payload)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, length)
                     + struct.pack("<II", len(encoded), 0x4E4F534A)+encoded
                     + struct.pack("<II", len(payload), 0x004E4942)+payload)


def previews(directory, rig, joints, physics, renderer, reference, fps):
    import imageio.v2 as iio
    import torch
    from .render import Renderer
    for label, trajectory in (("joint", joints), ("physics", physics)):
        for azimuth in (0, -45, 45):
            view = renderer if azimuth == 0 else Renderer(
                rig["vertices"].device,
                **{**renderer.settings, "azimuth": renderer.settings["azimuth"]+azimuth})
            with iio.get_writer(directory/f"{label}_{azimuth}.mp4", fps=fps,
                                macro_block_size=1) as writer:
                with torch.no_grad():
                    for t, vertices in enumerate(trajectory):
                        rgb, _ = view(vertices, rig["faces"], rig["colors"])
                        image = (rgb.clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
                        if azimuth == 0:
                            image = np.concatenate((reference[t], image), axis=1)
                        writer.append_data(image)
