"""Embed a local glTF, its buffers and images into a reproducible GLB."""
import json
from pathlib import Path, PurePosixPath
import struct
from urllib.parse import unquote, urlparse


def pack_gltf(source, output):
    source = Path(source)
    document = json.loads(source.read_text())
    payload = bytearray()

    def read(uri):
        path = PurePosixPath(unquote(uri))
        if urlparse(uri).scheme or path.is_absolute() or ".." in path.parts:
            raise ValueError("Only local relative glTF resources are supported")
        return (source.parent/path).read_bytes()

    def append(data):
        payload.extend(b"\0" * (-len(payload) % 4))
        start = len(payload)
        payload.extend(data)
        return start

    offsets = [append(read(buffer["uri"])) for buffer in document.get("buffers", [])]
    for view in document.get("bufferViews", []):
        view["byteOffset"] = offsets[view["buffer"]] + view.get("byteOffset", 0)
        view["buffer"] = 0
    views = document.setdefault("bufferViews", [])
    for image in document.get("images", []):
        if "uri" not in image:
            continue
        uri = image.pop("uri")
        data = read(uri)
        suffix = Path(uri).suffix.lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(suffix)
        if mime is None:
            raise ValueError(f"Unsupported image format: {suffix}")
        image.update(bufferView=len(views), mimeType=mime)
        views.append({"buffer": 0, "byteOffset": append(data), "byteLength": len(data)})
    document["buffers"] = [{"byteLength": len(payload)}]
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    payload.extend(b"\0" * (-len(payload) % 4))
    Path(output).write_bytes(struct.pack("<III", 0x46546C67, 2, 28+len(encoded)+len(payload))
        + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
        + struct.pack("<II", len(payload), 0x004E4942) + payload)
