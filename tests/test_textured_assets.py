import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from animimic.core import file_hash
from animimic.export import export_glb
from animimic.glb import load_glb, load_rig, load_uv_material, inspect_skin
from animimic.gltf_pack import pack_gltf


def fixture(root):
    rig = dict(vertices=np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]]),
               faces=np.array([[0, 1, 2]]), weights=np.ones((3, 1)),
               colors=np.ones((3, 3)), local=np.eye(4)[None],
               inverse_bind=np.eye(4)[None], parents=np.array([-1]))
    export_glb(root/"original.glb", rig, np.eye(4)[None, None].repeat(2, 0), 8)
    document, binary = load_glb(root/"original.glb")
    uv = np.array([[0., 0], [1., 0], [0., 1]], dtype="<f4")
    document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": uv.nbytes})
    document["accessors"].append({"bufferView": len(document["bufferViews"])-1,
                                  "componentType": 5126, "count": 3, "type": "VEC2"})
    document["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = len(document["accessors"])-1
    payload = bytes(binary)+uv.tobytes()
    document["buffers"] = [{"uri": "scene.bin", "byteLength": len(payload)}]
    document["materials"][0]["pbrMetallicRoughness"] = {"baseColorTexture": {"index": 0}}
    document["textures"] = [{"source": 0}]
    document["images"] = [{"uri": "coat.png"}]
    (root/"scene.bin").write_bytes(payload)
    (root/"scene.gltf").write_text(json.dumps(document))
    Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]],
                              [[0, 0, 255], [128, 128, 128]]], dtype=np.uint8)).save(root/"coat.png")
    pack_gltf(root/"scene.gltf", root/"packed.glb")


class TexturedAssetTests(unittest.TestCase):
    def test_bundle_fetch_packs_skin_and_checks_every_source(self):
        from animimic.cli import fetch_asset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            spec = dict(format="gltf", entrypoint="scene.gltf", source="fixture",
                        metadata_source="fixture", style="fixture",
                        files=[dict(path=name, url=(root/name).as_uri(), sha256=file_hash(root/name))
                               for name in ("scene.gltf", "scene.bin", "coat.png")])
            with patch.dict("animimic.cli.ASSETS", fixture=("Fixture", "unused", "CC0", "Fixture")), \
                 patch.dict("animimic.cli.ASSET_HASHES", fixture=file_hash(root/"packed.glb")), \
                 patch.dict("animimic.cli.ANIMAL_ASSETS", fixture=spec):
                path = fetch_asset("fixture", root/"download")
                self.assertEqual(file_hash(path), file_hash(root/"packed.glb"))
                rig = load_rig(path)
                self.assertEqual(inspect_skin(path)["skins"][0]["joints"], 1)
                np.testing.assert_allclose(rig["weights"].sum(-1), np.ones(3))
                material = load_uv_material(path)
                np.testing.assert_allclose(material["uv"], [[0, 1], [1, 1], [0, 0]])
                np.testing.assert_allclose(material["texture"][0, 0], [1, 0, 0])
                np.testing.assert_allclose(material["texture"][1, 1], [.2158605]*3, atol=1e-6)
                # A changed source must be rejected before producing a model.
                (root/"coat.png").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                    fetch_asset("fixture", root/"another-download")

    @unittest.skipUnless(importlib.util.find_spec("pytorch3d"), "PyTorch3D unavailable")
    def test_full_uv_texture_renders_with_finite_geometry_gradient(self):
        import torch
        from animimic.optimize import tensors
        from animimic.render import Renderer
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            raw = load_rig(root/"packed.glb")
            raw.update(load_uv_material(root/"packed.glb"))
            rig = tensors(raw, "cpu")
            rig["vertices"].requires_grad_()
            renderer = Renderer("cpu", 32, 32, lighting="unlit", srgb=True, use_texture=True)
            renderer.bind_material(rig)
            image, mask = renderer(rig["vertices"], rig["faces"], rig["colors"])
            self.assertTrue((mask>.5).any())
            self.assertGreater(float(image[mask>.5].std()), .05)
            image.sum().backward()
            self.assertTrue(torch.isfinite(rig["vertices"].grad).all())
