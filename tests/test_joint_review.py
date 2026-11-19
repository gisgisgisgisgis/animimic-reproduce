import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from animimic.joint_review import interpolate_local


class JointInterpolationTests(unittest.TestCase):
    def test_mid_rotation_preserves_length_instead_of_shrinking_vertices(self):
        local = np.tile(np.eye(4), (2, 2, 1, 1))
        local[:, 1, 1, 3] = -.5
        local[1, 1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        result = interpolate_local(local, [0, 1], [0, .5, 1])
        np.testing.assert_allclose(result[:, 1, :3, 3], local[0:1, 1, :3, 3].repeat(3, 0))
        midpoint = result[1, 1, :3, :3] @ [1., 0, 0]
        np.testing.assert_allclose(midpoint, [2**-.5, 2**-.5, 0])
        np.testing.assert_allclose(np.linalg.det(result[:, :, :3, :3]), 1)


@unittest.skipUnless(importlib.util.find_spec("pytorch3d"), "PyTorch3D unavailable")
class JointReviewTests(unittest.TestCase):
    def test_pipeline_stops_before_fem_and_outputs_native_video_rate(self):
        import imageio.v2 as iio
        from PIL import Image
        from animimic.core import Cache, file_hash, save_json
        from animimic.export import export_glb
        from animimic.pipeline import run
        rig = dict(vertices=np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]]),
                   faces=np.array([[0, 1, 2]]), weights=np.ones((3, 1)),
                   colors=np.ones((3, 3))*.2, local=np.eye(4)[None],
                   inverse_bind=np.eye(4)[None], parents=np.array([-1]))
        original_cache = Cache.run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root/"test.glb"
            export_glb(asset, rig, np.eye(4)[None, None].repeat(2, 0), 8)
            config = dict(asset="test", asset_path=str(asset), device="cpu",
                output_root=str(root/"runs"), cache_root=str(root/"cache"),
                camera=dict(width=32, height=32),
                video=dict(backend="ltx", prompt="fixture", num_frames=9, fps=24),
                joints=dict(root_iterations=0, geometry_iterations=1, full_iterations=0))
            def generate(image, output, settings):
                with Image.open(image) as picture:
                    frame = np.asarray(picture.convert("RGB"))
                iio.mimwrite(str(output), [frame]*9, fps=8)
            def cached_supervision(cache, stage, inputs, producer, force=False):
                if stage != "supervision":
                    return original_cache(cache, stage, inputs, producer, force)
                directory = root/"saved-supervision"
                directory.mkdir()
                np.savez(directory/"supervision.npz", frames=np.ones((3, 32, 32, 3)),
                    masks=np.ones((3, 32, 32)), triangles=np.array([0]),
                    barycentric=np.ones((1, 3))/3, tracks=np.ones((3, 1, 2))*16,
                    visibility=np.ones((3, 1)), confidence=np.ones((3, 1)),
                    depth=np.ones((3, 1)), indices=np.array([0, 4, 8]))
                save_json(directory/"stage.json", {"files": {
                    "supervision.npz": file_hash(directory/"supervision.npz")}})
                return directory
            with patch("animimic.video_backends.generate_video", side_effect=generate), \
                 patch.object(Cache, "run", cached_supervision), \
                 patch("animimic.physics.tetrahedralize") as tetrahedralize, \
                 patch("animimic.physics.fit_material") as physics:
                directory = run(config, until_stage="joints")
                tetrahedralize.assert_not_called()
                physics.assert_not_called()
            manifest = json.loads((directory/"manifest.json").read_text())
            self.assertEqual(list(manifest["stages"]), ["prepare", "render", "video", "supervision", "joints"])
            preview = directory/"joint-review"
            report = json.loads((preview/"joint-review.json").read_text())
            self.assertEqual(report["sampled_frames"], 3)
            self.assertEqual(report["output_frames"], 9)
            self.assertEqual(report["fps"], 8)
            reader = iio.get_reader(str(preview/"joint_0.mp4"))
            try:
                self.assertEqual(reader.count_frames(), 9)
                self.assertEqual(reader.get_meta_data()["fps"], 8)
            finally:
                reader.close()
            self.assertTrue((preview/"joint.glb").is_file())
