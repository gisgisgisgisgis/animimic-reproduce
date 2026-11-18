import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


@unittest.skipUnless(importlib.util.find_spec("pytorch3d"), "PyTorch3D unavailable")
class VideoReviewTests(unittest.TestCase):
    def test_render_and_video_stop_before_supervision_and_reuse_video(self):
        import imageio.v2 as iio
        from animimic.export import export_glb
        from animimic.pipeline import run
        rig = dict(vertices=np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0]]),
                   faces=np.array([[0, 1, 2]]), weights=np.ones((3, 1)),
                   colors=np.ones((3, 3))*.2, local=np.eye(4)[None],
                   inverse_bind=np.eye(4)[None], parents=np.array([-1]))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root/"test.glb"
            export_glb(asset, rig, np.eye(4)[None, None].repeat(2, 0), 8)
            config = dict(asset="test", asset_path=str(asset), device="cpu",
                          output_root=str(root/"runs"), cache_root=str(root/"cache"),
                          camera=dict(width=64, height=64, lighting="studio", srgb=True),
                          video=dict(backend="ltx", prompt="fixture", num_frames=9, fps=8))
            def generate(image, output, settings):
                with Image.open(image) as picture:
                    frame = np.asarray(picture.convert("RGB"))
                iio.mimwrite(str(output), [frame]*9, fps=8)
            with patch("animimic.models.segment_frames", side_effect=AssertionError("supervision ran")), \
                 patch("animimic.optimize.fit_joints", side_effect=AssertionError("joints ran")), \
                 patch("animimic.video_backends.generate_video", side_effect=generate) as generator:
                directory = run(config, until_stage="render")
                self.assertTrue((directory/"canonical.png").is_file())
                manifest = json.loads((directory/"manifest.json").read_text())
                self.assertEqual(set(manifest["stages"]), {"prepare", "render"})
                generator.assert_not_called()
                run(config, until_stage="video")
                run(config, until_stage="video")
                self.assertEqual(generator.call_count, 1)
            manifest = json.loads((directory/"manifest.json").read_text())
            self.assertEqual(set(manifest["stages"]), {"prepare", "render", "video"})
            self.assertTrue((directory/"reference.mp4").is_file())
            with Image.open(directory/"video-review.jpg") as picture:
                self.assertEqual(picture.size, (960, 612))
            review = json.loads((directory/"video-review.json").read_text())
            self.assertEqual(review["frames"], 9)
            self.assertEqual(review["status"], "needs_visual_review")

    def test_conflicting_stage_range_is_rejected(self):
        from animimic.pipeline import run
        with self.assertRaisesRegex(ValueError, "cannot be after"):
            run({}, from_stage="joints", until_stage="video")
