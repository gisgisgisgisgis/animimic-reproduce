import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from animimic.core import Cache, load_config
from animimic.video_backends import backend_name, generate_video, validate_video


class VideoBackendTests(unittest.TestCase):
    def test_legacy_config_keeps_wan(self):
        self.assertEqual(backend_name(load_config("configs/fox.yaml")["video"]), "wan")
        with patch("animimic.models.generate_video") as wan:
            generate_video("input.png", "output.mp4", {"prompt": "walk"})
            wan.assert_called_once_with("input.png", "output.mp4", {"prompt": "walk"})

    def test_ltx_routes_without_wan(self):
        with patch("animimic.video_backends.generate_ltx") as ltx:
            config = {"backend": "ltx", "prompt": "walk", "num_frames": 49}
            generate_video("input.png", "output.mp4", config)
            ltx.assert_called_once_with("input.png", "output.mp4", config)

    def test_temporal_and_spatial_constraints_are_backend_specific(self):
        self.assertEqual(validate_video({"num_frames": 5}), "wan")
        with self.assertRaises(ValueError):
            validate_video({"backend": "ltx", "num_frames": 5})
        with self.assertRaises(ValueError):
            validate_video({"backend": "ltx", "width": 500})
        with self.assertRaises(ValueError):
            validate_video({"backend": "unrecognized"})
        self.assertEqual(validate_video(load_config("configs/fox-ltx.yaml")["video"]), "ltx")

    def test_different_backend_gets_separate_cache_and_identical_input_reuses(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def produce(path):
                calls.append(path)
                (path/"reference.mp4").write_bytes(b"saved reference")
                return ["reference.mp4"]
            cache = Cache(root)
            wan = cache.run("video", {"backend": "wan", "seed": 1}, produce)
            ltx = cache.run("video", {"backend": "ltx", "seed": 1}, produce)
            again = cache.run("video", {"backend": "ltx", "seed": 1}, produce)
            self.assertNotEqual(wan, ltx)
            self.assertEqual(ltx, again)
            self.assertEqual(len(calls), 2)
