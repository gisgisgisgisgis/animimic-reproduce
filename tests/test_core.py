import tempfile
import unittest
from pathlib import Path

from animimic.core import Cache


class CacheTests(unittest.TestCase):
    def test_reuse_invalidation_and_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def producer(directory):
                calls.append(True)
                (directory/"video").write_bytes(b"saved-video")
                return ["video"]
            cache = Cache(root)
            a = cache.run("video", {"seed": 1}, producer)
            self.assertEqual(a, cache.run("video", {"seed": 1}, producer))
            self.assertEqual(len(calls), 1)
            cache.run("video", {"seed": 2}, producer)
            self.assertEqual(len(calls), 2)
            (a/"video").write_bytes(b"corrupt")
            cache.run("video", {"seed": 1}, producer)
            self.assertEqual(len(calls), 3)
