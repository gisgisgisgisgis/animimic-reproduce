import contextlib
import io
import unittest
from unittest.mock import patch

from animimic.progress import Progress, duration


class ProgressTests(unittest.TestCase):
    def test_iteration_eta_and_throttled_details(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch(
                "animimic.progress.time.monotonic", side_effect=[0, 5, 10, 20, 40]):
            progress = Progress("joints 1/3 root", 2)
            progress.detail("too early")
            progress.detail("rendering frame 2/25")
            progress.advance(.5)
            progress.advance(.25)
        log = output.getvalue()
        self.assertNotIn("too early", log)
        self.assertIn("rendering frame 2/25", log)
        self.assertIn("1/2 (50.0%) | loss=0.5 | elapsed 00:00:20 | ETA 00:00:20", log)
        self.assertIn("2/2 (100.0%) | loss=0.25 | elapsed 00:00:40 | ETA 00:00:00", log)

    def test_duration_long_run(self):
        self.assertEqual(duration(3661), "01:01:01")
