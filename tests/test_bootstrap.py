import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def test_wan_build_uses_current_python_after_installing_build_dependencies(self):
        spec = importlib.util.spec_from_file_location("bootstrap", ROOT/"scripts/bootstrap.py")
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"third_party/Wan2.2").mkdir(parents=True)
            (root/"third_party/alltracker").mkdir()
            calls = []
            def install(command, **kwargs):
                calls.append(command)
                if command[-1] == "third_party/Wan2.2/requirements.txt":
                    # Reproduce the reported failure when isolation hides torch.
                    if "--no-build-isolation" not in command:
                        raise ModuleNotFoundError("No module named 'torch'")
                    self.assertEqual(command[:4], [sys.executable, "-m", "pip", "install"])
                    prerequisites = calls[-2]
                    self.assertEqual(prerequisites[:4], command[:4])
                    for package in ("setuptools>=69", "wheel", "packaging", "psutil", "ninja"):
                        self.assertIn(package, prerequisites)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(bootstrap, "REPOS", {"Wan2.2": "unused", "alltracker": "unused"}), \
                     patch.object(sys, "argv", ["bootstrap.py", "--video-backend", "wan"]), \
                     patch.object(bootstrap.subprocess, "run", side_effect=install), \
                     patch.object(bootstrap.subprocess, "check_output", return_value="revision\n"):
                    bootstrap.main()
            finally:
                os.chdir(previous)
            self.assertEqual(len(calls), 3)
            self.assertNotIn("--no-build-isolation", calls[-1])
