import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bootstrap", ROOT/"scripts/bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/"
         "flash_attn-2.8.3.post2+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")


class BootstrapTests(unittest.TestCase):
    def test_wheel_tags_match_reported_cuda_python_torch_and_abi(self):
        import torch
        with patch.object(torch, "__version__", "2.6.0+cu124"), \
             patch.object(torch.version, "cuda", "12.4"), \
             patch.object(torch._C, "_GLIBCXX_USE_CXX11_ABI", False), \
             patch.object(bootstrap.sys, "version_info", type("Version", (), {"major": 3, "minor": 11})()), \
             patch.object(bootstrap.platform, "system", return_value="Linux"), \
             patch.object(bootstrap.platform, "machine", return_value="x86_64"):
            self.assertEqual(bootstrap.flash_attn_wheel_url(), WHEEL)

    def run_wan(self, installed):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"third_party/Wan2.2").mkdir(parents=True)
            (root/"third_party/alltracker").mkdir()
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                if command == [sys.executable, "-c", "import flash_attn"]:
                    return subprocess.CompletedProcess(command, 0 if installed else 1)
                if not installed and "third_party/Wan2.2/requirements.txt" in command:
                    # Without the direct wheel, the upstream source downloader
                    # reaches os.rename and reproduces the reported EXDEV error.
                    if WHEEL not in command:
                        raise OSError(18, "Invalid cross-device link")
                return subprocess.CompletedProcess(command, 0)
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(bootstrap, "REPOS", {"Wan2.2": "unused", "alltracker": "unused"}), \
                     patch.object(sys, "argv", ["bootstrap.py", "--video-backend", "wan"]), \
                     patch.object(bootstrap.subprocess, "run", side_effect=run), \
                     patch.object(bootstrap.subprocess, "check_output", return_value="revision\n"), \
                     patch.object(bootstrap, "flash_attn_wheel_url", return_value=WHEEL) as url:
                    bootstrap.main()
                    self.assertEqual(url.call_count, 0 if installed else 1)
            finally:
                os.chdir(previous)
            return calls

    def test_missing_flash_attention_installs_direct_wheel_and_checks_import(self):
        calls = self.run_wan(installed=False)
        install = next(c for c in calls if "third_party/Wan2.2/requirements.txt" in c)
        self.assertEqual(install[:4], [sys.executable, "-m", "pip", "install"])
        self.assertIn(WHEEL, install)
        self.assertTrue(any(c[1:3] == ["-c", "import flash_attn; print('FlashAttention:', flash_attn.__version__)"]
                            for c in calls))

    def test_working_flash_attention_is_reused_without_download(self):
        calls = self.run_wan(installed=True)
        self.assertFalse(any(WHEEL in c for c in calls))
        install = next(c for c in calls if "third_party/Wan2.2/requirements.txt" in c)
        self.assertIn("--no-build-isolation", install)
