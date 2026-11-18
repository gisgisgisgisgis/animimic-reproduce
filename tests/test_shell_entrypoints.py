"""Check shell argument/environment boundaries without installing or invoking models."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ShellEntrypointTests(unittest.TestCase):
    def test_tetwild_installs_compatible_cmake_and_builds(self):
        with tempfile.TemporaryDirectory(prefix="animimic tetwild ") as temporary:
            root = Path(temporary)
            scripts = root/"scripts"
            scripts.mkdir()
            for name in ("_common.sh", "04_build_tetwild.sh"):
                shutil.copy2(ROOT/"scripts"/name, scripts/name)
            source = root/"third_party/fTetWild"
            source.mkdir(parents=True)
            (source/"CMakeLists.txt").touch()
            prefix = root/"test env"
            binaries = prefix/"bin"
            binaries.mkdir(parents=True)
            (prefix/"conda-meta").mkdir()
            log = root/"commands.jsonl"
            for name in ("python", "conda", "cmake", "FloatTetwild_bin"):
                executable = binaries/name
                executable.write_text(
                    f"#!{sys.executable}\n"
                    "import json, pathlib, sys\n"
                    f"with open({str(log)!r}, 'a') as stream:\n"
                    "    stream.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')\n"
                    "if sys.argv[1:] == ['--version']:\n"
                    "    print('cmake version 3.29.9')\n"
                )
                executable.chmod(0o755)
            subprocess.run(["bash", str(scripts/"04_build_tetwild.sh")],
                           env={**os.environ, "ANIMIMIC_ENV_DIR": str(prefix)},
                           capture_output=True, text=True, check=True)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertIn(["conda", "install", "--prefix", str(prefix), "-c",
                           "conda-forge", "cmake=3.29.*", "ninja", "gmp", "-y"], commands)
            self.assertTrue(any(command[:2] == ["cmake", "-S"] for command in commands))
            self.assertTrue(any(command[:2] == ["cmake", "--build"] for command in commands))
            self.assertIn(["FloatTetwild_bin", "--help"], commands)

    def test_cuda_installer_passes_valid_conda_arguments(self):
        with tempfile.TemporaryDirectory(prefix="animimic toolkit ") as temporary:
            root = Path(temporary)
            scripts = root/"scripts"
            scripts.mkdir()
            for name in ("_common.sh", "12_install_cuda_toolkit.sh"):
                shutil.copy2(ROOT/"scripts"/name, scripts/name)
            binaries = root/"bin"
            binaries.mkdir()
            conda = binaries/"conda"
            conda.write_text(
                f"#!{sys.executable}\n"
                "import json, pathlib, sys\n"
                "args = sys.argv[1:]\n"
                "print(json.dumps(args))\n"
                "prefix = pathlib.Path(args[args.index('--prefix') + 1])\n"
                "(prefix/'conda-meta').mkdir(parents=True, exist_ok=True)\n"
                "(prefix/'bin').mkdir(exist_ok=True)\n"
                "nvcc = prefix/'bin/nvcc'\n"
                "nvcc.write_text(\"#!/bin/sh\\nprintf 'release 12.4\\\\n'\\n\")\n"
                "nvcc.chmod(0o755)\n"
            )
            conda.chmod(0o755)
            for action in ("create", "install"):
                result = subprocess.run(
                    ["bash", str(scripts/"12_install_cuda_toolkit.sh")],
                    env={**os.environ, "PATH": f"{binaries}:{os.environ['PATH']}"},
                    capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(result.stdout.splitlines()[0]), [
                    action, "--prefix", str(root/".cache/cuda-12.4"),
                    "--override-channels", "-c", "nvidia/label/cuda-12.4.1",
                    "-c", "conda-forge", "cuda-toolkit=12.4.1", "-y"])

    def test_cuda_home_is_discovered_and_added_to_path(self):
        with tempfile.TemporaryDirectory(prefix="animimic cuda ") as temporary:
            prefix = Path(temporary)
            (prefix/"bin").mkdir()
            nvcc = prefix/"bin/nvcc"
            nvcc.write_text("#!/bin/sh\nprintf 'Cuda compilation tools, release 12.4, V12.4.131\\n'\n")
            nvcc.chmod(0o755)
            result = subprocess.run(
                ["bash", "-c", 'source "$1"; use_cuda_toolkit; command -v nvcc',
                 "test", str(ROOT/"scripts/_common.sh")],
                env={**os.environ, "CUDA_HOME": str(prefix)},
                capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), str(nvcc))

    def invoke(self, script, args):
        with tempfile.TemporaryDirectory(prefix="animimic shell ") as temporary:
            caller = Path(temporary)
            prefix = caller/"test env"
            (prefix/"bin").mkdir(parents=True)
            python = prefix/"bin/python"
            # Only the shell boundary is under test. Python/CUDA probes succeed
            # in this stub, and every model invocation is recorded, never run.
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "if sys.argv[1:2] == ['-c']: sys.exit(0)\n"
                "print(json.dumps(dict(args=sys.argv[1:], cwd=os.getcwd(), "
                "gpu=os.getenv('CUDA_VISIBLE_DEVICES'), hf=os.getenv('HF_HOME'))))\n"
            )
            python.chmod(0o755)
            (caller/"input image.png").write_bytes(b"shell path fixture")
            environment = {**os.environ, "ANIMIMIC_ENV_DIR": str(prefix),
                           "CUDA_VISIBLE_DEVICES": "3", "HF_HOME": str(caller/"hf cache")}
            result = subprocess.run(["bash", str(ROOT/"scripts"/script), *args],
                                    cwd=caller, env=environment, text=True,
                                    capture_output=True, check=True)
            return json.loads(result.stdout.strip()), caller

    def test_run_forwards_flags_and_preserves_scheduler_device(self):
        record, caller = self.invoke("08_run_pipeline.sh",
                                    ["my config.yaml", "--from-stage", "joints"])
        self.assertEqual(record["args"], [
            "-m", "animimic.cli", "run", "--config", str(caller/"my config.yaml"),
            "--from-stage", "joints"])
        self.assertEqual(record["cwd"], str(ROOT))
        self.assertEqual(record["gpu"], "3")
        self.assertEqual(record["hf"], str(caller/"hf cache"))

    def test_video_keeps_caller_relative_paths_with_spaces(self):
        record, caller = self.invoke("07_generate_video.sh",
            ["input image.png", "my config.yaml", "my output.mp4"])
        self.assertEqual(record["args"], [
            "-m", "animimic.cli", "video", "--config", str(caller/"my config.yaml"),
            "--image", str(caller/"input image.png"), "--output", str(caller/"my output.mp4")])
