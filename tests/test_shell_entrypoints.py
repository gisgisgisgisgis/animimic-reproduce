"""Check shell argument/environment boundaries without installing or invoking models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ShellEntrypointTests(unittest.TestCase):
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
