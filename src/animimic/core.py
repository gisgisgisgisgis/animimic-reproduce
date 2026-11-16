"""Validated configuration and dependency-aware, integrity-checked artifact caching."""
import hashlib
import json
import os
from pathlib import Path

import yaml
import csv


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or "asset" not in config:
        raise ValueError("Configuration requires asset")
    video = config.get("video", {})
    count = video.get("num_frames", 121)
    if count < 5 or (count-1) % 4:
        raise ValueError("Wan frame count must be 4k+1 and at least 5")
    if video.get("sample_frames", 25) < 3:
        raise ValueError("Optimization requires at least three sampled frames")
    if config.get("physics", {}).get("substeps", 4) < 1:
        raise ValueError("FEM substeps must be positive")
    return config


def save_history(path, history):
    save_json(path, history)
    if history:
        with Path(path).with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
                key for row in history for key in row)))
            writer.writeheader()
            writer.writerows(history)


class Cache:
    def __init__(self, root):
        self.root = Path(root)

    def run(self, stage, inputs, producer, force=False):
        key = digest({"stage": stage, "inputs": inputs, "schema": 1})
        directory = self.root / stage / key
        record_path = directory / "stage.json"
        if record_path.exists() and not force:
            record = json.loads(record_path.read_text())
            if all((directory / p).is_file() and file_hash(directory / p) == h
                   for p, h in record["files"].items()):
                return directory
        directory.mkdir(parents=True, exist_ok=True)
        files = producer(directory)
        if not files:
            raise ValueError(f"{stage} produced no artifacts")
        save_json(record_path, {"key": key, "inputs": inputs,
                               "files": {p: file_hash(directory / p) for p in files}})
        return directory
