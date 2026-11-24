"""Portable asset layout shared by inference and scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HEIGHT, WIDTH = 252, 504
HISTORY_END_MS = [-100, -50, 0]
DEFAULT_HORIZONS = (200, 400, 600, 800)
EVENTSCAPE_HORIZONS = tuple(range(120, 801, 40))

P40530_SHA256 = "40bf069d659c9da560c5fd4bad27602ca93c09cda588220ba3d47f95e64fa8ee"
EVENTVGGT_SHA256 = "fd4f5f711a90a0a37ad885ca6cff208591711356573b316f5f1c40c9093c9e52"


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def asset(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve(strict=True)
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid dataset path: {relative}")
    return path


def verify_assets(root: Path) -> int:
    manifest = read_json(root / "assets.json")
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported dataset manifest")
    for name, digest in manifest["files"].items():
        if sha256(asset(root, name)) != digest:
            raise ValueError(f"Dataset file checksum mismatch: {name}")
    return len(manifest["files"])


def read_cases(root: Path) -> dict[str, dict]:
    index = read_json(root / "inputs/index.json")
    if index.get("format_version") != 1 or index.get("dataset") != "EventScape":
        raise ValueError("Unsupported EventScape input index")
    cases = {}
    for row in index["cases"]:
        identity = row["case_id"]
        if identity in cases or not identity or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in identity):
            raise ValueError(f"Invalid or duplicate case ID: {identity}")
        if row["history_end_ms"] != HISTORY_END_MS or len(row["events"]) != 3:
            raise ValueError(f"Expected three contiguous 50 ms packets: {identity}")
        cases[identity] = row
    if not cases:
        raise ValueError("Empty EventScape input index")
    return cases


def load_case_images(paths: list[Path]) -> np.ndarray:
    from PIL import Image
    if len(paths) != 3:
        raise ValueError("Supply three event images, from oldest to newest")
    images = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        if pixels.shape != (HEIGHT, WIDTH, 3):
            raise ValueError(f"Expected a native {WIDTH}x{HEIGHT} event image: {path}")
        images.append(pixels)
    return np.stack(images)


def load_motion(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        features = source["ordered"].astype(np.float32)
    if features.shape != (1, 3, 648, 96) or not np.isfinite(features).all():
        raise ValueError("Expected finite TESPEC features with shape [1, 3, 648, 96] in 'ordered'")
    return features
