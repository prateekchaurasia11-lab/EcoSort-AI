"""Synthetic fixtures for the detector tests.

STUB MODEL NOTICE: StubModel below is a unit-test double. It is not a trained detector and
nothing it returns may be reported as model performance.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Src.detector.classes import CLASS_NAMES  # noqa: E402
from Src.detector.e6_detector import PROTOCOL_VERSION  # noqa: E402


def write_image(path: Path, size: tuple[int, int] = (80, 60), seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=92)


def write_label(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c} {x} {y} {w} {h}\n" for c, x, y, w, h in rows), encoding="utf-8")


def make_dataset(root: Path, names=None) -> Path:
    """Valid 3-split dataset. train covers all six classes; val/test cover a few."""
    seed = 0
    plan = {
        "train": [[(i, 0.5, 0.5, 0.4, 0.4)] for i in range(6)],
        "val": [[(0, 0.4, 0.4, 0.3, 0.3), (4, 0.7, 0.7, 0.2, 0.2)], [(2, 0.5, 0.5, 0.5, 0.5)]],
        "test": [[(1, 0.5, 0.5, 0.4, 0.4)], [(5, 0.3, 0.3, 0.2, 0.2), (3, 0.7, 0.7, 0.2, 0.2)]],
    }
    for split, items in plan.items():
        for index, rows in enumerate(items):
            seed += 1
            # distinct sizes per image so stubs can recognise them
            size = (80 + 4 * seed, 60 + 2 * seed)
            write_image(root / "images" / split / f"{split}_{index}.jpg", size, seed)
            write_label(root / "labels" / split / f"{split}_{index}.txt", rows)
    cfg = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names or CLASS_NAMES)},
    }
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return data_yaml


class StubModel:
    """Unit-test double with the small surface E6Detector uses. NOT a trained model."""

    def __init__(self, output=None, names=None):
        self.names = names if names is not None else {i: n for i, n in enumerate(CLASS_NAMES)}
        self._output = output or (lambda image: ([], [], []))
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        xyxy, conf, cls = self._output(image)
        boxes = SimpleNamespace(
            xyxy=np.array(xyxy, dtype=float).reshape(-1, 4),
            conf=np.array(conf, dtype=float),
            cls=np.array(cls, dtype=float),
        )
        return [SimpleNamespace(boxes=boxes)]


def make_checkpoint_and_protocol(folder: Path, **overrides) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    ckpt = folder / "best.pt"
    ckpt.write_bytes(b"placeholder bytes for hashing only - not a real checkpoint")
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "class_names": list(CLASS_NAMES),
        "imgsz": 640,
        "conf_threshold": 0.25,
        "iou_threshold": 0.6,
        "max_det": 100,
        "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    }
    protocol.update(overrides)
    proto_path = folder / "protocol.json"
    proto_path.write_text(json.dumps(protocol), encoding="utf-8")
    return ckpt, proto_path
