"""Helpers for reading Ultralytics-style YOLO detection datasets (images/ + labels/)."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
DEFAULT_IMGSZ = 640
SMALL_OBJECT_PX = 32.0  # proposal guideline: about 32 x 32 px at 640 x 640 input
EDGE_TOL = 1e-6


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"data.yaml not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("data.yaml must contain a mapping")
    return cfg


def normalise_names(names: Any) -> list[str] | None:
    """Return class names as an ordered list, or None if the layout is invalid."""
    if isinstance(names, dict):
        try:
            keys = sorted(int(k) for k in names)
        except (TypeError, ValueError):
            return None
        if keys != list(range(len(keys))):
            return None
        return [str(names[k] if k in names else names[str(k)]) for k in keys]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return None


def dataset_root(cfg: dict[str, Any], yaml_path: str | Path) -> Path:
    yaml_path = Path(yaml_path).resolve()
    raw = cfg.get("path")
    root = Path(str(raw)) if raw else yaml_path.parent
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def split_image_dirs(cfg: dict[str, Any], root: Path, split: str) -> list[Path]:
    entry = cfg.get(split)
    if entry is None:
        return []
    items = entry if isinstance(entry, (list, tuple)) else [entry]
    out: list[Path] = []
    for item in items:
        q = Path(str(item))
        out.append(q if q.is_absolute() else root / q)
    return out


def list_images(dirs: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for d in dirs:
        if d.is_dir():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    found.add(p)
    return sorted(found)


def label_path_for(image_path: Path) -> Path | None:
    """Ultralytics rule: replace the last 'images' path component with 'labels'."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return None


def oriented_size(path: str | Path) -> tuple[int, int]:
    """Image (width, height) after applying EXIF orientation."""
    with Image.open(path) as im:
        w, h = im.size
        try:
            orientation = im.getexif().get(0x0112)
        except Exception:  # noqa: BLE001 - damaged EXIF must not break size lookup
            orientation = None
    if orientation in (5, 6, 7, 8):
        w, h = h, w
    return w, h


def parse_label_file(path: Path, nc: int = 6) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Parse a YOLO detection label file. Returns (rows, problems)."""
    rows: list[tuple[int, float, float, float, float]] = []
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return rows, [f"cannot read label file: {exc}"]
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) != 5:
            problems.append(f"line {line_no}: expected 5 values, found {len(tokens)}")
            continue
        if not tokens[0].isdigit():
            problems.append(f"line {line_no}: class id '{tokens[0]}' is not a non-negative integer")
            continue
        cls = int(tokens[0])
        if cls >= nc:
            problems.append(f"line {line_no}: class id {cls} outside 0..{nc - 1}")
            continue
        try:
            xc, yc, w, h = (float(t) for t in tokens[1:])
        except ValueError:
            problems.append(f"line {line_no}: non-numeric coordinate")
            continue
        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            problems.append(f"line {line_no}: non-finite coordinate")
            continue
        if w <= 0 or h <= 0:
            problems.append(f"line {line_no}: non-positive width or height")
            continue
        if min(xc - w / 2, yc - h / 2) < -EDGE_TOL or max(xc + w / 2, yc + h / 2) > 1 + EDGE_TOL:
            problems.append(f"line {line_no}: box extends outside the image (0..1)")
            continue
        rows.append((cls, xc, yc, w, h))
    return rows, problems


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    return (
        (xc - w / 2) * img_w,
        (yc - h / 2) * img_h,
        (xc + w / 2) * img_w,
        (yc + h / 2) * img_h,
    )


def size_at_imgsz(box_w_px: float, box_h_px: float, img_w: int, img_h: int, imgsz: int = DEFAULT_IMGSZ) -> float:
    """sqrt(area) of a box after the image's longest side is scaled to ``imgsz``."""
    scale = imgsz / max(img_w, img_h)
    return math.sqrt(max(box_w_px, 0.0) * max(box_h_px, 0.0)) * scale


def is_small_box(box_w_px: float, box_h_px: float, img_w: int, img_h: int, imgsz: int = DEFAULT_IMGSZ) -> bool:
    return size_at_imgsz(box_w_px, box_h_px, img_w, img_h, imgsz) < SMALL_OBJECT_PX
