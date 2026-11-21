"""E6 detector adapter: checkpoint-verified, honest multi-object inference.

Contract (mirrors the E2 predictor style)::

    detector = E6Detector(checkpoint_path, protocol_path)
    result = detector.detect(image_path)        # or detector.detect_image(pil_image)
    result.detections                            # real boxes only; may be empty
    result.to_dict()                             # JSON-ready, includes calibrated=False

The adapter never invents boxes. It refuses to run if the checkpoint hash, class order
or protocol is wrong. Scores are raw detector scores and are NOT calibrated.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .classes import CLASS_NAMES
from .yolo_io import sha256_file

PROTOCOL_VERSION = "e6-detector-protocol-1"
MAX_IMAGE_PIXELS = 50_000_000
MIN_IMAGE_SIDE = 8
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class DetectorError(RuntimeError):
    """Raised for any condition under which the detector must not produce output."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]  # pixels, original (EXIF-oriented) image


@dataclass(frozen=True)
class DetectionResult:
    image_width: int
    image_height: int
    detections: tuple[Detection, ...]
    inference_ms: float
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    dropped_invalid: int
    checkpoint_sha256: str
    protocol_version: str
    calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        w, h = self.image_width, self.image_height
        return {
            "protocol_version": self.protocol_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calibrated": self.calibrated,
            "score_note": "Raw detector scores. Not calibrated probabilities. No unknown-object rejection.",
            "image": {"width": w, "height": h},
            "settings": {
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "imgsz": self.imgsz,
            },
            "inference_ms": round(self.inference_ms, 2),
            "dropped_invalid_boxes": self.dropped_invalid,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "score": round(d.score, 6),
                    "box_xyxy": [round(v, 2) for v in d.box_xyxy],
                    "box_xyxy_norm": [
                        round(d.box_xyxy[0] / w, 6),
                        round(d.box_xyxy[1] / h, 6),
                        round(d.box_xyxy[2] / w, 6),
                        round(d.box_xyxy[3] / h, 6),
                    ],
                }
                for d in self.detections
            ],
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_protocol(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise DetectorError(f"Detector protocol file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DetectorError(f"Cannot read detector protocol {p.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise DetectorError("Detector protocol must be a JSON object")
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise DetectorError(f"Unsupported protocol_version {data.get('protocol_version')!r}; expected {PROTOCOL_VERSION!r}")
    if list(data.get("class_names", [])) != list(CLASS_NAMES):
        raise DetectorError(f"Protocol class_names must equal {list(CLASS_NAMES)} in that order")
    imgsz = data.get("imgsz")
    if not _is_int(imgsz) or not 32 <= imgsz <= 2048:
        raise DetectorError("Protocol imgsz must be an integer between 32 and 2048")
    conf = data.get("conf_threshold")
    if not _is_number(conf) or not 0 < conf < 1:
        raise DetectorError("Protocol conf_threshold must be a number in (0, 1)")
    iou = data.get("iou_threshold")
    if not _is_number(iou) or not 0 < iou <= 1:
        raise DetectorError("Protocol iou_threshold must be a number in (0, 1]")
    max_det = data.get("max_det", 100)
    if not _is_int(max_det) or not 1 <= max_det <= 1000:
        raise DetectorError("Protocol max_det must be an integer between 1 and 1000")
    sha = data.get("checkpoint_sha256")
    if not isinstance(sha, str) or not _SHA_RE.match(sha):
        raise DetectorError("Protocol checkpoint_sha256 must be 64 lowercase hex characters")
    out = dict(data)
    out["max_det"] = max_det
    return out


def load_rgb_image(source: str | Path | Any) -> Image.Image:
    """Decode an image (path or binary file object) to RGB with EXIF orientation applied."""
    try:
        with Image.open(source) as img:
            img.load()
            rgb = ImageOps.exif_transpose(img).convert("RGB")
    except FileNotFoundError as exc:
        raise DetectorError(f"Image not found: {source}") from exc
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise DetectorError(f"Could not read the image: {exc}") from exc
    width, height = rgb.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise DetectorError(f"Image is too small ({width}x{height}); minimum side is {MIN_IMAGE_SIDE} px")
    if width * height > MAX_IMAGE_PIXELS:
        raise DetectorError(f"Image is too large ({width}x{height} pixels)")
    return rgb


def _load_ultralytics(checkpoint: str) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorError("The 'ultralytics' package is not installed. Run: python -m pip install ultralytics") from exc
    try:
        return YOLO(checkpoint)
    except Exception as exc:  # noqa: BLE001 - surface any loader failure clearly
        raise DetectorError(f"Could not load detector checkpoint: {exc}") from exc


def _model_class_names(model: Any) -> list[str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        if sorted(names) != list(range(len(names))):
            raise DetectorError("Model class ids are not contiguous from 0")
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    raise DetectorError("Model does not expose class names")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _extract_arrays(results: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0))
    if results is None or len(results) == 0:
        return empty
    boxes = getattr(results[0], "boxes", None)
    if boxes is None:
        return empty
    xyxy = _as_numpy(boxes.xyxy).reshape(-1, 4)
    conf = _as_numpy(boxes.conf).reshape(-1)
    cls = _as_numpy(boxes.cls).reshape(-1)
    if not (len(xyxy) == len(conf) == len(cls)):
        raise DetectorError("Model returned inconsistent box, score and class counts")
    return xyxy, conf, cls


class E6Detector:
    """Hash-verified wrapper around a trained YOLO checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        protocol_path: str | Path,
        *,
        model_loader: Callable[[str], Any] | None = None,
        device: str | None = None,
    ) -> None:
        self.protocol = load_protocol(protocol_path)
        ckpt = Path(checkpoint_path)
        if not ckpt.is_file():
            raise DetectorError(f"Detector checkpoint not found: {ckpt}")
        actual = sha256_file(ckpt)
        if actual != self.protocol["checkpoint_sha256"]:
            raise DetectorError(
                "Checkpoint SHA-256 does not match the protocol "
                f"(expected {self.protocol['checkpoint_sha256'][:12]}..., got {actual[:12]}...)"
            )
        self.checkpoint_sha256 = actual
        self.imgsz: int = self.protocol["imgsz"]
        self.conf_threshold: float = float(self.protocol["conf_threshold"])
        self.iou_threshold: float = float(self.protocol["iou_threshold"])
        self.max_det: int = self.protocol["max_det"]
        self.device = device
        self.model = (model_loader or _load_ultralytics)(str(ckpt))
        if _model_class_names(self.model) != list(CLASS_NAMES):
            raise DetectorError(f"Model class names do not match the required order {list(CLASS_NAMES)}")

    def detect(self, image_path: str | Path, conf: float | None = None) -> DetectionResult:
        return self.detect_image(load_rgb_image(image_path), conf)

    def detect_image(self, image: Image.Image, conf: float | None = None) -> DetectionResult:
        conf_thr = self.conf_threshold if conf is None else float(conf)
        if not math.isfinite(conf_thr) or not 0 < conf_thr < 1:
            raise DetectorError("conf must be a number in (0, 1)")
        rgb = image.convert("RGB")
        width, height = rgb.size
        kwargs: dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": conf_thr,
            "iou": self.iou_threshold,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        start = time.perf_counter()
        try:
            results = self.model.predict(rgb, **kwargs)
        except DetectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - never return partial output after a failure
            raise DetectorError(f"Detector inference failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        xyxy, scores, cls_ids = _extract_arrays(results)
        detections, dropped = _build_detections(xyxy, scores, cls_ids, width, height)
        return DetectionResult(
            image_width=width,
            image_height=height,
            detections=tuple(detections),
            inference_ms=elapsed_ms,
            conf_threshold=conf_thr,
            iou_threshold=self.iou_threshold,
            imgsz=self.imgsz,
            dropped_invalid=dropped,
            checkpoint_sha256=self.checkpoint_sha256,
            protocol_version=self.protocol["protocol_version"],
        )


def _build_detections(
    xyxy: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray, width: int, height: int
) -> tuple[list[Detection], int]:
    detections: list[Detection] = []
    dropped = 0
    for box, score, cls in zip(xyxy, scores, cls_ids):
        values = [*box, score, cls]
        if not all(math.isfinite(v) for v in values):
            dropped += 1
            continue
        class_id = int(round(cls))
        if class_id != cls or not 0 <= class_id < len(CLASS_NAMES) or not 0.0 <= score <= 1.0:
            dropped += 1
            continue
        x1 = min(max(float(box[0]), 0.0), float(width))
        y1 = min(max(float(box[1]), 0.0), float(height))
        x2 = min(max(float(box[2]), 0.0), float(width))
        y2 = min(max(float(box[3]), 0.0), float(height))
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            dropped += 1
            continue
        detections.append(Detection(class_id, CLASS_NAMES[class_id], float(score), (x1, y1, x2, y2)))
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections, dropped
