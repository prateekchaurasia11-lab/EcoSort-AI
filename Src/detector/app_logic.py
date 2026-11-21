"""UI-independent logic for the detection page: validation, inference, summaries, JSON."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .classes import CLASS_NAMES, GUIDANCE, GUIDANCE_NOTICE
from .e6_detector import DetectionResult, DetectorError, E6Detector, load_rgb_image
from .render import count_by_class, draw_detections

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class UploadError(ValueError):
    """The uploaded file is not acceptable. The message is safe to show to the user."""


@dataclass(frozen=True)
class Outcome:
    result: DetectionResult
    annotated_png: bytes
    counts: dict[str, int]
    object_rows: list[dict[str, Any]]
    guidance_rows: list[dict[str, Any]]


def validate_upload(filename: str, data: bytes) -> None:
    name = (filename or "").lower()
    if not name.endswith(ALLOWED_EXTENSIONS):
        raise UploadError("Please upload a JPEG or PNG image.")
    if not data:
        raise UploadError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not (data.startswith(_JPEG_MAGIC) or data.startswith(_PNG_MAGIC)):
        raise UploadError("The file content is not a valid JPEG or PNG image.")


def guidance_rows(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"category": name, "count": counts[name], "advice": GUIDANCE[name]}
        for name in CLASS_NAMES
        if counts[name] > 0
    ]


def run_detection(detector: E6Detector, data: bytes, conf: float | None = None) -> Outcome:
    """Decode the upload once, run the real detector, and build everything the page shows."""
    image = load_rgb_image(io.BytesIO(data))
    result = detector.detect_image(image, conf)
    annotated = draw_detections(image, result.detections)
    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    counts = count_by_class(result.detections)
    rows = [
        {
            "#": index,
            "category": det.class_name,
            "score": round(det.score, 3),
            "box_xyxy_px": [round(v) for v in det.box_xyxy],
        }
        for index, det in enumerate(result.detections, start=1)
    ]
    return Outcome(result, buffer.getvalue(), counts, rows, guidance_rows(counts))


def build_report(outcome: Outcome, filename: str, data: bytes) -> dict[str, Any]:
    report = outcome.result.to_dict()
    report["source_file"] = {"name": filename, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    report["counts_by_category"] = outcome.counts
    report["guidance_notice"] = GUIDANCE_NOTICE
    return report


__all__ = [
    "ALLOWED_EXTENSIONS",
    "DetectorError",
    "MAX_UPLOAD_BYTES",
    "Outcome",
    "UploadError",
    "build_report",
    "guidance_rows",
    "run_detection",
    "validate_upload",
]
