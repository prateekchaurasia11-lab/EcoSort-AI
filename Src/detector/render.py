"""Draw genuine detections on an image and summarise them. Takes detector output only."""
from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from .classes import CLASS_NAMES
from .e6_detector import Detection

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "cardboard": (166, 118, 29),
    "glass": (86, 180, 233),
    "metal": (0, 158, 115),
    "paper": (240, 228, 66),
    "plastic": (213, 94, 0),
    "trash": (204, 121, 167),
}


def count_by_class(detections: Sequence[Detection]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for det in detections:
        counts[det.class_name] += 1
    return counts


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without the size argument
        return ImageFont.load_default()


def _text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (0, 0, 0) if luminance > 150 else (255, 255, 255)


def draw_detections(image: Image.Image, detections: Sequence[Detection]) -> Image.Image:
    """Return a copy of ``image`` with one numbered box and label per detection."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line = max(2, round(min(width, height) / 200))
    font = _font(max(12, round(min(width, height) / 40)))
    pad = max(2, line)
    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.box_xyxy
        color = CLASS_COLORS[det.class_name]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line)
        label = f"{index}. {det.class_name} {det.score:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
        box_h = text_h + 2 * pad
        label_top = y1 - box_h if y1 - box_h >= 0 else y1
        label_left = min(max(x1, 0), max(width - (text_w + 2 * pad), 0))
        draw.rectangle([label_left, label_top, label_left + text_w + 2 * pad, label_top + box_h], fill=color)
        draw.text((label_left + pad - left, label_top + pad - top), label, fill=_text_color(color), font=font)
    return canvas
