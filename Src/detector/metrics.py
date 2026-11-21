"""Class-aware box matching used for operating-point precision/recall and error lists."""
from __future__ import annotations

from typing import Sequence

Box = tuple[float, float, float, float]


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    gt: Sequence[tuple[int, float, float, float, float]],
    pred: Sequence[tuple[int, float, float, float, float, float]],
    iou_thr: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy class-aware matching, highest-score predictions first.

    gt rows:   (class_id, x1, y1, x2, y2)
    pred rows: (class_id, score, x1, y1, x2, y2)
    Returns (matches[(gt_idx, pred_idx, iou)], false_positive_pred_idx, false_negative_gt_idx).
    """
    order = sorted(range(len(pred)), key=lambda i: -pred[i][1])
    used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    false_positives: list[int] = []
    for pi in order:
        p_cls, _score, *p_box = pred[pi]
        best, best_iou = -1, -1.0
        for gi, (g_cls, *g_box) in enumerate(gt):
            if gi in used or g_cls != p_cls:
                continue
            value = iou_xyxy(p_box, g_box)
            if value >= iou_thr and value > best_iou:
                best, best_iou = gi, value
        if best >= 0:
            used.add(best)
            matches.append((best, pi, best_iou))
        else:
            false_positives.append(pi)
    false_negatives = [gi for gi in range(len(gt)) if gi not in used]
    return matches, sorted(false_positives), false_negatives
