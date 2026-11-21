"""Evaluate a trained E6 detector and save reproducible evidence.

    python -m Src.detector.evaluate_detector --checkpoint best.pt --protocol protocol.json \\
        --data data.yaml --split val --out-dir Results/e6_val

Outputs (in --out-dir): metrics.json, per_image_predictions.json, per_class.csv, errors.csv,
plus Ultralytics plots from its own validation pass.

Two kinds of numbers are produced and must not be mixed up:
  * ultralytics_val : mAP@0.5, mAP@0.5:0.95, per-class P/R/AP from Ultralytics' validator
                      (low conf threshold, standard COCO-style AP).
  * operating_point : precision/recall/FN/FP at the protocol's conf threshold with IoU >= 0.5
                      class-aware matching, i.e. what a user of the app actually sees.

Choose thresholds on val. Run --split test once, at the end, and do not tune afterwards.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classes import CLASS_NAMES
from .e6_detector import DetectorError, E6Detector, load_rgb_image
from .metrics import match_boxes
from .yolo_io import (
    DEFAULT_IMGSZ,
    dataset_root,
    is_small_box,
    label_path_for,
    list_images,
    load_data_yaml,
    parse_label_file,
    sha256_file,
    split_image_dirs,
    yolo_to_xyxy,
)

WARMUP_IMAGES = 3


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def evaluate_operating_point(
    detector: E6Detector,
    images: list[Path],
    *,
    conf: float | None,
    match_iou: float,
    imgsz: int,
) -> dict[str, Any]:
    per_class = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}
    small = {"gt": 0, "found": 0}
    per_image: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for index, img_path in enumerate(images):
        image = load_rgb_image(img_path)
        width, height = image.size
        lbl = label_path_for(img_path)
        rows, problems = parse_label_file(lbl, nc=len(CLASS_NAMES)) if lbl and lbl.is_file() else ([], ["missing label file"])
        gt = []
        for cls, xc, yc, w, h in rows:
            gt.append((cls, *yolo_to_xyxy(xc, yc, w, h, width, height)))
        result = detector.detect_image(image, conf)
        if index >= WARMUP_IMAGES:
            latencies.append(result.inference_ms)
        pred = [(d.class_id, d.score, *d.box_xyxy) for d in result.detections]
        matches, fps, fns = match_boxes(gt, pred, match_iou)

        for gi, _pi, _iou in matches:
            per_class[CLASS_NAMES[gt[gi][0]]]["tp"] += 1
        for pi in fps:
            per_class[CLASS_NAMES[pred[pi][0]]]["fp"] += 1
        for gi in fns:
            per_class[CLASS_NAMES[gt[gi][0]]]["fn"] += 1

        matched_gt = {gi for gi, _pi, _iou in matches}
        for gi, (cls, x1, y1, x2, y2) in enumerate(gt):
            is_small = is_small_box(x2 - x1, y2 - y1, width, height, imgsz)
            if is_small:
                small["gt"] += 1
                small["found"] += int(gi in matched_gt)
            if gi in fns:
                error_rows.append({"image": img_path.name, "kind": "missed_detection", "class": CLASS_NAMES[cls],
                                   "score": "", "box_xyxy": [round(v, 1) for v in (x1, y1, x2, y2)], "small_object": is_small})
        for pi in fps:
            p_cls, p_score, *p_box = pred[pi]
            error_rows.append({"image": img_path.name, "kind": "false_detection", "class": CLASS_NAMES[p_cls],
                               "score": round(p_score, 4), "box_xyxy": [round(v, 1) for v in p_box], "small_object": ""})

        per_image.append({
            "image": img_path.name,
            "width": width,
            "height": height,
            "label_problems": problems,
            "gt": [[CLASS_NAMES[g[0]], *[round(v, 1) for v in g[1:]]] for g in gt],
            "pred": [[CLASS_NAMES[p[0]], round(p[1], 4), *[round(v, 1) for v in p[2:]]] for p in pred],
            "matched": len(matches),
            "false_positives": len(fps),
            "false_negatives": len(fns),
            "inference_ms": round(result.inference_ms, 2),
        })

    totals = {"tp": 0, "fp": 0, "fn": 0}
    class_summary: dict[str, Any] = {}
    for name, counts in per_class.items():
        for key in totals:
            totals[key] += counts[key]
        class_summary[name] = {
            **counts,
            "gt_boxes": counts["tp"] + counts["fn"],
            "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
        }
    return {
        "conf_threshold": detector.conf_threshold if conf is None else conf,
        "match_iou": match_iou,
        "overall": {
            **totals,
            "precision": _ratio(totals["tp"], totals["tp"] + totals["fp"]),
            "recall": _ratio(totals["tp"], totals["tp"] + totals["fn"]),
        },
        "per_class": class_summary,
        "small_objects": {
            "definition": f"sqrt(box area) < 32 px after scaling the longest image side to {imgsz} px",
            "gt_small_boxes": small["gt"],
            "found": small["found"],
            "recall": _ratio(small["found"], small["gt"]),
        },
        "latency_ms": {
            "definition": "model.predict() call only (resize, forward pass, NMS); excludes file decoding; first "
                          f"{WARMUP_IMAGES} images excluded as warm-up",
            "images_timed": len(latencies),
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(_percentile(latencies, 95), 2) if latencies else None,
        },
        "images_with_any_error": sum(1 for row in per_image if row["false_positives"] or row["false_negatives"]),
        "_per_image": per_image,
        "_errors": error_rows,
    }


def ultralytics_val(detector: E6Detector, data_yaml: Path, split: str, out_dir: Path, batch: int) -> dict[str, Any]:
    try:
        metrics = detector.model.val(
            data=str(data_yaml.resolve()),
            split=split,
            imgsz=detector.imgsz,
            conf=0.001,
            iou=0.6,
            batch=batch,
            plots=True,
            project=str(out_dir.resolve()),
            name="ultralytics_val",
            exist_ok=True,
            verbose=False,
        )
        box = metrics.box
        per_class: dict[str, Any] = {name: {"note": "no ground-truth boxes in this split"} for name in CLASS_NAMES}
        for i, class_index in enumerate(box.ap_class_index):
            p, r, ap50, ap = box.class_result(i)
            per_class[CLASS_NAMES[int(class_index)]] = {
                "precision": round(float(p), 6),
                "recall": round(float(r), 6),
                "ap50": round(float(ap50), 6),
                "ap50_95": round(float(ap), 6),
            }
        return {
            "map50": round(float(box.map50), 6),
            "map50_95": round(float(box.map), 6),
            "mean_precision": round(float(box.mp), 6),
            "mean_recall": round(float(box.mr), 6),
            "conf_used": 0.001,
            "iou_nms_used": 0.6,
            "per_class": per_class,
            "plots_dir": str(out_dir / "ultralytics_val"),
        }
    except Exception as exc:  # noqa: BLE001 - report the failure instead of hiding or faking numbers
        return {"error": f"{type(exc).__name__}: {exc}"}


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for module in ("ultralytics", "torch"):
        try:
            env[module] = str(__import__(module).__version__)
        except ImportError:
            env[module] = "not installed"
    try:
        import torch

        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        env["device"] = "unknown"
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the E6 detector.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--conf", type=float, default=None, help="override the protocol conf threshold (validation experiments only)")
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--skip-ultralytics-val", action="store_true")
    args = parser.parse_args(argv)

    if args.split == "test":
        print("NOTE: test split. Report these numbers once, unchanged. Do not tune thresholds or pick checkpoints from them.")
        if args.conf is not None:
            print("Refusing to override --conf on the test split: choose the threshold on val first.")
            return 1

    try:
        detector = E6Detector(args.checkpoint, args.protocol)
    except DetectorError as exc:
        print(f"Cannot evaluate: {exc}")
        return 2

    data_yaml = Path(args.data)
    cfg = load_data_yaml(data_yaml)
    root = dataset_root(cfg, data_yaml)
    images = list_images(split_image_dirs(cfg, root, args.split))
    if not images:
        print(f"No images found for split '{args.split}'.")
        return 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Operating-point evaluation on {len(images)} {args.split} images ...")
    operating = evaluate_operating_point(
        detector, images, conf=args.conf, match_iou=args.match_iou, imgsz=detector.imgsz or DEFAULT_IMGSZ
    )
    per_image = operating.pop("_per_image")
    errors = operating.pop("_errors")

    ultra: dict[str, Any] | None = None
    if not args.skip_ultralytics_val:
        print("Ultralytics validation pass (mAP) ...")
        ultra = ultralytics_val(detector, data_yaml, args.split, out_dir, args.batch)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "images": len(images),
        "checkpoint": {"path": str(args.checkpoint), "sha256": detector.checkpoint_sha256},
        "protocol": {"path": str(args.protocol), "sha256": sha256_file(args.protocol)},
        "data_yaml_sha256": sha256_file(data_yaml),
        "environment": _environment(),
        "ultralytics_val": ultra,
        "operating_point": operating,
        "calibrated": False,
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "per_image_predictions.json").write_text(json.dumps(per_image, indent=1), encoding="utf-8")
    with (out_dir / "errors.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "kind", "class", "score", "box_xyxy", "small_object"])
        writer.writeheader()
        for row in errors:
            writer.writerow({**row, "box_xyxy": json.dumps(row["box_xyxy"])})
    with (out_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "gt_boxes", "tp", "fp", "fn", "precision", "recall", "ap50", "ap50_95"])
        for name in CLASS_NAMES:
            op = operating["per_class"][name]
            ap = (ultra or {}).get("per_class", {}).get(name, {}) if ultra and "error" not in ultra else {}
            writer.writerow([name, op["gt_boxes"], op["tp"], op["fp"], op["fn"], op["precision"], op["recall"],
                             ap.get("ap50", ""), ap.get("ap50_95", "")])

    overall = operating["overall"]
    print(f"\n{args.split} results  (conf={operating['conf_threshold']}, match IoU={args.match_iou})")
    print(f"  precision={overall['precision']}  recall={overall['recall']}  TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}")
    print(f"  small-object recall={operating['small_objects']['recall']} on {operating['small_objects']['gt_small_boxes']} small boxes")
    print(f"  latency median={operating['latency_ms']['median']} ms  p95={operating['latency_ms']['p95']} ms  ({report['environment']['device']})")
    if ultra and "error" not in ultra:
        print(f"  mAP@0.5={ultra['map50']}  mAP@0.5:0.95={ultra['map50_95']}")
    elif ultra:
        print(f"  Ultralytics validation failed: {ultra['error']}")
    print(f"Evidence written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
