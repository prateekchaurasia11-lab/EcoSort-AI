"""Validate a YOLO detection dataset before any training run.

Checks: class order, split directories, image/label pairing, decodable images, label
syntax and coordinate ranges, empty splits, class coverage, exact (SHA-256) duplicates
across splits, and optional group leakage from a manifest CSV.

    python -m Src.detector.check_dataset --data Data\\det_v1\\data.yaml --report Results\\det_v1_check.json

Exit code 0 only when there are no errors. It never modifies the dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from .classes import CLASS_NAMES
from .yolo_io import (
    DEFAULT_IMGSZ,
    SPLITS,
    dataset_root,
    is_small_box,
    label_path_for,
    list_images,
    load_data_yaml,
    normalise_names,
    oriented_size,
    parse_label_file,
    sha256_file,
    split_image_dirs,
)

MIN_SIDE = 8
MAX_LISTED = 200


def _load_groups(manifest_csv: Path) -> dict[str, str]:
    with manifest_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "file_name" not in fields or "group_id" not in fields:
            raise ValueError("group manifest needs 'file_name' and 'group_id' columns")
        return {Path(row["file_name"]).name: (row["group_id"] or "").strip() for row in reader}


def check_dataset(
    data_yaml: str | Path,
    *,
    manifest_csv: str | Path | None = None,
    full_decode: bool = False,
    imgsz: int = DEFAULT_IMGSZ,
) -> dict[str, Any]:
    data_yaml = Path(data_yaml)
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "ok": False,
        "errors": errors,
        "warnings": warnings,
        "splits": {},
        "class_boxes_total": {name: 0 for name in CLASS_NAMES},
    }
    try:
        cfg = load_data_yaml(data_yaml)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        errors.append(str(exc))
        return report
    report["data_yaml_sha256"] = sha256_file(data_yaml)

    names = normalise_names(cfg.get("names"))
    if names != list(CLASS_NAMES):
        errors.append(f"data.yaml names must be exactly {list(CLASS_NAMES)} in that order; found {names}")
    if cfg.get("nc") is not None and cfg.get("nc") != len(CLASS_NAMES):
        errors.append(f"data.yaml nc must be {len(CLASS_NAMES)}; found {cfg.get('nc')}")
    raw_path = cfg.get("path")
    if raw_path and not Path(str(raw_path)).is_absolute():
        warnings.append(
            "data.yaml 'path' is relative. Ultralytics resolves a relative path against its own datasets "
            "directory, so use an absolute path when training."
        )
    root = dataset_root(cfg, data_yaml)
    report["dataset_root"] = str(root)

    groups: dict[str, str] | None = None
    if manifest_csv is not None:
        try:
            groups = _load_groups(Path(manifest_csv))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"cannot use group manifest: {exc}")

    seen: dict[str, list[tuple[str, str]]] = defaultdict(list)
    group_splits: dict[str, set[str]] = defaultdict(set)
    missing_group = 0

    for split in SPLITS:
        dirs = split_image_dirs(cfg, root, split)
        stats: dict[str, Any] = {
            "images": 0,
            "boxes": 0,
            "images_without_boxes": 0,
            "small_boxes_lt_32px_at_imgsz": 0,
            "class_boxes": {name: 0 for name in CLASS_NAMES},
        }
        report["splits"][split] = stats
        if not dirs:
            errors.append(f"data.yaml has no '{split}' entry")
            continue
        for d in dirs:
            if not d.is_dir():
                errors.append(f"{split}: image directory not found: {d}")
        images = list_images(dirs)
        stats["images"] = len(images)
        if not images:
            errors.append(f"{split}: no images found (an empty split cannot be evaluated)")
            continue

        expected_labels: set[Path] = set()
        for img_path in images:
            rel = img_path.name
            lbl_path = label_path_for(img_path)
            if lbl_path is None:
                errors.append(f"{split}: {rel}: path has no 'images' directory component for label lookup")
                continue
            expected_labels.add(lbl_path)

            try:
                with Image.open(img_path) as im:
                    im.verify()
                if full_decode:
                    with Image.open(img_path) as im:
                        im.load()
                img_w, img_h = oriented_size(img_path)
            except Exception as exc:  # noqa: BLE001 - any decode failure is a dataset error
                errors.append(f"{split}: {rel}: cannot decode image ({exc})")
                continue
            if min(img_w, img_h) < MIN_SIDE:
                errors.append(f"{split}: {rel}: image too small ({img_w}x{img_h})")
                continue

            digest = sha256_file(img_path)
            seen[digest].append((split, rel))

            if groups is not None:
                group = groups.get(img_path.name)
                if group:
                    group_splits[group].add(split)
                else:
                    missing_group += 1

            if not lbl_path.is_file():
                errors.append(f"{split}: {rel}: no label file (write an empty .txt only for an intentional negative image)")
                continue
            rows, problems = parse_label_file(lbl_path, nc=len(CLASS_NAMES))
            for problem in problems:
                errors.append(f"{split}: {lbl_path.name}: {problem}")
            if len(set(rows)) != len(rows):
                warnings.append(f"{split}: {lbl_path.name}: duplicate identical boxes")
            if not rows:
                stats["images_without_boxes"] += 1
            for cls, _xc, _yc, w, h in rows:
                stats["boxes"] += 1
                stats["class_boxes"][CLASS_NAMES[cls]] += 1
                report["class_boxes_total"][CLASS_NAMES[cls]] += 1
                if is_small_box(w * img_w, h * img_h, img_w, img_h, imgsz):
                    stats["small_boxes_lt_32px_at_imgsz"] += 1

        label_dirs = {p.parent for p in expected_labels}
        for label_dir in label_dirs:
            if label_dir.is_dir():
                for orphan in sorted(label_dir.glob("*.txt")):
                    if orphan not in expected_labels:
                        errors.append(f"{split}: label without a matching image: {orphan.name}")

    for digest, owners in seen.items():
        splits_hit = {s for s, _ in owners}
        if len(splits_hit) > 1:
            names_hit = ", ".join(f"{s}/{n}" for s, n in owners[:4])
            errors.append(f"cross-split exact duplicate image (leakage): {names_hit}")
        elif len(owners) > 1:
            warnings.append(f"duplicate images inside {owners[0][0]}: {', '.join(n for _, n in owners[:4])}")

    if groups is not None:
        for group, splits_hit in sorted(group_splits.items()):
            if len(splits_hit) > 1:
                errors.append(f"group '{group}' appears in several splits: {sorted(splits_hit)}")
        if missing_group:
            warnings.append(f"{missing_group} images have no group_id in the manifest (group leakage cannot be ruled out for them)")
    else:
        warnings.append("no group manifest supplied: only exact-duplicate leakage was checked, not scene/session leakage")

    for name in CLASS_NAMES:
        if report["splits"].get("train", {}).get("class_boxes", {}).get(name, 0) == 0:
            errors.append(f"class '{name}' has no boxes in train")
        for split in ("val", "test"):
            if report["splits"].get(split, {}).get("images", 0) and report["splits"][split]["class_boxes"][name] == 0:
                warnings.append(f"class '{name}' has no boxes in {split}; its metrics cannot be computed there")

    report["ok"] = not errors
    report["error_count"] = len(errors)
    report["warning_count"] = len(warnings)
    if len(errors) > MAX_LISTED:
        report["errors"] = errors[:MAX_LISTED] + [f"... and {len(errors) - MAX_LISTED} more"]
    if len(warnings) > MAX_LISTED:
        report["warnings"] = warnings[:MAX_LISTED] + [f"... and {len(warnings) - MAX_LISTED} more"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset (read-only).")
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--manifest", help="optional CSV with file_name and group_id columns")
    parser.add_argument("--report", help="write the full JSON report here")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--full-decode", action="store_true", help="fully decode every image (slower)")
    args = parser.parse_args(argv)

    report = check_dataset(args.data, manifest_csv=args.manifest, full_decode=args.full_decode, imgsz=args.imgsz)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset check: {'PASS' if report['ok'] else 'FAIL'}  ({report.get('error_count', len(report['errors']))} errors, "
          f"{report.get('warning_count', len(report['warnings']))} warnings)")
    for split, stats in report["splits"].items():
        print(f"  {split:5s} images={stats['images']:5d} boxes={stats['boxes']:5d} "
              f"no-box images={stats['images_without_boxes']:4d} small boxes={stats['small_boxes_lt_32px_at_imgsz']:4d}")
        print("        " + "  ".join(f"{k}={v}" for k, v in stats["class_boxes"].items()))
    for message in report["errors"][:25]:
        print(f"  ERROR: {message}")
    for message in report["warnings"][:10]:
        print(f"  warning: {message}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
