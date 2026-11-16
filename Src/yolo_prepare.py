"""TACO / local COCO-JSON to 6-class YOLO converter, group-aware splitter, and visual check generator."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Conservative TACO category -> EcoSort 6-class mapping (uncertain/composite -> None = deferred)
TACO_TO_ECOSORT: dict[str, str | None] = {
    "Cardboard": "cardboard",
    "Corrugated carton": "cardboard",
    "Egg carton": "cardboard",
    "Paper bag": "paper",
    "Paper cup": "paper",
    "Tissues": "trash",
    "Wrapping paper": "paper",
    "Magazine paper": "paper",
    "Normal paper": "paper",
    "Glass bottle": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",
    "Broken glass": "glass",
    "Aluminium foil": "metal",
    "Aluminium blister pack": "trash",
    "Can": "metal",
    "Drink can": "metal",
    "Food can": "metal",
    "Aerosol": "metal",
    "Metal bottle cap": "metal",
    "Scrap metal": "metal",
    "Clear plastic bottle": "plastic",
    "Other plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Plastic lid": "plastic",
    "Plastic film": "plastic",
    "Single-use carrier bag": "plastic",
    "Garbage bag": "plastic",
    "Plastic straw": "plastic",
    "Plastic utensils": "plastic",
    "Disposable plastic cup": "plastic",
    "Other plastic container": "plastic",
    "Other plastic wrapper": "trash",
    "Crisp packet": "trash",
    "Drink carton": None,  # Deferred pending Prateek/supervisor policy confirmation
    "Tetrapak": None,      # Deferred pending Prateek/supervisor policy confirmation
    "Cigarette": "trash",
    "Styrofoam piece": "trash",
    "Shoe": "trash",
    "Battery": None,       # Hazardous/e-waste -> deferred / unknown rejection
    "Unlabeled litter": None,
}


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    x_center = max(0.0, min(1.0, (x + w / 2.0) / img_w))
    y_center = max(0.0, min(1.0, (y + h / 2.0) / img_h))
    norm_w = max(0.0, min(1.0, w / img_w))
    norm_h = max(0.0, min(1.0, h / img_h))
    if norm_w <= 0 or norm_h <= 0:
        return None
    return x_center, y_center, norm_w, norm_h


def safe_output_stem(rel_name: str, img_id: int) -> str:
    rel = Path(rel_name)
    parent_part = rel.parent.as_posix().replace("/", "_")
    if parent_part in ("", "."):
        return f"{rel.stem}_id{img_id}"
    return f"{parent_part}_{rel.stem}_id{img_id}"


def convert_coco_to_yolo(
    coco_json: Path,
    images_root: Path,
    output_dir: Path,
    seed: int = 42,
    visual_samples: int = 12,
) -> int:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    images_by_id = {img["id"]: img for img in data.get("images", [])}

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_image[ann["image_id"]].append(ann)

    # Group by batch/folder prefix to prevent scene/session leakage across train/val/test
    groups: dict[str, list[int]] = defaultdict(list)
    for img_id, img_info in images_by_id.items():
        rel = Path(img_info["file_name"])
        group_key = rel.parent.as_posix() if rel.parent.as_posix() != "." else rel.stem.split("_")[0]
        groups[group_key].append(img_id)

    rng = random.Random(seed)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_train = max(1, int(0.7 * n_groups)) if n_groups >= 3 else n_groups
    n_val = max(1, int(0.15 * n_groups)) if n_groups >= 3 else 0
    split_for_group = {}
    for idx, g in enumerate(group_keys):
        if idx < n_train:
            split_for_group[g] = "train"
        elif idx < n_train + n_val:
            split_for_group[g] = "val"
        else:
            split_for_group[g] = "test"

    box_counts: Counter[str] = Counter()
    deferred_cats: Counter[str] = Counter()
    split_image_counts: Counter[str] = Counter()
    visual_dir = output_dir / "visual_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0

    for g, img_ids in groups.items():
        split = split_for_group[g]
        label_dir = output_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)

        for img_id in img_ids:
            img_info = images_by_id[img_id]
            rel_name = img_info["file_name"]
            w, h = int(img_info["width"]), int(img_info["height"])
            yolo_lines: list[str] = []
            draw_boxes: list[tuple[list[float], str]] = []

            for ann in anns_by_image.get(img_id, []):
                raw_cat = cat_id_to_name.get(ann["category_id"], "Unknown")
                mapped = TACO_TO_ECOSORT.get(raw_cat)
                if mapped is None:
                    deferred_cats[raw_cat] += 1
                    continue
                yolo_box = coco_bbox_to_yolo(ann["bbox"], w, h)
                if yolo_box is None:
                    continue
                cls_id = CLASS_TO_ID[mapped]
                xc, yc, bw, bh = yolo_box
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                box_counts[mapped] += 1
                draw_boxes.append((ann["bbox"], mapped))

            if not yolo_lines:
                continue

            out_stem = safe_output_stem(rel_name, img_id)
            (label_dir / f"{out_stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            split_image_counts[split] += 1

            src_img = images_root / rel_name
            if rendered < visual_samples and src_img.is_file():
                with Image.open(src_img) as im:
                    canvas = im.convert("RGB")
                    draw = ImageDraw.Draw(canvas)
                    for (bx, by, bw_px, bh_px), lbl in draw_boxes:
                        draw.rectangle([bx, by, bx + bw_px, by + bh_px], outline="lime", width=3)
                        draw.text((bx + 4, max(0.0, by - 14)), lbl, fill="lime")
                    canvas.save(visual_dir / f"{split}_{out_stem}_check.jpg")
                rendered += 1

    yaml_text = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Converted images per split: {dict(split_image_counts)}")
    print(f"Converted boxes per class: {dict(box_counts)}")
    print(f"Deferred/unmapped category boxes: {dict(deferred_cats)}")
    print(f"Visual check overlays saved: {rendered} -> {visual_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert COCO/TACO annotations to 6-class YOLO labels with visual checks.")
    parser.add_argument("coco_json", type=Path, help="Path to COCO/TACO annotations.json")
    parser.add_argument("images_root", type=Path, help="Root directory containing images")
    parser.add_argument("--out", type=Path, default=Path("Data/yolo_6class"), help="Output directory (gitignored)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group-aware split")
    parser.add_argument("--visual-samples", type=int, default=12, help="Number of visual bounding-box check images")
    args = parser.parse_args()
    return convert_coco_to_yolo(args.coco_json, args.images_root, args.out, args.seed, args.visual_samples)


if __name__ == "__main__":
    raise SystemExit(main())
