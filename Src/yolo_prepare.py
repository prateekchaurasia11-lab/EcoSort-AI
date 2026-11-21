"""TACO / local COCO-JSON to 6-class YOLO converter, group-aware splitter, and visual check generator."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from Src.dataset_audit import resolve_dataset_path

try:
    from PIL import Image, ImageDraw, ImageOps
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Conservative TACO category -> EcoSort 6-class mapping (uncertain/composite/hazardous -> None = deferred)
TACO_TO_ECOSORT: dict[str, str | None] = {
    "Cardboard": "cardboard",
    "Corrugated carton": "cardboard",
    "Egg carton": "cardboard",
    "Toilet tube": "cardboard",
    "Paper bag": "paper",
    "Paper cup": "paper",
    "Paper straw": "paper",
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
    "Food Can": "metal",
    "Food can": "metal",
    "Aerosol": "metal",
    "Metal bottle cap": "metal",
    "Metal lid": "metal",
    "Pop tab": "metal",
    "Scrap metal": "metal",
    "Clear plastic bottle": "plastic",
    "Other plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Plastic lid": "plastic",
    "Plastic film": "plastic",
    "Single-use carrier bag": "plastic",
    "Garbage bag": "plastic",
    "Polypropylene bag": "plastic",
    "Plastic straw": "plastic",
    "Plastic utensils": "plastic",
    "Disposable plastic cup": "plastic",
    "Other plastic cup": "plastic",
    "Other plastic container": "plastic",
    "Other plastic": "plastic",
    "Tupperware": "plastic",
    "Spread tub": "plastic",
    "Six pack rings": "plastic",
    "Other plastic wrapper": None,
    "Crisp packet": None,
    "Cigarette": "trash",
    "Styrofoam piece": "trash",
    "Foam food container": "trash",
    "Foam cup": "trash",
    "Food waste": "trash",
    "Shoe": "trash",
    # Deferred pending Prateek/supervisor material & unknown-rejection policy:
    "Drink carton": None,
    "Tetrapak": None,
    "Other carton": None,
    "Meal carton": None,
    "Pizza box": None,
    "Disposable food container": None,
    "Carded blister pack": None,
    "Squeezable tube": None,
    "Plastic glooves": None,
    "Rope & strings": None,
    "Battery": None,
    "Unlabeled litter": None,
}


def coco_bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = map(float, bbox)
        if not all(math.isfinite(v) for v in (x, y, w, h, img_w, img_h)):
            return None
    except (TypeError, ValueError):
        return None
    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(img_w), x + w), min(float(img_h), y + h)
    if right <= left or bottom <= top:
        return None
    return ((left + right) / (2 * img_w), (top + bottom) / (2 * img_h),
            (right - left) / img_w, (bottom - top) / img_h)


def safe_output_stem(rel_name: str, img_id: int) -> str:
    rel = Path(rel_name)
    parent_part = rel.parent.as_posix().replace("/", "_")
    if parent_part in ("", "."):
        return f"{rel.stem}_id{img_id}"
    return f"{parent_part}_{rel.stem}_id{img_id}"


def _convert_coco_to_yolo(
    coco_json: Path,
    images_root: Path,
    output_dir: Path,
    seed: int = 42,
    visual_samples: int = 12,
    require_images: bool = False,
) -> int:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    cat_id_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    images_by_id = {img["id"]: img for img in data.get("images", [])}
    if Image is None or not images_by_id:
        print("ERROR: Pillow and a nonempty image inventory are required.")
        return 1
    # Preflight every source before creating any output, even with --visual-samples 0.
    try:
        for img in images_by_id.values():
            source = resolve_dataset_path(img["file_name"], images_root)
            with Image.open(source) as image:
                image.load()
                if ImageOps.exif_transpose(image).size != (img["width"], img["height"]):
                    raise ValueError(f"dimension mismatch: {img['file_name']}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: Source image validation failed: {exc}")
        return 1

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        anns_by_image[ann["image_id"]].append(ann)

    # Explicit group IDs are preferred; folder grouping remains a heuristic.
    groups: dict[str, list[int]] = defaultdict(list)
    for img_id, img_info in images_by_id.items():
        annotations = anns_by_image.get(img_id, [])
        # Hold out the whole image to avoid learning deferred objects as background.
        if not annotations or any(
            TACO_TO_ECOSORT.get(cat_id_to_name.get(ann["category_id"])) is None
            or coco_bbox_to_yolo(ann.get("bbox"), img_info["width"], img_info["height"]) is None
            for ann in annotations
        ):
            continue
        rel = Path(img_info["file_name"])
        group_key = str(img_info.get("group_id") or (rel.parent.as_posix() if rel.parent.as_posix() != "." else rel.stem.split("_")[0]))
        groups[group_key].append(img_id)

    rng = random.Random(seed)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    if n_groups < 3:
        print("ERROR: At least three eligible groups are required for train/val/test export.")
        return 1
    if output_dir.exists() and any(output_dir.iterdir()):
        print("ERROR: Choose a fresh output directory to avoid mixing old and new splits.")
        return 1
    n_train = min(n_groups - 2, max(1, int(0.7 * n_groups)))
    n_val = min(n_groups - n_train - 1, max(1, int(0.15 * n_groups)))
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
    copied_images = 0
    visual_dir = output_dir / "visual_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0

    if visual_samples > 0 and Image is None:
        print("WARNING: Pillow (PIL) is not installed; skipping visual check overlays (install via `pip install pillow`).")

    for g, img_ids in groups.items():
        split = split_for_group[g]
        label_dir = output_dir / "labels" / split
        image_dir = output_dir / "images" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        for img_id in img_ids:
            img_info = images_by_id[img_id]
            rel_name = img_info["file_name"]
            src_img = resolve_dataset_path(rel_name, images_root)

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
                draw_boxes.append(([(xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h], mapped))

            if not yolo_lines:
                continue

            out_stem = safe_output_stem(rel_name, img_id)
            (label_dir / f"{out_stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
            split_image_counts[split] += 1

            if src_img.is_file():
                dst_img = image_dir / f"{out_stem}{src_img.suffix.lower()}"
                if not dst_img.exists():
                    with Image.open(src_img) as original:
                        if original.getexif().get(274, 1) == 1:
                            shutil.copy2(src_img, dst_img)
                        else:
                            corrected = ImageOps.exif_transpose(original)
                            corrected.save(dst_img)
                copied_images += 1

                if Image is not None and ImageDraw is not None and rendered < visual_samples:
                    with Image.open(src_img) as im:
                        canvas = ImageOps.exif_transpose(im).convert("RGB")
                        draw = ImageDraw.Draw(canvas)
                        for (bx, by, bw_px, bh_px), lbl in draw_boxes:
                            draw.rectangle([bx, by, bx + bw_px, by + bh_px], outline="lime", width=3)
                            draw.text((bx + 4, max(0.0, by - 14)), lbl, fill="lime")
                        canvas.save(visual_dir / f"{split}_{out_stem}_check.jpg")
                    rendered += 1

    yaml_text = (
        f"path: {json.dumps(output_dir.resolve().as_posix())}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print(f"Converted images per split: {dict(split_image_counts)}")
    print(f"Images copied to YOLO folders: {copied_images}")
    print(f"Converted boxes per class: {dict(box_counts)}")
    print(f"Deferred/unmapped category boxes: {dict(deferred_cats)}")
    print(f"Visual check overlays saved: {rendered} -> {visual_dir}")
    print("Mapping version: reviewed-defer-wrappers-2026-11-19; images with deferred/invalid boxes held out.")
    print("Group disjointness follows supplied IDs/folder heuristic; item/session independence requires review.")
    return 0


def convert_coco_to_yolo(
    coco_json: Path,
    images_root: Path,
    output_dir: Path,
    seed: int = 42,
    visual_samples: int = 12,
    require_images: bool = False,
) -> int:
    """Publish a complete export from staging, preserving previous exports."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        print("ERROR: Output already exists; choose a new export directory.")
        return 1
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ecosort-export-", dir=output_dir.parent) as tmp:
        staged = Path(tmp) / "dataset"
        result = _convert_coco_to_yolo(coco_json, images_root, staged, seed, visual_samples, require_images)
        if result:
            return result
        # The export's root changes when the staged directory is published.
        yaml_path = staged / "data.yaml"
        yaml_text = yaml_path.read_text(encoding="utf-8").splitlines()
        yaml_text[0] = f"path: {json.dumps(output_dir.as_posix())}"
        yaml_path.write_text("\n".join(yaml_text) + "\n", encoding="utf-8")
        for split in ("train", "val", "test"):
            images = {path.stem for path in (staged / "images" / split).iterdir()}
            labels = {path.stem for path in (staged / "labels" / split).glob("*.txt")}
            if not images or images != labels:
                print(f"ERROR: Incomplete image/label pairs in {split}.")
                return 1
        staged.rename(output_dir)
    print(f"Published complete dataset: {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert COCO/TACO annotations to 6-class YOLO labels with visual checks.")
    parser.add_argument("coco_json", type=Path, help="Path to COCO/TACO annotations.json")
    parser.add_argument("images_root", type=Path, help="Root directory containing images")
    parser.add_argument("--out", type=Path, default=Path("Data/yolo_6class"), help="Output directory (gitignored)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group-aware split")
    parser.add_argument("--visual-samples", type=int, default=12, help="Number of visual bounding-box check images")
    parser.add_argument("--require-images", action="store_true", help="Compatibility option; source images are always required")
    args = parser.parse_args()
    return convert_coco_to_yolo(
        args.coco_json,
        args.images_root,
        args.out,
        args.seed,
        args.visual_samples,
        args.require_images,
    )


if __name__ == "__main__":
    raise SystemExit(main())
