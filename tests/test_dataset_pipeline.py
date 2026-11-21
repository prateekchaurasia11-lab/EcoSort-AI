"""Regression tests for Src/dataset_audit.py and Src/yolo_prepare.py."""
from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from Src.dataset_audit import compute_sha256, verify_manifest
from Src.yolo_prepare import coco_bbox_to_yolo, convert_coco_to_yolo


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "relative_path",
        "sha256",
        "source_or_license",
        "label",
        "review_decision",
        "group_id",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DatasetAuditRegressionTests(unittest.TestCase):
    def test_valid_manifest_passes_with_disk_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "plastic").mkdir()
            img1 = root / "plastic" / "p1.jpg"
            img2 = root / "plastic" / "p2.jpg"
            img1.write_bytes(b"image-bytes-1")
            img2.write_bytes(b"image-bytes-2")

            manifest = root / "manifest.csv"
            _write_csv(
                manifest,
                [
                    {
                        "relative_path": "plastic/p1.jpg",
                        "sha256": compute_sha256(img1),
                        "source_or_license": "trashnet",
                        "label": "plastic",
                        "review_decision": "approved",
                        "group_id": "item_001",
                        "split": "train",
                    },
                    {
                        "relative_path": "plastic/p2.jpg",
                        "sha256": compute_sha256(img2),
                        "source_or_license": "local_capture",
                        "label": "plastic",
                        "review_decision": "deferred",
                        "group_id": "item_002",
                        "split": "unassigned",
                    },
                ],
            )
            self.assertEqual(verify_manifest(manifest, root), 0)

    def test_detects_group_leakage_across_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            _write_csv(
                manifest,
                [
                    {
                        "relative_path": "glass/g1.jpg",
                        "sha256": "a" * 64,
                        "source_or_license": "local",
                        "label": "glass",
                        "review_decision": "approved",
                        "group_id": "same_bottle_session",
                        "split": "train",
                    },
                    {
                        "relative_path": "glass/g2.jpg",
                        "sha256": "b" * 64,
                        "source_or_license": "local",
                        "label": "glass",
                        "review_decision": "approved",
                        "group_id": "same_bottle_session",
                        "split": "test",
                    },
                ],
            )
            self.assertEqual(verify_manifest(manifest), 1)

    def test_blocks_deferred_or_pending_in_splits_and_missing_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            _write_csv(
                manifest,
                [
                    {
                        "relative_path": "trash/w1.jpg",
                        "sha256": "c" * 64,
                        "source_or_license": "local",
                        "label": "trash",
                        "review_decision": "deferred",
                        "group_id": "grp_1",
                        "split": "val",
                    },
                    {
                        "relative_path": "paper/p1.jpg",
                        "sha256": "d" * 64,
                        "source_or_license": "local",
                        "label": "paper",
                        "review_decision": "approved",
                        "group_id": "",
                        "split": "train",
                    },
                ],
            )
            self.assertEqual(verify_manifest(manifest), 1)

    def test_detects_duplicate_sha256_and_invalid_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            _write_csv(
                manifest,
                [
                    {
                        "relative_path": "metal/m1.jpg",
                        "sha256": "e" * 64,
                        "source_or_license": "realwaste",
                        "label": "metal",
                        "review_decision": "approved",
                        "group_id": "g1",
                        "split": "train",
                    },
                    {
                        "relative_path": "metal/m1_resized.jpg",
                        "sha256": "e" * 64,
                        "source_or_license": "realwaste",
                        "label": "metal",
                        "review_decision": "auto_relabeled",
                        "group_id": "g2",
                        "split": "train",
                    },
                ],
            )
            self.assertEqual(verify_manifest(manifest), 1)


class YoloPrepareRegressionTests(unittest.TestCase):
    def test_bbox_conversion_and_deferred_drink_carton(self) -> None:
        self.assertIsNone(coco_bbox_to_yolo([10, 10, 0, 20], 100, 100))
        box = coco_bbox_to_yolo([20, 40, 40, 20], 200, 100)
        self.assertEqual(box, (0.2, 0.5, 0.2, 0.2))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coco = root / "annotations.json"
            coco.write_text(
                json.dumps(
                    {
                        "categories": [
                            {"id": 1, "name": "Glass bottle"},
                            {"id": 2, "name": "Drink carton"},
                        ],
                        "images": [
                            {"id": 10, "file_name": "batch_1/001.jpg", "width": 100, "height": 100},
                            {"id": 11, "file_name": "batch_2/001.jpg", "width": 100, "height": 100},
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [10, 10, 20, 30]},
                            {"id": 2, "image_id": 10, "category_id": 2, "bbox": [50, 50, 10, 10]},
                            {"id": 3, "image_id": 11, "category_id": 1, "bbox": [0, 0, 50, 50]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            out_dir = root / "yolo_out"
            rc = convert_coco_to_yolo(coco, root, out_dir, visual_samples=0)
            self.assertEqual(rc, 0)
            txt_files = list((out_dir / "labels").rglob("*.txt"))
            self.assertEqual(len(txt_files), 2)
            for f in txt_files:
                lines = f.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), 1)
                self.assertTrue(lines[0].startswith("1 "))


if __name__ == "__main__":
    unittest.main()
