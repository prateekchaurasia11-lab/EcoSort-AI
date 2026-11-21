"""Regression fixtures for the reproduced PR #4 data failures."""
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from PIL import Image
from Src.dataset_audit import REQUIRED_COLUMNS, compute_sha256, verify_manifest
from Src.yolo_prepare import coco_bbox_to_yolo, convert_coco_to_yolo


class ManifestChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "dataset"
        self.root.mkdir()
        self.source = self.root / "source.jpg"
        self.source.write_bytes(b"fixture")
        self.row = dict(relative_path="source.jpg", sha256=compute_sha256(self.source),
                        source_or_license="synthetic", label="glass", review_decision="approve",
                        group_id="g1", split="train")
        self.manifest = self.root.parent / "manifest.csv"

    def write(self):
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerow(self.row)

    def test_valid_bytes(self):
        self.write()
        self.assertEqual(verify_manifest(self.manifest, self.root), 0)

    def test_traversal_and_absolute(self):
        outside = self.root.parent / "outside.jpg"
        outside.write_bytes(b"fixture")
        for path in ("../outside.jpg", str(outside), r"C:\outside.jpg", r"..\outside.jpg"):
            with self.subTest(path=path):
                self.row["relative_path"] = path
                self.write()
                self.assertEqual(verify_manifest(self.manifest, self.root), 1)

    def test_symlink_escape(self):
        outside = self.root.parent / "outside.jpg"
        outside.write_bytes(b"fixture")
        (self.root / "link.jpg").symlink_to(outside)
        self.row["relative_path"] = "link.jpg"
        self.write()
        self.assertEqual(verify_manifest(self.manifest, self.root), 1)

    def test_protected_collision(self):
        self.write()
        protected = self.root.parent / "protected.csv"
        protected.write_text("sha256\n" + self.row["sha256"] + "\n")
        self.assertEqual(verify_manifest(self.manifest, self.root, protected), 1)
        protected.write_text("sha256\n" + "a" * 64 + "\n")
        self.assertEqual(verify_manifest(self.manifest, self.root, protected), 0)

    def test_existing_rejection_vocabulary(self):
        for decision in ("reject_wrong_label", "reject_bad_image", "defer_wrapper"):
            with self.subTest(decision=decision):
                self.row.update(review_decision=decision, split="unassigned")
                self.write()
                self.assertEqual(verify_manifest(self.manifest), 0)
                self.row["split"] = "train"
                self.write()
                self.assertEqual(verify_manifest(self.manifest), 1)


class YoloChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.data = dict(categories=[dict(id=1, name="Glass bottle"), dict(id=2, name="Crisp packet")],
                         images=[], annotations=[])
        for i in range(4):
            file = f"batch{i}/image.jpg"
            image = self.images / file
            image.parent.mkdir()
            Image.new("RGB", (100, 100), (i * 50, 0, 0)).save(image)
            self.data["images"].append(dict(id=i, file_name=file, width=100, height=100))
            self.data["annotations"].append(dict(id=i, image_id=i, category_id=1, bbox=[-10, 0, 20, 10]))
        self.coco = self.root / "annotations.json"
        self.output = self.root / "output"

    def run_export(self):
        self.coco.write_text(json.dumps(self.data))
        return convert_coco_to_yolo(self.coco, self.images, self.output, visual_samples=0)

    def test_clipping_and_invalid_values(self):
        self.assertEqual(coco_bbox_to_yolo([-10, 0, 20, 10], 100, 100), (.05, .05, .1, .1))
        for bbox in ([110, 0, 10, 10], [0, 0, 0, 10], [0, 0, float("nan"), 10],
                     [0, 0, float("inf"), 10], [1, 2], ["bad", 0, 10, 10]):
            with self.subTest(bbox=bbox):
                self.assertIsNone(coco_bbox_to_yolo(bbox, 100, 100))

    def test_three_groups_have_all_splits_and_pairs(self):
        self.data["images"] = self.data["images"][:3]
        self.data["annotations"] = self.data["annotations"][:3]
        self.assertEqual(self.run_export(), 0)
        for split in ("train", "val", "test"):
            images = list((self.output / "images" / split).glob("*.jpg"))
            labels = list((self.output / "labels" / split).glob("*.txt"))
            self.assertEqual(len(images), 1)
            self.assertEqual({p.stem for p in images}, {p.stem for p in labels})
            self.assertEqual(labels[0].read_text().split()[1:], ["0.050000", "0.050000", "0.100000", "0.100000"])

    def test_missing_source_fails_before_output(self):
        (self.images / "batch0/image.jpg").unlink()
        self.assertEqual(self.run_export(), 1)
        self.assertFalse(self.output.exists())

    def test_corrupt_source_fails_before_output(self):
        (self.images / "batch0/image.jpg").write_bytes(b"invalid")
        self.assertEqual(self.run_export(), 1)
        self.assertFalse(self.output.exists())

    def test_wrong_dimensions(self):
        self.data["images"][0]["width"] = 99
        self.assertEqual(self.run_export(), 1)
        self.assertFalse(self.output.exists())

    def test_deferred_object_holds_entire_image(self):
        self.data["annotations"].append(dict(id=9, image_id=0, category_id=2, bbox=[20, 20, 10, 10]))
        self.assertEqual(self.run_export(), 0)
        self.assertEqual(len(list(self.output.glob("labels/*/*.txt"))), 3)
        self.assertFalse(any("id0" in p.name for p in self.output.glob("labels/*/*.txt")))

    def test_insufficient_groups(self):
        for image in self.data["images"]:
            image["group_id"] = "same_session"
        self.assertEqual(self.run_export(), 1)
        self.assertFalse(self.output.exists())

    def test_existing_output_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("previous export")
        self.assertEqual(self.run_export(), 1)
        self.assertEqual(marker.read_text(), "previous export")

    def test_copy_failure_does_not_publish_partial_export(self):
        with patch("Src.yolo_prepare.shutil.copy2", side_effect=OSError("fixture disk failure")):
            with self.assertRaises(OSError):
                self.run_export()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".ecosort-export-*")), [])

    def test_export_yaml_points_to_published_directory(self):
        self.assertEqual(self.run_export(), 0)
        root_line = (self.output / "data.yaml").read_text().splitlines()[0]
        self.assertEqual(root_line, "path: " + json.dumps(self.output.resolve().as_posix()))
