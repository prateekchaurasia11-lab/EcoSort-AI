import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

import detector_fixtures as fx
from Src.detector import check_dataset as cd
from Src.detector.yolo_io import is_small_box, parse_label_file


class DatasetCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = fx.make_dataset(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def errors(self, **kw):
        report = cd.check_dataset(self.data, **kw)
        return report, " | ".join(report["errors"])

    def test_valid_dataset_passes_and_counts_boxes(self):
        report, text = self.errors()
        self.assertTrue(report["ok"], text)
        self.assertEqual(report["splits"]["train"]["images"], 6)
        self.assertEqual(report["splits"]["train"]["boxes"], 6)
        self.assertEqual(report["splits"]["test"]["boxes"], 3)
        self.assertEqual(report["class_boxes_total"]["plastic"], 2)
        self.assertTrue(any("group manifest" in w for w in report["warnings"]))

    def test_wrong_class_order_fails(self):
        cfg = yaml.safe_load(self.data.read_text())
        cfg["names"] = {0: "glass", 1: "cardboard", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}
        self.data.write_text(yaml.safe_dump(cfg))
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("names must be exactly", text)

    def test_empty_test_split_fails(self):
        shutil.rmtree(self.root / "images" / "test")
        shutil.rmtree(self.root / "labels" / "test")
        (self.root / "images" / "test").mkdir()
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("test: no images found", text)

    def test_missing_label_file_fails(self):
        (self.root / "labels" / "val" / "val_0.txt").unlink()
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("no label file", text)

    def test_box_outside_image_fails(self):
        fx.write_label(self.root / "labels" / "val" / "val_0.txt", [(0, 0.95, 0.5, 0.3, 0.3)])
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("extends outside", text)

    def test_class_id_out_of_range_fails(self):
        fx.write_label(self.root / "labels" / "val" / "val_0.txt", [(7, 0.5, 0.5, 0.2, 0.2)])
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("outside 0..5", text)

    def test_wrong_field_count_and_nan_fail(self):
        path = self.root / "labels" / "val" / "val_0.txt"
        path.write_text("0 0.5 0.5 0.2\n")
        self.assertFalse(self.errors()[0]["ok"])
        path.write_text("0 nan 0.5 0.2 0.2\n")
        self.assertFalse(self.errors()[0]["ok"])

    def test_orphan_label_fails(self):
        fx.write_label(self.root / "labels" / "train" / "ghost.txt", [(0, 0.5, 0.5, 0.2, 0.2)])
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("without a matching image", text)

    def test_cross_split_duplicate_is_leakage(self):
        shutil.copy(self.root / "images" / "train" / "train_0.jpg", self.root / "images" / "test" / "copy_of_train.jpg")
        fx.write_label(self.root / "labels" / "test" / "copy_of_train.txt", [(0, 0.5, 0.5, 0.4, 0.4)])
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("cross-split exact duplicate", text)

    def test_corrupt_image_fails(self):
        (self.root / "images" / "val" / "val_1.jpg").write_bytes(b"this is not an image")
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("cannot decode", text)

    def test_class_missing_from_train_fails(self):
        (self.root / "labels" / "train" / "train_5.txt").write_text("")
        report, text = self.errors()
        self.assertFalse(report["ok"])
        self.assertIn("class 'trash' has no boxes in train", text)

    def test_group_leakage_detected_via_manifest(self):
        manifest = self.root / "groups.csv"
        with manifest.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file_name", "group_id"])
            for p in sorted((self.root / "images").rglob("*.jpg")):
                writer.writerow([p.name, f"g_{p.stem}"])
            writer.writerow(["test_0.jpg", "g_train_0"])  # same scene as a train image
        report, text = self.errors(manifest_csv=manifest)
        self.assertFalse(report["ok"])
        self.assertIn("appears in several splits", text)

    def test_clean_group_manifest_passes_without_group_warning(self):
        manifest = self.root / "groups.csv"
        with manifest.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file_name", "group_id"])
            for p in sorted((self.root / "images").rglob("*.jpg")):
                writer.writerow([p.name, f"g_{p.stem}"])
        report, text = self.errors(manifest_csv=manifest)
        self.assertTrue(report["ok"], text)
        self.assertFalse(any("group manifest" in w for w in report["warnings"]))

    def test_cli_exit_codes_and_report_file(self):
        out = self.root / "report.json"
        self.assertEqual(cd.main(["--data", str(self.data), "--report", str(out)]), 0)
        self.assertTrue(json.loads(out.read_text())["ok"])
        (self.root / "labels" / "val" / "val_0.txt").unlink()
        self.assertEqual(cd.main(["--data", str(self.data)]), 1)

    def test_checker_does_not_modify_dataset(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        cd.check_dataset(self.data)
        after = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


class LabelParsingTests(unittest.TestCase):
    def test_small_box_rule_uses_scaled_size(self):
        self.assertTrue(is_small_box(10, 10, 640, 640, 640))
        self.assertFalse(is_small_box(40, 40, 640, 640, 640))
        # a 100 px box in a 3200 px wide photo shrinks to 20 px at 640
        self.assertTrue(is_small_box(100, 100, 3200, 2400, 640))

    def test_valid_label_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "a.txt"
            p.write_text("2 0.5 0.5 0.2 0.2\n\n5 0.1 0.1 0.2 0.2\n")
            rows, problems = parse_label_file(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
