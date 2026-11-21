import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from Src.evaluate_keras import audit_dataset_manifest, classification_metrics


class EvaluationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.csv_path = self.root / "manifest.csv"
        self.zip_path = self.root / "dataset.zip"

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def image_bytes(color):
        buffer = io.BytesIO()
        Image.new("RGB", (12, 10), color).save(buffer, format="PNG")
        return buffer.getvalue()

    def write_dataset(self, rows, image_data, include_group=True):
        fields = ["split", "label", "source_archive_path"]
        if include_group:
            fields.append("group_id")

        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        with zipfile.ZipFile(self.zip_path, "w") as archive:
            for member, data in image_data.items():
                archive.writestr(member, data)

    def test_valid_manifest_passes_hash_and_group_audit(self):
        rows = [
            {
                "split": "train",
                "label": "cardboard",
                "source_archive_path": "dataset-resized/cardboard/a.png",
                "group_id": "item-a",
            },
            {
                "split": "test",
                "label": "glass",
                "source_archive_path": "dataset-resized/glass/b.png",
                "group_id": "item-b",
            },
        ]
        images = {
            "split/train/cardboard/a.png": self.image_bytes((10, 20, 30)),
            "split/test/glass/b.png": self.image_bytes((30, 20, 10)),
        }
        self.write_dataset(rows, images)

        records, audit = audit_dataset_manifest(self.csv_path, self.zip_path)

        self.assertEqual(len(records), 2)
        self.assertEqual(audit["manifest_rows"], 2)
        self.assertEqual(audit["missing_files"], 0)
        self.assertEqual(audit["decode_errors"], 0)
        self.assertEqual(audit["cross_split_exact_duplicates"], 0)
        self.assertEqual(audit["cross_split_group_leaks"], 0)
        self.assertEqual(audit["group_leakage_check"], "performed")

    def test_reports_group_leakage(self):
        rows = [
            {
                "split": "train",
                "label": "cardboard",
                "source_archive_path": "dataset-resized/cardboard/a.png",
                "group_id": "same-object",
            },
            {
                "split": "test",
                "label": "cardboard",
                "source_archive_path": "dataset-resized/cardboard/b.png",
                "group_id": "same-object",
            },
        ]
        images = {
            "split/train/cardboard/a.png": self.image_bytes((10, 20, 30)),
            "split/test/cardboard/b.png": self.image_bytes((30, 20, 10)),
        }
        self.write_dataset(rows, images)

        with self.assertRaisesRegex(ValueError, "groups cross dataset splits"):
            audit_dataset_manifest(self.csv_path, self.zip_path)

    def test_reports_exact_duplicate_bytes_across_splits(self):
        duplicate = self.image_bytes((70, 80, 90))
        rows = [
            {
                "split": "train",
                "label": "cardboard",
                "source_archive_path": "dataset-resized/cardboard/a.png",
                "group_id": "train-a",
            },
            {
                "split": "test",
                "label": "cardboard",
                "source_archive_path": "dataset-resized/cardboard/b.png",
                "group_id": "test-b",
            },
        ]
        images = {
            "split/train/cardboard/a.png": duplicate,
            "split/test/cardboard/b.png": duplicate,
        }
        self.write_dataset(rows, images)

        with self.assertRaisesRegex(ValueError, "cross dataset splits"):
            audit_dataset_manifest(self.csv_path, self.zip_path)

    def test_rejects_missing_image(self):
        rows = [{
            "split": "test",
            "label": "cardboard",
            "source_archive_path": "dataset-resized/cardboard/missing.png",
            "group_id": "missing",
        }]
        self.write_dataset(rows, {})

        with self.assertRaisesRegex(ValueError, "images are missing"):
            audit_dataset_manifest(self.csv_path, self.zip_path)

    def test_rejects_source_path_label_mismatch(self):
        rows = [{
            "split": "test",
            "label": "glass",
            "source_archive_path": "dataset-resized/plastic/example.png",
            "group_id": "mismatch",
        }]
        self.write_dataset(rows, {})

        with self.assertRaisesRegex(ValueError, "does not match label"):
            audit_dataset_manifest(self.csv_path, self.zip_path)

    def test_reports_when_group_ids_are_not_available(self):
        rows = [{
            "split": "test",
            "label": "cardboard",
            "source_archive_path": "dataset-resized/cardboard/a.png",
        }]
        images = {
            "split/test/cardboard/a.png": self.image_bytes((1, 2, 3))
        }
        self.write_dataset(rows, images, include_group=False)

        _, audit = audit_dataset_manifest(self.csv_path, self.zip_path)

        self.assertFalse(audit["group_id_column_present"])
        self.assertIn("not available", audit["group_leakage_check"])

    def test_calculates_accuracy_macro_metrics_and_confusion_matrix(self):
        labels = [0, 0, 1, 1]
        predictions = [0, 1, 1, 1]

        result = classification_metrics(labels, predictions)

        self.assertEqual(result["sample_count"], 4)
        self.assertEqual(result["accuracy"], 0.75)
        self.assertEqual(result["confusion_matrix"][0][:2], [1, 1])
        self.assertEqual(result["confusion_matrix"][1][:2], [0, 2])
        self.assertEqual(result["per_class"]["cardboard"]["recall"], 0.5)
        self.assertEqual(result["per_class"]["glass"]["recall"], 1.0)
        self.assertEqual(result["per_class"]["metal"]["support"], 0)

    def test_blank_group_column_is_not_a_completed_audit(self):
        rows = [{"split": "test", "label": "glass",
                 "source_archive_path": "dataset-resized/glass/a.png", "group_id": ""}]
        self.write_dataset(rows, {"split/test/glass/a.png": self.image_bytes((4, 5, 6))})
        _, audit = audit_dataset_manifest(self.csv_path, self.zip_path)
        self.assertIn("not available", audit["group_leakage_check"])
        self.assertEqual(audit["rows_without_group_id"], 1)

    def test_partial_group_coverage_is_reported(self):
        rows = [{"split": "test", "label": "glass",
                 "source_archive_path": "dataset-resized/glass/a.png", "group_id": "g1"},
                {"split": "test", "label": "glass",
                 "source_archive_path": "dataset-resized/glass/b.png", "group_id": ""}]
        self.write_dataset(rows, {
            "split/test/glass/a.png": self.image_bytes((4, 5, 6)),
            "split/test/glass/b.png": self.image_bytes((6, 5, 4)),
        })
        _, audit = audit_dataset_manifest(self.csv_path, self.zip_path)
        self.assertIn("not available", audit["group_leakage_check"])
        self.assertEqual(audit["rows_without_group_id"], 1)

    def test_rejects_empty_metric_input(self):
        with self.assertRaisesRegex(ValueError, "empty split"):
            classification_metrics([], [])


if __name__ == "__main__":
    unittest.main()
