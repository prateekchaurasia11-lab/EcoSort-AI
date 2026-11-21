"""EXIF orientation regression coverage for TACO export."""
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageOps
from Src.yolo_prepare import convert_coco_to_yolo


class ExifExportTests(unittest.TestCase):
    def test_orientation_six_exports_upright_images_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = []
            annotations = []
            for index in range(3):
                folder = root / f"batch_{index}"
                folder.mkdir()
                image_path = folder / "image.jpg"
                original = Image.new("RGB", (80, 40), (10, 20, 30))
                exif = original.getexif()
                exif[274] = 6
                original.save(image_path, exif=exif)
                images.append({"id": index, "file_name": f"batch_{index}/image.jpg", "width": 40, "height": 80})
                annotations.append({"id": index, "image_id": index, "category_id": 1, "bbox": [5, 10, 20, 30]})
            coco = root / "annotations.json"
            coco.write_text(json.dumps({"images": images, "annotations": annotations,
                "categories": [{"id": 1, "name": "Glass bottle"}]}), encoding="utf-8")
            output = root / "export"
            self.assertEqual(convert_coco_to_yolo(coco, root, output, visual_samples=3), 0)
            exports = list((output / "images").rglob("*.jpg"))
            self.assertEqual(len(exports), 3)
            for path in exports:
                with Image.open(path) as result:
                    self.assertEqual(result.size, (40, 80))
                    self.assertEqual(ImageOps.exif_transpose(result).size, (40, 80))
                label = next((output / "labels").rglob(path.stem + ".txt"))
                self.assertEqual(label.read_text().strip(), "1 0.375000 0.312500 0.500000 0.375000")
            self.assertEqual(len(list((output / "visual_checks").glob("*.jpg"))), 3)

    def test_unreconciled_dimensions_fail_without_publishing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "batch"
            folder.mkdir()
            Image.new("RGB", (80, 40)).save(folder / "image.jpg")
            coco = root / "annotations.json"
            coco.write_text(json.dumps({"images": [{"id": 1, "file_name": "batch/image.jpg", "width": 40, "height": 80}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5]}],
                "categories": [{"id": 1, "name": "Glass bottle"}]}), encoding="utf-8")
            output = root / "export"
            self.assertEqual(convert_coco_to_yolo(coco, root, output, visual_samples=0), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
