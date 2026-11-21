import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import detector_fixtures as fx
from Src.detector import app_logic
from Src.detector.classes import CLASS_NAMES, GUIDANCE
from Src.detector.e6_detector import DetectorError, E6Detector, load_protocol, load_rgb_image
from Src.detector.metrics import iou_xyxy, match_boxes
from Src.detector.render import CLASS_COLORS, count_by_class, draw_detections


def make_detector(tmp: Path, output=None, names=None, **protocol_overrides) -> E6Detector:
    ckpt, proto = fx.make_checkpoint_and_protocol(tmp, **protocol_overrides)
    stub = fx.StubModel(output, names)
    return E6Detector(ckpt, proto, model_loader=lambda _p: stub)


class ProtocolTests(unittest.TestCase):
    def test_valid_protocol_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, proto = fx.make_checkpoint_and_protocol(Path(tmp))
            self.assertEqual(load_protocol(proto)["imgsz"], 640)

    def test_bad_protocols_rejected(self):
        bad_values = [
            {"class_names": list(reversed(CLASS_NAMES))},
            {"checkpoint_sha256": "ABC"},
            {"conf_threshold": 1.5},
            {"conf_threshold": True},
            {"imgsz": 10},
            {"iou_threshold": 0},
            {"protocol_version": "other"},
            {"max_det": 0},
        ]
        for override in bad_values:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as tmp:
                _, proto = fx.make_checkpoint_and_protocol(Path(tmp), **override)
                with self.assertRaises(DetectorError):
                    load_protocol(proto)

    def test_missing_or_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DetectorError):
                load_protocol(Path(tmp) / "nope.json")
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json")
            with self.assertRaises(DetectorError):
                load_protocol(bad)


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.img = self.tmp / "photo.jpg"
        fx.write_image(self.img, (200, 100), seed=1)

    def tearDown(self):
        self._tmp.cleanup()

    def test_checkpoint_hash_mismatch_rejected(self):
        ckpt, proto = fx.make_checkpoint_and_protocol(self.tmp / "m")
        ckpt.write_bytes(ckpt.read_bytes() + b"tampered")
        with self.assertRaisesRegex(DetectorError, "SHA-256"):
            E6Detector(ckpt, proto, model_loader=lambda _p: fx.StubModel())

    def test_missing_checkpoint_rejected(self):
        _, proto = fx.make_checkpoint_and_protocol(self.tmp / "m")
        with self.assertRaisesRegex(DetectorError, "not found"):
            E6Detector(self.tmp / "missing.pt", proto, model_loader=lambda _p: fx.StubModel())

    def test_wrong_model_class_order_rejected(self):
        names = {0: "glass", 1: "cardboard", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}
        with self.assertRaisesRegex(DetectorError, "class names"):
            make_detector(self.tmp / "m", names=names)

    def test_missing_ultralytics_gives_clear_error(self):
        ckpt, proto = fx.make_checkpoint_and_protocol(self.tmp / "m")
        import sys
        saved = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = None  # makes `import ultralytics` raise ImportError
        try:
            with self.assertRaisesRegex(DetectorError, "ultralytics"):
                E6Detector(ckpt, proto)
        finally:
            if saved is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = saved

    def test_detections_sorted_clipped_and_named(self):
        out = lambda im: ([[10, 10, 60, 50], [-20, -5, 40, 30], [100, 20, 190, 95]], [0.4, 0.9, 0.6], [4, 1, 0])
        det = make_detector(self.tmp / "m", out)
        result = det.detect(self.img)
        self.assertEqual([d.class_name for d in result.detections], ["glass", "cardboard", "plastic"])
        self.assertEqual(result.detections[0].box_xyxy, (0.0, 0.0, 40.0, 30.0))  # clipped to the image
        self.assertEqual((result.image_width, result.image_height), (200, 100))
        self.assertFalse(result.calibrated)
        self.assertEqual(result.dropped_invalid, 0)

    def test_invalid_model_output_is_dropped_not_shown(self):
        out = lambda im: (
            [[10, 10, 60, 50], [30, 30, 30, 80], [float("nan"), 0, 10, 10], [1, 1, 20, 20], [5, 5, 50, 50]],
            [0.8, 0.8, 0.8, 0.8, 1.7],
            [0, 0, 0, 9, 0],
        )
        result = make_detector(self.tmp / "m", out).detect(self.img)
        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.dropped_invalid, 4)

    def test_no_detections_returns_empty_list_not_fake_box(self):
        result = make_detector(self.tmp / "m").detect(self.img)
        self.assertEqual(result.detections, ())
        self.assertEqual(result.to_dict()["detections"], [])

    def test_settings_forwarded_and_conf_override_validated(self):
        det = make_detector(self.tmp / "m")
        det.detect(self.img)
        call = det.model.calls[-1]
        self.assertEqual((call["imgsz"], call["conf"], call["iou"], call["max_det"]), (640, 0.25, 0.6, 100))
        det.detect(self.img, conf=0.5)
        self.assertEqual(det.model.calls[-1]["conf"], 0.5)
        for bad in (0, 1, -0.2, float("nan")):
            with self.assertRaises(DetectorError):
                det.detect(self.img, conf=bad)

    def test_bad_images_rejected(self):
        det = make_detector(self.tmp / "m")
        with self.assertRaisesRegex(DetectorError, "not found"):
            det.detect(self.tmp / "missing.jpg")
        broken = self.tmp / "broken.jpg"
        broken.write_bytes(b"garbage")
        with self.assertRaisesRegex(DetectorError, "Could not read"):
            det.detect(broken)
        tiny = self.tmp / "tiny.png"
        Image.new("RGB", (4, 4)).save(tiny)
        with self.assertRaisesRegex(DetectorError, "too small"):
            det.detect(tiny)

    def test_inference_failure_is_reported(self):
        def boom(_image):
            raise RuntimeError("cuda out of memory")

        with self.assertRaisesRegex(DetectorError, "inference failed"):
            make_detector(self.tmp / "m", boom).detect(self.img)

    def test_exif_orientation_applied(self):
        img = Image.new("RGB", (120, 60), (10, 20, 30))
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 270 -> becomes portrait
        path = self.tmp / "rot.jpg"
        img.save(path, exif=exif)
        self.assertEqual(load_rgb_image(path).size, (60, 120))

    def test_to_dict_is_json_ready_with_normalised_boxes(self):
        out = lambda im: ([[20, 10, 120, 60]], [0.75], [2])
        data = make_detector(self.tmp / "m", out).detect(self.img).to_dict()
        json.dumps(data)
        det = data["detections"][0]
        self.assertEqual(det["class_name"], "metal")
        self.assertEqual(det["box_xyxy_norm"], [0.1, 0.1, 0.6, 0.6])
        self.assertIs(data["calibrated"], False)
        self.assertIn("checkpoint_sha256", data)


class RenderAndMetricTests(unittest.TestCase):
    def _detections(self):
        out = lambda im: ([[10, 10, 60, 50], [70, 20, 120, 80]], [0.9, 0.5], [4, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jpg"
            fx.write_image(path, (200, 100), 3)
            return make_detector(Path(tmp) / "m", out).detect(path).detections

    def test_counts_include_all_classes(self):
        counts = count_by_class(self._detections())
        self.assertEqual(set(counts), set(CLASS_NAMES))
        self.assertEqual((counts["plastic"], counts["glass"], counts["metal"]), (1, 1, 0))

    def test_draw_changes_only_when_detections_exist(self):
        image = Image.fromarray(np.full((100, 200, 3), 128, dtype=np.uint8))
        self.assertEqual(np.asarray(draw_detections(image, [])).tolist(), np.asarray(image).tolist())
        drawn = draw_detections(image, self._detections())
        self.assertEqual(drawn.size, image.size)
        self.assertNotEqual(np.asarray(drawn).tolist(), np.asarray(image).tolist())
        px = drawn.getpixel((35, 10))  # on the top edge of the plastic box (x 10..60)
        self.assertEqual(px, CLASS_COLORS["plastic"])
        self.assertEqual(image.getpixel((35, 10)), (128, 128, 128))  # input not modified

    def test_iou_and_matching(self):
        self.assertAlmostEqual(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertAlmostEqual(iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        gt = [(1, 0, 0, 10, 10), (2, 20, 20, 40, 40)]
        pred = [(1, 0.9, 0, 0, 10, 10), (3, 0.8, 20, 20, 40, 40), (1, 0.7, 0, 0, 10, 10)]
        matches, fps, fns = match_boxes(gt, pred, 0.5)
        self.assertEqual(matches, [(0, 0, 1.0)])
        self.assertEqual(fps, [1, 2])      # wrong class, and a duplicate of an already-matched box
        self.assertEqual(fns, [1])         # the class-2 object was missed

    def test_low_iou_is_both_fp_and_fn(self):
        matches, fps, fns = match_boxes([(0, 0, 0, 10, 10)], [(0, 0.9, 6, 0, 16, 10)], 0.5)
        self.assertEqual((matches, fps, fns), ([], [0], [0]))


class AppLogicTests(unittest.TestCase):
    JPEG = b"\xff\xd8\xff" + b"\x00" * 20
    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20

    def test_upload_validation(self):
        app_logic.validate_upload("a.jpg", self.JPEG)
        app_logic.validate_upload("A.PNG", self.PNG)
        cases = [("a.gif", self.JPEG), ("a.jpg", b""), ("a.jpg", b"x" * 20), ("a.png", self.JPEG[:2]),
                 ("a.jpg", self.JPEG + b"0" * app_logic.MAX_UPLOAD_BYTES)]
        for name, data in cases:
            with self.subTest(name=name, size=len(data)), self.assertRaises(app_logic.UploadError):
                app_logic.validate_upload(name, data)

    def test_run_detection_builds_consistent_outcome(self):
        buffer = io.BytesIO()
        fx.write_image(Path(tempfile.gettempdir()) / "ecoshot_unit.jpg", (160, 120), 5)
        Image.open(Path(tempfile.gettempdir()) / "ecoshot_unit.jpg").save(buffer, format="JPEG")
        data = buffer.getvalue()
        out = lambda im: ([[10, 10, 80, 70], [90, 30, 150, 110]], [0.8, 0.6], [4, 4])
        with tempfile.TemporaryDirectory() as tmp:
            outcome = app_logic.run_detection(make_detector(Path(tmp), out), data)
        self.assertEqual(outcome.counts["plastic"], 2)
        self.assertEqual([row["#"] for row in outcome.object_rows], [1, 2])
        self.assertEqual([g["category"] for g in outcome.guidance_rows], ["plastic"])
        self.assertEqual(outcome.guidance_rows[0]["advice"], GUIDANCE["plastic"])
        self.assertTrue(outcome.annotated_png.startswith(b"\x89PNG"))
        report = app_logic.build_report(outcome, "x.jpg", data)
        json.dumps(report)
        self.assertEqual(len(report["source_file"]["sha256"]), 64)
        self.assertIs(report["calibrated"], False)

    def test_zero_detections_produces_no_guidance(self):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 64), (1, 2, 3)).save(buffer, format="PNG")
        with tempfile.TemporaryDirectory() as tmp:
            outcome = app_logic.run_detection(make_detector(Path(tmp)), buffer.getvalue())
        self.assertEqual((outcome.object_rows, outcome.guidance_rows), ([], []))

    def test_undecodable_bytes_raise_detector_error(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(DetectorError):
            app_logic.run_detection(make_detector(Path(tmp)), self.JPEG)


if __name__ == "__main__":
    unittest.main()
