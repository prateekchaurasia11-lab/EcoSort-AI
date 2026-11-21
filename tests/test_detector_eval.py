import json
import tempfile
import unittest
from pathlib import Path

import detector_fixtures as fx
from Src.detector import evaluate_detector as ev
from Src.detector import make_protocol as mp
from Src.detector.classes import CLASS_NAMES
from Src.detector.e6_detector import E6Detector, PROTOCOL_VERSION, load_protocol
from Src.detector.yolo_io import dataset_root, list_images, load_data_yaml, split_image_dirs


class OperatingPointTests(unittest.TestCase):
    """Uses a stub that replays ground truth, so only the evaluation arithmetic is tested."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.data = fx.make_dataset(self.root)
        cfg = load_data_yaml(self.data)
        self.test_images = list_images(split_image_dirs(cfg, dataset_root(cfg, self.data), "test"))

    def tearDown(self):
        self._tmp.cleanup()

    def _detector(self, answers):
        ckpt, proto = fx.make_checkpoint_and_protocol(self.root / "model")
        stub = fx.StubModel(lambda image: answers(image.size))
        return E6Detector(ckpt, proto, model_loader=lambda _p: stub)

    def _size_of(self, name):
        from PIL import Image
        with Image.open(next(p for p in self.test_images if p.name == name)) as im:
            return im.size

    def test_perfect_replay_gives_full_precision_and_recall(self):
        def answers(size):
            w, h = size
            if size == self._size_of("test_0.jpg"):
                return ([[0.3 * w, 0.3 * h, 0.7 * w, 0.7 * h]], [0.9], [1])
            return ([[0.2 * w, 0.2 * h, 0.4 * w, 0.4 * h], [0.6 * w, 0.6 * h, 0.8 * w, 0.8 * h]], [0.9, 0.8], [5, 3])

        result = ev.evaluate_operating_point(self._detector(answers), self.test_images, conf=None, match_iou=0.5, imgsz=640)
        self.assertEqual(result["overall"]["tp"], 3)
        self.assertEqual((result["overall"]["fp"], result["overall"]["fn"]), (0, 0))
        self.assertEqual(result["overall"]["precision"], 1.0)
        self.assertEqual(result["per_class"]["trash"]["recall"], 1.0)
        self.assertIsNone(result["per_class"]["metal"]["recall"])  # no ground truth: not reported as 0
        self.assertEqual(result["_errors"], [])

    def test_misses_and_false_detections_are_listed(self):
        def answers(size):
            w, h = size
            if size == self._size_of("test_0.jpg"):
                return ([], [], [])  # miss the glass bottle
            return ([[0.2 * w, 0.2 * h, 0.4 * w, 0.4 * h], [0.0, 0.0, 0.1 * w, 0.1 * h]], [0.9, 0.7], [5, 2])

        result = ev.evaluate_operating_point(self._detector(answers), self.test_images, conf=None, match_iou=0.5, imgsz=640)
        self.assertEqual((result["overall"]["tp"], result["overall"]["fp"], result["overall"]["fn"]), (1, 1, 2))
        self.assertEqual(result["overall"]["precision"], 0.5)
        kinds = sorted(row["kind"] for row in result["_errors"])
        self.assertEqual(kinds, ["false_detection", "missed_detection", "missed_detection"])
        self.assertEqual(result["images_with_any_error"], 2)
        per_image = {row["image"]: row for row in result["_per_image"]}
        self.assertEqual(per_image["test_0.jpg"]["false_negatives"], 1)

    def test_latency_excludes_warmup(self):
        result = ev.evaluate_operating_point(self._detector(lambda size: ([], [], [])), self.test_images * 3, conf=None, match_iou=0.5, imgsz=640)
        self.assertEqual(result["latency_ms"]["images_timed"], len(self.test_images) * 3 - ev.WARMUP_IMAGES)

    def test_cli_refuses_conf_override_on_test_split(self):
        ckpt, proto = fx.make_checkpoint_and_protocol(self.root / "model2")
        code = ev.main(["--checkpoint", str(ckpt), "--protocol", str(proto), "--data", str(self.data),
                        "--split", "test", "--conf", "0.1", "--out-dir", str(self.root / "out")])
        self.assertEqual(code, 1)

    def test_cli_reports_hash_failure(self):
        ckpt, proto = fx.make_checkpoint_and_protocol(self.root / "model3")
        ckpt.write_bytes(b"changed")
        code = ev.main(["--checkpoint", str(ckpt), "--protocol", str(proto), "--data", str(self.data),
                        "--out-dir", str(self.root / "out2")])
        self.assertEqual(code, 2)


class ProtocolBuilderTests(unittest.TestCase):
    def test_built_protocol_loads_and_records_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "best.pt"
            ckpt.write_bytes(b"abc")
            out = Path(tmp) / "protocol.json"
            self.assertEqual(mp.main(["--checkpoint", str(ckpt), "--out", str(out), "--conf", "0.3"]), 0)
            protocol = load_protocol(out)
            self.assertEqual(protocol["protocol_version"], PROTOCOL_VERSION)
            self.assertEqual(protocol["conf_threshold"], 0.3)
            self.assertEqual(protocol["class_names"], list(CLASS_NAMES))
            self.assertIs(json.loads(out.read_text())["calibrated"], False)

    def test_missing_checkpoint_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mp.main(["--checkpoint", str(Path(tmp) / "no.pt"), "--out", str(Path(tmp) / "p.json")]), 1)


if __name__ == "__main__":
    unittest.main()
