import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from Src import predict_keras


class FakeModel:
    input_shape = (None, 224, 224, 3)
    output_shape = (None, 6)

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.last_input = None

    def predict(self, batch, verbose=0):
        self.last_input = np.asarray(batch)
        return self.scores.reshape(1, -1)


class E2PredictorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.checkpoint = self.root / "best.keras"
        self.protocol_path = self.root / "protocol.json"
        self.image_path = self.root / "sample.png"
        self.checkpoint.write_bytes(b"test-checkpoint-bytes")
        self.checkpoint_hash = hashlib.sha256(
            self.checkpoint.read_bytes()
        ).hexdigest()
        Image.new("RGB", (32, 24), (20, 80, 140)).save(self.image_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def protocol(self):
        return {
            "classes": predict_keras.CLASSES.copy(),
            "architecture": "MobileNetV2",
            "input_size": [224, 224],
            "preprocessing": (
                "RGB 0..255; embedded Rescaling(1/127.5, offset=-1)"
            ),
        }

    def write_protocol(self, values=None):
        payload = self.protocol()
        if values:
            payload.update(values)
        self.protocol_path.write_text(json.dumps(payload), encoding="utf-8")

    def create_predictor(self, scores=None, model=None):
        self.write_protocol()
        fake_model = model or FakeModel(
            scores if scores is not None else [0.80, 0.05, 0.05, 0.04, 0.04, 0.02]
        )
        with (
            mock.patch.object(predict_keras, "EXPECTED_SHA256", self.checkpoint_hash),
            mock.patch.object(
                predict_keras.tf.keras.models,
                "load_model",
                return_value=fake_model,
            ),
        ):
            predictor = predict_keras.E2Predictor(
                self.checkpoint, self.protocol_path
            )
        return predictor, fake_model

    def test_valid_prediction_returns_class_scores_and_metadata(self):
        predictor, fake_model = self.create_predictor()
        result = predictor.predict(self.image_path)

        self.assertEqual(result["class"], "cardboard")
        self.assertAlmostEqual(result["model_score"], 0.80, places=5)
        self.assertEqual(list(result["scores"]), predict_keras.CLASSES)
        self.assertFalse(result["calibrated"])
        self.assertEqual(result["checkpoint_sha256"], self.checkpoint_hash)
        self.assertEqual(fake_model.last_input.shape, (1, 224, 224, 3))
        self.assertEqual(fake_model.last_input.dtype, np.float32)

    def test_rejects_wrong_class_order(self):
        payload = self.protocol()
        payload["classes"] = list(reversed(payload["classes"]))
        self.write_protocol(payload)

        with self.assertRaisesRegex(ValueError, "class order"):
            predict_keras.E2Predictor(self.checkpoint, self.protocol_path)

    def test_rejects_wrong_architecture(self):
        self.write_protocol({"architecture": "ResNet18"})

        with self.assertRaisesRegex(ValueError, "architecture"):
            predict_keras.E2Predictor(self.checkpoint, self.protocol_path)

    def test_rejects_wrong_input_size_in_protocol(self):
        self.write_protocol({"input_size": [299, 299]})

        with self.assertRaisesRegex(ValueError, "input size"):
            predict_keras.E2Predictor(self.checkpoint, self.protocol_path)

    def test_rejects_wrong_preprocessing(self):
        self.write_protocol({"preprocessing": "RGB normalized externally"})

        with self.assertRaisesRegex(ValueError, "preprocessing"):
            predict_keras.E2Predictor(self.checkpoint, self.protocol_path)

    def test_rejects_checkpoint_hash_mismatch(self):
        self.write_protocol()

        with mock.patch.object(
            predict_keras.tf.keras.models, "load_model"
        ) as load_model:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                predict_keras.E2Predictor(
                    self.checkpoint, self.protocol_path
                )
        load_model.assert_not_called()

    def test_rejects_incorrect_model_input_shape(self):
        self.write_protocol()
        model = FakeModel([1 / 6] * 6)
        model.input_shape = (None, 299, 299, 3)

        with (
            mock.patch.object(predict_keras, "EXPECTED_SHA256", self.checkpoint_hash),
            mock.patch.object(
                predict_keras.tf.keras.models,
                "load_model",
                return_value=model,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "input shape"):
                predict_keras.E2Predictor(
                    self.checkpoint, self.protocol_path
                )

    def test_rejects_incorrect_model_output_shape(self):
        self.write_protocol()
        model = FakeModel([1 / 6] * 6)
        model.output_shape = (None, 5)

        with (
            mock.patch.object(predict_keras, "EXPECTED_SHA256", self.checkpoint_hash),
            mock.patch.object(
                predict_keras.tf.keras.models,
                "load_model",
                return_value=model,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "output shape"):
                predict_keras.E2Predictor(
                    self.checkpoint, self.protocol_path
                )

    def test_rejects_wrong_score_vector_shape(self):
        predictor, _ = self.create_predictor(scores=[0.2] * 5)
        with self.assertRaisesRegex(ValueError, "score shape"):
            predictor.predict(self.image_path)

    def test_rejects_non_finite_scores(self):
        predictor, _ = self.create_predictor(scores=[np.nan, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "Invalid model scores"):
            predictor.predict(self.image_path)

    def test_rejects_negative_scores(self):
        predictor, _ = self.create_predictor(scores=[1.1, -0.1, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "Invalid model scores"):
            predictor.predict(self.image_path)

    def test_rejects_scores_that_do_not_sum_to_one(self):
        predictor, _ = self.create_predictor(scores=[0.2] * 6)
        with self.assertRaisesRegex(ValueError, "Invalid model scores"):
            predictor.predict(self.image_path)

    def test_missing_image_is_reported(self):
        predictor, _ = self.create_predictor()
        with self.assertRaises(FileNotFoundError):
            predictor.predict(self.root / "missing.png")


if __name__ == "__main__":
    unittest.main()
