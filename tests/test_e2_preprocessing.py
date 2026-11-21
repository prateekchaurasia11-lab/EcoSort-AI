import io
import unittest

import numpy as np
from PIL import Image
import tensorflow as tf

from Src.e2_preprocessing import decode_pixels
from Src import evaluate_keras, predict_keras


class DecoderTests(unittest.TestCase):
    def setUp(self):
        buffer = io.BytesIO()
        values = np.arange(37 * 53 * 3, dtype=np.uint8).reshape(37, 53, 3)
        Image.fromarray(values).save(buffer, format="JPEG", quality=85)
        self.raw = buffer.getvalue()

    def test_tensorflow_decoder_matches_explicit_reference(self):
        image = tf.io.decode_image(self.raw, channels=3, expand_animations=False)
        expected = tf.image.resize(tf.cast(image, tf.float32), [224, 224],
                                   method="bilinear", antialias=False).numpy()
        np.testing.assert_array_equal(decode_pixels(self.raw), expected)

    def test_pil_decoder_preserves_previous_path(self):
        with Image.open(io.BytesIO(self.raw)) as opened:
            image = np.asarray(opened.convert("RGB"))
        expected = tf.image.resize(image, [224, 224], method="bilinear", antialias=False).numpy()
        np.testing.assert_array_equal(decode_pixels(self.raw, "pil"), expected)

    def test_both_entry_points_share_decoder(self):
        self.assertIs(evaluate_keras.decode_pixels, predict_keras.decode_pixels)

    def test_unknown_decoder_rejected(self):
        with self.assertRaisesRegex(ValueError, "Decoder"):
            decode_pixels(self.raw, "unknown")
