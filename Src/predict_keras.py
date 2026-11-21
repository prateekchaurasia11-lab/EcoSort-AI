import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
if __package__:
    from .e2_preprocessing import decode_pixels
else:
    from e2_preprocessing import decode_pixels

CLASSES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

EXPECTED_SHA256 = (
    "466d5514de963ccbb3606ff3cbe39c77c42fde2442ab3f108d555722e408a27e"
)


class E2Predictor:

    def __init__(self, checkpoint, protocol_path, decoder="tensorflow"):
        if decoder not in {"tensorflow", "pil"}:
            raise ValueError("Decoder must be tensorflow or pil")
        self.decoder = decoder
        self.checkpoint = Path(checkpoint)
        self.protocol = json.loads(
            Path(protocol_path).read_text(encoding="utf-8")
        )

        if self.protocol["classes"] != CLASSES:
            raise ValueError("Unexpected class order")

        if self.protocol["architecture"] != "MobileNetV2":
            raise ValueError("Unexpected architecture")

        if self.protocol["input_size"] != [224, 224]:
            raise ValueError("Unexpected input size")

        expected = (
            "RGB 0..255; embedded Rescaling(1/127.5, offset=-1)"
        )

        if self.protocol["preprocessing"] != expected:
            raise ValueError("Unexpected preprocessing")

        digest = hashlib.sha256(
            self.checkpoint.read_bytes()
        ).hexdigest()

        if digest != EXPECTED_SHA256:
            raise ValueError("Checkpoint SHA-256 mismatch")

        self.checkpoint_sha256 = digest

        self.model = tf.keras.models.load_model(
            self.checkpoint, compile=False
        )

        if tuple(self.model.input_shape[1:]) != (224, 224, 3):
            raise ValueError("Incorrect input shape")

        if self.model.output_shape[-1] != 6:
            raise ValueError("Incorrect output shape")

    def predict(self, image_path):
        pixels = decode_pixels(Path(image_path).read_bytes(), self.decoder)

        scores = self.model.predict(
            pixels.astype(np.float32)[None],
            verbose=0
        )[0]

        if scores.shape != (len(CLASSES),):
            raise ValueError("Invalid model score shape")

        if (
            not np.isfinite(scores).all()
            or (scores < 0).any()
            or not np.isclose(scores.sum(), 1, atol=1e-4)
        ):
            raise ValueError("Invalid model scores")

        index = int(scores.argmax())

        return {
            "class": CLASSES[index],
            "model_score": float(scores[index]),
            "scores": dict(zip(CLASSES, map(float, scores))),
            "calibrated": False,
            "checkpoint_sha256": self.checkpoint_sha256,
            "decoder": self.decoder,
            "preprocessing_version": "e2-tf-bilinear-224-" + self.decoder,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--decoder", choices=["tensorflow", "pil"], default="tensorflow")

    args = parser.parse_args()

    predictor = E2Predictor(
        args.checkpoint, args.protocol, args.decoder
    )

    print(json.dumps(
        predictor.predict(args.image), indent=2
    ))


if __name__ == "__main__":
    main()
