"""The two explicitly versioned E2 decoder paths, shared by CLI and evaluator."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


def decode_pixels(image_bytes, decoder="tensorflow"):
    import tensorflow as tf

    if decoder == "tensorflow":
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
    elif decoder == "pil":
        with Image.open(BytesIO(image_bytes)) as opened:
            image = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    else:
        raise ValueError("Decoder must be tensorflow or pil")
    return tf.image.resize(
        tf.cast(image, tf.float32), [224, 224],
        method="bilinear", antialias=False, preserve_aspect_ratio=False,
    ).numpy().astype(np.float32)
