"""Bounded upload validation and ephemeral inference for the demo."""
import io
import tempfile
import warnings
from pathlib import Path
from PIL import Image, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
GUIDANCE = {
    'cardboard': 'Keep clean and dry. Flatten boxes and check your local paper/cardboard collection rules.',
    'glass': 'Empty containers and check local glass collection rules. Keep broken glass and non-container glass separate.',
    'metal': 'Empty containers. Check local metal collection rules; batteries and pressurised containers need separate handling.',
    'paper': 'Keep clean and dry. Food-soiled, laminated and coated paper may need a different collection stream.',
    'plastic': 'Check the packaging material and local acceptance rules. Flexible wrappers and rigid containers may need different streams.',
    'trash': 'Inspect the item manually and check local residual-waste rules. Keep batteries, electronics and hazardous waste separate.',
}


def validate_upload(raw):
    if not raw:
        raise ValueError('Choose a non-empty image.')
    if len(raw) > MAX_BYTES:
        raise ValueError('Please use an image smaller than 10 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {'JPEG', 'PNG'}:
                    raise ValueError('Use a JPEG or PNG image.')
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('Please use an image below 20 megapixels.')
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Use a still image rather than an animation.')
                image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('This image could not be read. Try another JPEG or PNG.') from exc


def predict_upload(predictor, raw):
    validate_upload(raw)
    # No user filename enters the filesystem. All temporary bytes are removed on failure too.
    with tempfile.TemporaryDirectory(prefix='ecosort-') as directory:
        path = Path(directory) / 'upload.image'
        path.write_bytes(raw)
        result = predictor.predict(path)
    return result
