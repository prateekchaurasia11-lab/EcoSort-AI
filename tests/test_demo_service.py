import io
import unittest
from pathlib import Path
from PIL import Image
from Src.demo_service import validate_upload, predict_upload, MAX_BYTES

class UploadTests(unittest.TestCase):
    def setUp(self):
        b = io.BytesIO(); Image.new('RGB', (24, 30), 'green').save(b, format='PNG'); self.raw = b.getvalue()
    def test_valid_image(self):
        validate_upload(self.raw)
    def test_empty_corrupt_and_oversized(self):
        for raw in [b'', b'not an image', b'x' * (MAX_BYTES + 1)]:
            with self.subTest(size=len(raw)), self.assertRaises(ValueError): validate_upload(raw)
    def test_format_rejected(self):
        b=io.BytesIO();Image.new('RGB',(2,2)).save(b,format='GIF')
        with self.assertRaises(ValueError): validate_upload(b.getvalue())
    def test_cleanup_even_on_prediction_failure(self):
        paths=[]
        class Fail:
            def predict(self,path):
                paths.append(path); assert path.is_file(); raise RuntimeError('model failed')
        with self.assertRaises(RuntimeError): predict_upload(Fail(),self.raw)
        self.assertFalse(paths[0].exists())
    def test_success_cleanup(self):
        paths=[]
        class Predictor:
            def predict(self,path): paths.append(path); return {'class':'glass'}
        self.assertEqual(predict_upload(Predictor(),self.raw), {'class':'glass'})
        self.assertFalse(paths[0].exists())
