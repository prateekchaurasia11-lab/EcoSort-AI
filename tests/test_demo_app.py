"""Optional checkpoint-backed app test: set model, protocol and sample paths."""
import os
import unittest
from pathlib import Path

@unittest.skipUnless(all(os.environ.get(k) for k in ['ECOSORT_CHECKPOINT','ECOSORT_PROTOCOL','ECOSORT_TEST_IMAGE']), 'Set the three ECOSORT model/protocol/test image paths')
class RealAppTests(unittest.TestCase):
    def test_upload_classify_and_result(self):
        from streamlit.testing.v1 import AppTest
        script = '''
import runpy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
raw = Path(SAMPLE).read_bytes()
with patch('streamlit.file_uploader', return_value=SimpleNamespace(getvalue=lambda: raw)):
    runpy.run_path(APP, run_name='__main__')
'''.replace('SAMPLE',repr(os.environ['ECOSORT_TEST_IMAGE'])).replace('APP',repr(str(Path(__file__).resolve().parents[1]/'app.py')))
        app=AppTest.from_string(script).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertFalse(app.button[0].disabled)
        app.button[0].click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value,'Glass')
        self.assertEqual(app.session_state['prediction']['decoder'],'tensorflow')
