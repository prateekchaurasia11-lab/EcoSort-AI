# EcoSort-AI working demo

Use Python 3.11 or 3.12 in a virtual environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-demo.txt
python -m streamlit run app.py
```

Download the verified historical E2 bundle:
https://drive.google.com/file/d/1uF85me4kCOrrO-h-Z9erj7ZHWE9QCJFa/view

Place its `best.keras` and `protocol.json` in `Models/e2/` (create that folder).
The adapter checks the model SHA-256 before loading:
466d5514de963ccbb3606ff3cbe39c77c42fde2442ab3f108d555722e408a27e

Alternatively set ECOSORT_CHECKPOINT and ECOSORT_PROTOCOL to their absolute paths.
Do not substitute a ResNet or E1 checkpoint. Do not scale images yourself.
Open the localhost URL printed by Streamlit. Upload one JPEG/PNG and click Classify.
The app previews it, shows all six scores, general disposal guidance and downloadable JSON.
Temporary inference files are deleted after prediction, including failures. Streamlit holds
uploads in memory for the active session; do not treat this prototype as a retention guarantee.
No image is intentionally written to a project dataset or sent to an external inference API.

## Verification

```sh
python -m unittest discover -s tests -v
```

Checkpoint-backed app smoke tests additionally use ECOSORT_CHECKPOINT/ECOSORT_PROTOCOL.
Historical test317/382 (82.9843%), macroF1 .8012786799, TF decoding. Validation327/377
(86.7374%), macroF1 .8540741020. PIL mode has separate results; do not mix them.
No calibrated confidence, rejection threshold, detection, or completed independent-local
cohort evaluation is claimed. Group/session IDs unavailable in the public baseline.
E1 weights still missing. E3–E6 are pending research work. Preserve existing split assignments.

The self-contained demo ZIP already includes the E2 assets in Models/e2, so no separate
model download is needed for that ZIP. `sample_glass.jpg` is an existing public test image
for a reproducibility smoke check, not an independent evaluation sample.

To include the checkpoint-backed Streamlit test (PowerShell):

```powershell
$env:ECOSORT_CHECKPOINT = (Resolve-Path Models/e2/best.keras).Path
$env:ECOSORT_PROTOCOL = (Resolve-Path Models/e2/protocol.json).Path
$env:ECOSORT_TEST_IMAGE = (Resolve-Path sample_glass.jpg).Path
python -m unittest discover -s tests -v
```

The optional test supplies an upload through a mock file-upload widget and clicks the
actual Streamlit classify button using AppTest. It runs the real model and checks the
result. A full graphical browser upload/layout check still needs a local browser.
