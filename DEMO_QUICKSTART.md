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

Place its best.keras and protocol.json in Models/e2/ (create that folder).
The adapter checks the model SHA-256 before loading:
466d5514de963ccbb3606ff3cbe39c77c42fde2442ab3f108d555722e408a27e

Alternatively set ECOSORT_CHECKPOINT and ECOSORT_PROTOCOL to their absolute paths.
Do not substitute a ResNet or E1 checkpoint. Do not scale images yourself.
Open the localhost URL printed by Streamlit. Upload one JPEG/PNG and click Classify.
The app previews it, shows all six uncalibrated scores, general disposal guidance and downloadable JSON.
Uploads are limited to JPEG/PNG, 10 MB and 20 megapixels. Temporary inference files are
deleted after prediction, including failures. Streamlit holds uploads in memory for the
active session; do not treat this prototype as a retention guarantee. No image is
intentionally written to a project dataset or sent to an external inference API.

## Verification

Run:
```sh
python -m unittest discover -s tests -v
```

Prateek's supplied bundle verification log records 33 tests passing, including a
real-checkpoint Streamlit AppTest. The upload widget is mocked in that AppTest; the
actual model, classify and result rendering code run. A graphical browser upload and
mobile-layout check remains pending.

TensorFlow decoder results: test diagnostic 317/382 (82.9843%), macro F1 0.8012787;
validation 327/377 (86.7374%), macro F1 0.8540741. PIL-decoder metrics are separate;
do not mix decoder results. No calibrated confidence or rejection threshold is claimed.
Group/session IDs are unavailable in the public baseline, so physical-session
independence has not been established. E1 weights remain missing. E3/E4/E5 work remains
pending; E6 detector tooling exists separately on main and is not integrated into this page.
Preserve existing split assignments.

This self-contained demo ZIP includes the E2 assets in Models/e2, so no separate model
download is needed when running that ZIP. The checkpoint and protocol are not committed
to Git. sample_glass.jpg is an existing public test image for a reproducibility smoke check,
not an independent evaluation sample.

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
