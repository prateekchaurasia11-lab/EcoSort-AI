# E6 Multi-object Detector: Workflow and Status

Branch: `feature/e6-detector` (based on `main` at `10ef90b`). Adds a separate detection pipeline. It does not change `Src/predict.py`, the E2 classifier, `app.py` or any dataset/checkpoint.

## What exists (verified by running it)
| Piece | File | Verified how |
|---|---|---|
| Dataset checker (class order, pairing, coordinates, empty splits, duplicates, optional group leakage) | `Src/detector/check_dataset.py` | Unit tests + run on a synthetic dataset |
| Training wrapper (refuses to train on a failing dataset; writes `run_manifest.json` with checkpoint SHA-256) | `Src/detector/train_yolo.py` | Real 8-epoch CPU run with Ultralytics 8.4.216 on synthetic data |
| Protocol writer | `Src/detector/make_protocol.py` | Unit tests + real run |
| Hash-verified detector adapter `E6Detector` | `Src/detector/e6_detector.py` | Unit tests (stub model) + a real Ultralytics checkpoint |
| Evaluator (precision/recall at the app threshold, per class, small-object recall, FP/FN lists, latency, mAP@0.5 and mAP@0.5:0.95) | `Src/detector/evaluate_detector.py` | Unit tests + real run |
| Streamlit detection page `app_detect.py` | `app_detect.py` | Driven in a headless browser: missing-model state, upload, detect, counts, per-object table, guidance, downloads, stale-result clearing, corrupt file message, phone width |
| Colab notebook | `NoteBook/EcoShot_E6_Detector_Training.ipynb` | Valid notebook, **not executed on Colab** |

## What does NOT exist yet
- **A trained waste detector.** All runs above used synthetic coloured rectangles purely to prove the code path. Their metrics (0 mAP) mean nothing and must never be quoted.
- A frozen, verified detection dataset (Prince's regenerated TACO export and group IDs).
- Real-waste metrics of any kind, calibration, unknown-object rejection.
- The "detector + E2 crop classifier" comparison (Approach 2). Only the direct six-class detector (Approach 1) is built.
- Mixed-scene photos for an independent test set.

## Run order (Windows PowerShell, repo root)
```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-detector.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
Training is done on Colab (T4): open `NoteBook/EcoShot_E6_Detector_Training.ipynb`.

After training, copy `best.pt` and `protocol.json` to `Models\e6\` (both are git-ignored), then:
```
.\.venv\Scripts\python.exe -m streamlit run app_detect.py
```
To use another folder set `ECOSHOT_E6_DIR`. The page refuses to show any box unless the checkpoint hash matches `protocol.json`.

## Rules these tools enforce
- Class order is fixed: cardboard, glass, metal, paper, plastic, trash.
- Pick thresholds and checkpoints on **val**. `evaluate_detector --split test` refuses `--conf` overrides. Run test once.
- Exact-duplicate leakage is always checked. Scene/session leakage is checked only if you pass a group manifest, and the checker warns when you do not.
- Scores are raw detector scores: `calibrated=false` in every JSON output.
- Boxes with non-finite values, bad class ids or zero area are dropped and counted, never drawn.

## Metric definitions (so the report is precise)
- `ultralytics_val`: mAP@0.5, mAP@0.5:0.95 and per-class AP from Ultralytics at conf 0.001.
- `operating_point`: precision/recall at the protocol threshold with class-aware IoU >= 0.5 matching. This is what a user of the app sees.
- Small object: sqrt(box area) < 32 px after scaling the longest image side to 640.
- Latency: `model.predict()` only, excludes file decoding, first 3 images excluded; depends on the machine (the report records the device).

## Known limits
- AGPL-3.0: Ultralytics YOLO is AGPL-3.0 licensed. Fine for coursework; state it in the report.
- A detector trained on TACO may not transfer to tidy tabletop photos. Only an independent mixed-scene test set can show that.
- Images that contain any deferred object are held out of Prince's export under the current policy, so the usable image count must be re-measured.
