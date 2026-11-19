# EcoSort-AI

## E2 MobileNetV2 Inference

E2 is maintained separately from the existing PyTorch ResNet18 predictor.

Install dependencies:

```bash
python -m pip install -r requirements-e2-inference.txt
```

Run a single-image prediction:

```bash
python Src/predict_keras.py image.jpg --checkpoint path/to/best.keras --protocol path/to/protocol.json
```

The checkpoint and protocol must be stored outside Git. The adapter verifies the pinned checkpoint SHA-256, class order, architecture, input size and preprocessing protocol. Input is RGB at 224x224 with values in 0..255; rescaling is embedded in the model. Returned scores are uncalibrated.

## E2 Regression Tests

Run the adapter and evaluation-audit tests:

```bash
python -m unittest discover -s tests -v
```

The adapter tests cover protocol and checkpoint rejection, model input/output shape validation, score shape and probability validation, preprocessing output shape/dtype, and missing images. They use a mocked model and do not require model weights.

## E2 Configurable Evaluation

The evaluation command audits the clean CSV against the original split ZIP before inference. It checks manifest schema, allowed labels/splits, source-path-to-label consistency, missing/corrupt images, exact byte duplicates across splits, conflicting labels, and cross-split `group_id` values when that column exists.

Example validation-split run:

```bash
python Src/evaluate_keras.py \
  --checkpoint path/to/best.keras \
  --protocol path/to/protocol.json \
  --manifest-csv path/to/EcoSort_TrashNet_Clean_Split.csv \
  --dataset-zip path/to/EcoSort_TrashNet_split.zip \
  --split val \
  --batch-size 32 \
  --device cpu \
  --output-json Results/e2_validation_metrics.json \
  --output-npz /path/outside/repository/e2_validation_predictions.npz
```

Select `--split train`, `val` or `test` explicitly. The test split is diagnostic only and must not be used to tune model settings or thresholds. The provided clean TrashNet manifest has no group-ID column, so the evaluator reports group-level leakage checking as unavailable rather than claiming it passed. Supply a manifest with `group_id` or `source_group_id` to enable that check.

Reports include accuracy, macro precision/recall/F1, per-class precision/recall/F1/support, confusion matrix, sample counts, manifest/archive/checkpoint hashes, and separate model-load, preprocessing and inference timings. Prediction NPZ output should remain outside Git.

The checkpoint, protocol, dataset ZIP, and generated prediction files are not committed.
