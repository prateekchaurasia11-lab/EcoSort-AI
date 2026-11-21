# EcoSort-AI

## E2 MobileNetV2 Inference

E2 is maintained separately from the existing PyTorch ResNet18 predictor.

Install inference dependencies:

```bash
python -m pip install -r requirements-e2-inference.txt
```

Run a single-image prediction:

```bash
python Src/predict_keras.py image.jpg --checkpoint path/to/best.keras --protocol path/to/protocol.json
```

The checkpoint and protocol must be stored outside Git. The adapter verifies the pinned checkpoint SHA-256, class order, architecture, input size and preprocessing protocol. Input is RGB at 224x224 with values in 0..255; rescaling is embedded in the model. Returned scores are uncalibrated.

Both prediction and evaluation use the same TensorFlow decoder by default, followed by float32 TensorFlow bilinear resize to 224x224, antialias=False and no aspect-ratio preservation. Use --decoder pil in either CLI for the previous PIL/EXIF path. JSON and NPZ outputs identify the decoder/preprocessing version. Keep decoder-specific results separate.

The controlled TensorFlow decoder diagnostic matched the stored historical prediction argmax on all 382 test images (82.98% accuracy, macro F1 0.8013; maximum probability difference 4.83e-6). This reproduces the historical outputs but does not prove the missing original historical test-generation source or checkpoint-to-prediction binding.

## E2 Regression Tests

Run the suite:

```bash
python -m unittest discover -s tests -v
```

The tests cover checkpoint/protocol validation, decoder consistency, output-score checks, upload validation/cleanup, manifest audit and metric calculations. The checkpoint-backed Streamlit AppTest needs the three ECOSORT environment variables described in DEMO_QUICKSTART.md.

## E2 Configurable Evaluation

The evaluation command audits the clean CSV against the original split ZIP before inference. It checks manifest schema, allowed labels/splits, source-path-to-label consistency, missing/corrupt images, exact byte duplicates across splits, conflicting labels, and group-ID coverage/leakage when IDs exist.

Example validation run:

```bash
python Src/evaluate_keras.py --checkpoint path/to/best.keras --protocol path/to/protocol.json --manifest-csv path/to/EcoSort_TrashNet_Clean_Split.csv --dataset-zip path/to/EcoSort_TrashNet_split.zip --split val --batch-size 32 --device cpu --decoder tensorflow --output-json Results/e2_validation_metrics.json --output-npz /path/outside/repository/e2_validation_predictions.npz
```

Select train, val or test explicitly. The test split is diagnostic only and must not be used to tune model settings or thresholds. The public clean TrashNet manifest has no group IDs; the evaluator reports incomplete group coverage rather than claiming a group-level audit passed.

Decoder-specific evaluation results:
- PIL diagnostic: 377 validation images, 87.2679% accuracy, macro F1 0.85899 (the earlier run; keep separately labelled).
- TensorFlow decoder: 327/377 validation correct, 86.7374% accuracy, macro F1 0.85407.
- TensorFlow decoder test diagnostic: 317/382 correct, 82.9843% accuracy, macro F1 0.80128; 382/382 argmax agreement with stored historical predictions.

The checkpoint, protocol, dataset ZIP, model weights and generated prediction files are not committed.

## Runnable Streamlit E2 Demo

Install demo dependencies and start the app:

```bash
python -m pip install -r requirements-demo.txt
python -m streamlit run app.py
```

See DEMO_QUICKSTART.md for model setup, upload limits, test commands and demo limitations. The app accepts bounded JPEG/PNG uploads, displays the predicted class and all six uncalibrated scores, provides general disposal guidance, and offers a downloadable prediction JSON. The E2 checkpoint is hash-verified by the adapter. The model and protocol remain in restricted storage and are ignored by Git.

## Other project work

The E6 detector tooling is present separately on main; it is not integrated into this single-image E2 Streamlit page. E1 checkpoint recovery, expanded physical-item/session group review, E3/E4 and E5 calibration/rejection work remain separate items.
