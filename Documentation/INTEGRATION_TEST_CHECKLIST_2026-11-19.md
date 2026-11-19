# EcoSort-AI Integration Test and Demo Checklist
**Owner:** Rahul Singh (integration-testing)  
**Evidence date:** 19 November 2026  
**Related PR:** #5 — Add E2 inference, regression tests and evaluation CLI

## 1. Reproducible E2 environment
- [x] Verify E2 checkpoint SHA-256: `466d5514de963ccbb3606ff3cbe39c77c42fde2442ab3f108d555722e408a27e`
- [x] Verify protocol architecture, six-class order, 224x224 input and embedded rescaling.
- [x] Pin TensorFlow 2.21.0 and Keras 3.15.1 in the E2 inference requirements.
- [x] Restart Colab after dependency installation and verify TensorFlow 2.21.0 / Keras 3.15.1.
- [x] Keep checkpoints, original archives and generated prediction NPZ files outside Git.

## 2. Regression tests
- [x] Run `python -m unittest discover -s tests -v` in Colab.
- [x] Confirm 21 tests pass under TensorFlow 2.21.0 / Keras 3.15.1.
- [x] Check invalid checkpoint hash, architecture, class order, input size and preprocessing are rejected.
- [x] Check invalid/non-finite/negative/unnormalized model scores are rejected.
- [x] Check manifest missing files, invalid label/path relation, duplicate content and optional group leakage.
- [x] Confirm existing PyTorch `Src/predict.py` remains unchanged.

## 3. Dataset audit and validation
- [x] Verify clean manifest membership against the original split ZIP.
- [x] Verify 2,521 manifest rows: 1,762 train / 377 validation / 382 test.
- [x] Verify zero missing images, decode errors, exact cross-split byte duplicates and conflicting-label duplicate hashes.
- [ ] Obtain a manifest with item/session `group_id` values and verify group leakage before claiming session independence.
- [x] Run configurable evaluator on validation split only.
- [x] Preserve separate Keras 3.13.2 and 3.15.1 validation JSON/NPZ outputs.
- [x] Verify cross-version validation prediction equality: 377/377 argmax agreement and zero probability difference.
- [ ] Do not tune on or promote settings based on the reused public test split.

## 4. Historical E2 provenance
- [x] Preserve historical predictions and fresh PIL reproduction separately.
- [x] Export the original recovered Keras 3.15.1 inference cell and helper cells 1–4.
- [x] Run TensorFlow decoder diagnostic with the supplied checkpoint and fixed 382-image membership.
- [x] Preserve TensorFlow decoder NPZ, pixel differences, tensor hashes and report in restricted Drive.
- [ ] Await reviewer assessment of the decoder diagnostic and original pipeline provenance.
- [ ] Do not claim the original historical test-generation pipeline is proven: the historical source and checkpoint-to-prediction binding were not originally recorded.

## 5. CLI and UI integration
- [x] E2 single-image CLI smoke test with supplied checkpoint.
- [ ] Run the full regression suite again after any future code changes.
- [ ] Obtain the actual frontend/backend UI source from the app-interface owner.
- [ ] Test UI upload, model selection, API/CLI response mapping, class ordering and error handling.
- [ ] Verify UI displays raw model scores as uncalibrated; do not present them as confidence probabilities without calibration.
- [ ] Verify UI and CLI use the same checkpoint and preprocessing protocol.

**Current blocker:** the repository's `app-interface` branch currently contains no frontend/backend application source, so UI integration tests cannot yet be performed.

## 6. Demo readiness
- [x] Keep E2 MobileNetV2 separate from the original PyTorch ResNet18 predictor.
- [x] Document that E2 scores are uncalibrated.
- [ ] Recover the separate E1 `best.keras` checkpoint if required for the approved E1/E2 demonstration.
- [ ] Finish UI integration and end-to-end demo regression.
- [ ] Confirm reviewer feedback and resolve requested changes before marking PR #5 ready.
- [ ] Keep PR #5 open and draft until integration/review gates are satisfied.

## Evidence interpretation
The validation score (87.27% accuracy, macro F1 0.85899 on 377 images) is a validation result for the supplied E2 checkpoint, not an independent deployment estimate. CPU timing is specific to the measured Colab session. Exact SHA-256 duplicate checks do not establish perceptual or capture-session independence. The 382-image public test set has been reused for historical and diagnostic comparisons and remains diagnostic only.
