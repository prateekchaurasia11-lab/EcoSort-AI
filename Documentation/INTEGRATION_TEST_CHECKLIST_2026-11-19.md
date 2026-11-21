# EcoSort-AI Integration Test and Demo Checklist
**Owner:** Rahul Singh (integration-testing)  
**Evidence date:** 21 November 2026  
**Related PR:** #5 — E2 inference, decoder, evaluator and Streamlit baseline demo

## 1. Reproducible E2 environment
- [x] Verify E2 checkpoint SHA-256: 466d5514de963ccbb3606ff3cbe39c77c42fde2442ab3f108d555722e408a27e
- [x] Verify protocol architecture, six-class order, 224x224 input and embedded rescaling.
- [x] Pin TensorFlow 2.21.0 and Keras 3.15.1 in the E2 inference requirements.
- [x] Verify TensorFlow 2.21.0 / Keras 3.15.1 in the restarted Colab runtime.
- [x] Keep model weights, protocols, datasets and generated prediction NPZ files outside Git.

## 2. Regression tests
- [x] v2 decoder/evaluator patch test log: 27 tests, all passed.
- [x] Supplied runnable Streamlit demo log: 33 tests, 0 failures.
- [x] Test wrong checkpoint hash, architecture, class order, input size and preprocessing.
- [x] Test invalid/non-finite/negative/unnormalized score outputs.
- [x] Test both decoder paths and shared decoder use by adapter/evaluator.
- [x] Test blank/partial group-ID coverage is reported as incomplete.
- [x] Test upload size/type/pixel constraints and temporary-file cleanup.
- [x] Keep existing PyTorch Src/predict.py unchanged.
- [ ] Re-run the full 33-test suite in Rahul's own clean local/Colab environment after applying this commit.

## 3. Dataset audit and evaluation
- [x] Verify clean manifest membership against the original split ZIP.
- [x] Verify 2,521 manifest rows: 1,762 train / 377 validation / 382 test.
- [x] Verify zero missing images, decode errors, exact cross-split byte duplicates and conflicting-label duplicate hashes.
- [x] Record group-ID coverage explicitly: the public manifest has 2,521 rows without group IDs; physical item/session leakage is not established.
- [x] Keep PIL and TensorFlow decoder results as separate reports.
- [x] TensorFlow decoder validation: 327/377 correct (86.7374%), macro F1 0.8540741.
- [x] TensorFlow decoder test diagnostic: 317/382 correct (82.9843%), macro F1 0.8012787; 382/382 stored historical argmax agreement.
- [x] Preserve previous PIL validation result separately: 87.2679%, macro F1 0.8589876.
- [ ] Obtain reviewed physical item/session group IDs for any expanded/local challenge cohort before split freezing.
- [ ] Do not tune model settings or thresholds against the reused public test split.

## 4. E2 historical provenance
- [x] Preserve historical predictions and fresh PIL reproduction separately.
- [x] Export recovered Keras 3.15.1 inference cell and helper cells 1–4 verbatim.
- [x] Run and preserve TensorFlow decoder NPZ, report, three-image pixel differences and input tensor hashes.
- [ ] Await review of the decoder evidence. Prediction reproduction does not prove the original historical generation source or checkpoint-to-prediction binding.

## 5. Streamlit E2 application
- [x] Add app.py and Src/demo_service.py. The app calls E2Predictor directly; a separate HTTP backend is not required for this baseline.
- [x] Restrict uploads to JPEG/PNG, 10 MB and 20 megapixels; reject animated/corrupt files.
- [x] Display predicted class, six raw uncalibrated scores, general rule-dependent disposal guidance and downloadable JSON.
- [x] Show checkpoint SHA-256, decoder and preprocessing version.
- [x] Clean temporary inference files on success and failure.
- [x] Supplied AppTest log exercises upload, Classify and result rendering with a real checkpoint; upload widget is mocked.
- [ ] Run a graphical desktop browser upload and mobile-width layout check.
- [ ] Confirm the app's local model paths and dependencies on the target demo machine.

## 6. Other project items
- [x] E6 detector tooling exists separately on latest main via PR #6; this E2 Streamlit page does not yet integrate E6 detection.
- [ ] Recover the separate E1 best.keras checkpoint if needed for the approved E1/E2 demo.
- [ ] Obtain reviewed physical item/session group IDs for expanded/local challenge data.
- [ ] E3/E4 research and E5 calibration/rejection remain separate pending work.
- [ ] Final integrated regression, demo recording and report/presentation.

## Interpretation and limits
The TensorFlow-decoder validation result and earlier PIL-decoder validation result are distinct. The public 382-image test set is reused for historical/diagnostic reproduction and is not an independent local cohort. Raw model scores are uncalibrated. The Streamlit demo is a runnable baseline, not a claim that all research experiments are complete. Group/session independence is not established without reviewed IDs. The provided AppTest uses a mocked upload widget; a graphical browser check remains pending.
