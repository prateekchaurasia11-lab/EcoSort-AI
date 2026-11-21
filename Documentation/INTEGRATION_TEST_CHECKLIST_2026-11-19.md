# EcoSort-AI Integration Test and Demo Checklist
**Owner:** Rahul Singh (integration-testing)  
**Evidence date:** 22 November 2026  
**Related PR:** #5 — E2 inference, evaluator, decoder and Streamlit baseline demo

## 1. Source and integrity
- [x] Verify v2 patch ZIP SHA-256: `de9c1665c0dc0fa7d3a2c068684f50d89fc82b5f3db499ac18ebd5ff452f599d`.
- [x] Verify runnable demo ZIP SHA-256: `745137b80e373b62175904f6dfc76382e20e71fd98a580657e4fc8499b78b666`.
- [x] Verify both ZIP CRCs.
- [x] Verify E2 checkpoint SHA-256: `466d5514de963ccbb3606ff3cbe39c77c42fde2442ab3f108d555722e408a27e`.
- [x] Preserve the existing PyTorch `Src/predict.py` and the E6 detector files from latest `main`.
- [x] Keep model weights, protocols, data archives, sample image and generated NPZs out of Git.

## 2. E2 inference and preprocessing
- [x] Shared TensorFlow decoder used by both E2 prediction and evaluation.
- [x] Keep `--decoder tensorflow` as default and `--decoder pil` as the separately versioned comparison path.
- [x] Record decoder and preprocessing version in prediction/evaluation output.
- [x] Validate class order, architecture, input shape, output shape, checkpoint hash and score vector.
- [x] Do not apply external normalization; the model embeds its Rescaling layer.
- [x] Mark scores as uncalibrated.

## 3. Regression evidence
- [x] Original PR #5 suite: 21 tests passed in Colab with TensorFlow 2.21.0 / Keras 3.15.1.
- [x] Prateek's v2 patch verification log: 27 tests passed.
- [x] Supplied demo verification log: 33 tests passed, 0 failures.
- [x] Demo AppTest uses a mocked uploader but executes the real E2 checkpoint, classification and result-rendering path.
- [x] Independent local syntax compilation passed for `Src/`, `app.py` and `tests/`.
- [x] Independent local upload-service unit tests: 5/5 passed.
- [ ] Full 33-test suite rerun in Rahul's own TensorFlow/Streamlit runtime.
- [ ] Graphical browser upload and mobile-width layout smoke test.

**Verification limitation:** the 27/33 results above are recorded in the supplied Prateek bundle logs. This container has no TensorFlow/Keras/Streamlit packages and cannot reach PyPI (DNS/network resolution failed), so the full suite could not be rerun here. The local independent checks are limited to syntax compilation and five upload-service tests.

## 4. Dataset audit and evaluation
- [x] Clean manifest: 2,521 rows — 1,762 train / 377 validation / 382 test.
- [x] Zero missing images, decode errors, exact cross-split byte duplicates or conflicting-label duplicate hashes.
- [x] Group coverage is explicitly incomplete/unavailable: the public baseline manifest has no physical item/session IDs.
- [x] Preserve PIL and TensorFlow decoder outputs as separate results.
- [x] TensorFlow decoder validation: 327/377 correct, 86.7374% accuracy, macro F1 0.8540741.
- [x] TensorFlow decoder test diagnostic: 317/382 correct, 82.9843% accuracy, macro F1 0.8012787; 382/382 historical argmax agreement, max score difference 4.827976e-6.
- [x] Earlier PIL validation result remains separate: 87.2679% accuracy, macro F1 0.8589876.
- [ ] Obtain reviewed physical item/session group IDs for expanded/local challenge data.
- [ ] Do not tune against the reused public test cohort.

## 5. Streamlit E2 baseline
- [x] Add `app.py` and `Src/demo_service.py`; app calls the E2 predictor directly.
- [x] Bounded JPEG/PNG upload; max 10 MB and 20 megapixels; still images only.
- [x] Display predicted class, raw uncalibrated model score, six scores, general local-rule-dependent disposal guidance and downloadable JSON.
- [x] Show checkpoint identity, decoder and preprocessing version.
- [x] Clean temporary inference files after success and failure.
- [x] Supplied real-checkpoint AppTest covers upload/classify/result path with mocked upload widget.
- [ ] Manual graphical browser upload and mobile layout check.

## 6. Other pending work
- [x] E6 detector tooling exists separately on latest `main`; it is not integrated into this E2 Streamlit page.
- [ ] Recover the separate E1 `best.keras` checkpoint if needed for the approved E1/E2 demo.
- [ ] Obtain reviewed physical item/session IDs for expanded/local challenge cohort before split freezing.
- [ ] E3/E4 experiments and E5 calibration/rejection work remain pending.
- [ ] Final end-to-end regression, demo recording and report/presentation.

## Interpretation
PIL and TensorFlow decoder metrics are separate and must not be mixed. The public 382-image test set is reused for historical/diagnostic reproduction, not an independent local cohort. App scores are uncalibrated. Exact-byte hash checks do not establish physical-session independence. Passing AppTest is not a substitute for a manual graphical browser/mobile check.
