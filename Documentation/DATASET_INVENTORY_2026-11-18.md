# EcoSort-AI Dataset Inventory & 6-Class Manifest Audit (18 Nov 2026)

**Author:** Prince Verma (230108048 — Dataset Lead)  
**Branch:** `dataset-preparation`  
**Verified Source Checkpoint:** `EcoSort_Reviewed_Progress_2026-11-16.zip` (Drive handoff checkpoint)

---

## 1. Verified Dataset Inventory Summary

All SHA256 hashes, prior review decisions, and group IDs from the 16 Nov checkpoint (`review_decisions.csv`) plus the 20 local wrapper captures have been reconciled via `Src/build_manifest.py` and verified with `Src/dataset_audit.py` (`4,925` total unique SHA256 rows; `0` duplicate hashes; `0` split leaks).

| Dataset / Pool | Total Items | Current Split / Admission Status | Notes |
|---|---:|---|---|
| **Clean TrashNet Baseline** | 2,521 | `1,762` train / `377` val / `382` test | 6 conflicting copies excluded; reused public test set (do not tune thresholds on test). |
| **Six-Class Kaggle Originals (Full Pool)** | 7,463 | Candidate pool (`0` decode errors) | Full source archive recovered on 16 Nov; `4,905` tracked in `review_decisions.csv`. |
| **Checkpoint Reviewed Candidates (`review_decisions.csv`)** | 4,905 | `unassigned` (`training_admission: false`) | `548` `approve`, `472` `defer_ambiguous`, `46` `defer_wrapper`, `3,839` `pending`. |
| **Local Wrapper Captures (`additional_training/plastic`)** | 20 | `unassigned` (`defer_wrapper`) | Preserved with SHA256 hashes; paused per wrapper-only halt and composite/film policy. |
| **Combined Audited Manifest (`Data/ecosort_manifest_2026-11-20.csv`)** | **4,925** | **`unassigned` (awaiting group freeze)** | Generated reproducibly by `py -3.11 -m Src.build_manifest`. |

---

## 2. Class-by-Class Breakdown (`4,925` Manifest Rows)

| Class (`CLASS_NAMES` order) | `approve` | `defer_ambiguous` | `defer_wrapper` | `pending` | Total Tracked | Key Bottleneck / Action Needed |
|---|---:|---:|---:|---:|---:|---|
| `0: cardboard` | 60 | 86 | 0 | 742 | **888** | Review `742` pending; confirm clean vs. food-soiled/coated carton boundary. |
| `1: glass` | 123 | 9 | 0 | 1,123 | **1,255** | Largest pending pool (`1,123`); verify item/session grouping before admission. |
| `2: metal` | 124 | 8 | 0 | 387 | **519** | Strongest `approve` count (`124`); review remaining `387` pending. |
| `3: paper` | 116 | 14 | 0 | 634 | **764** | Review `634` pending; separate clean paper from soiled tissue/food wrap. |
| `4: plastic` | 115 | 13 | 24 | 953 | **1,105** | Includes `4` checkpoint + `20` local `defer_wrapper` items; review `953` pending rigid/film items. |
| `5: trash` | 10 | 342 | 42 | 0 | **394** | **Critical bottleneck:** only `10` approved items; requires diverse independent residual-trash examples. |
| **Total** | **548** | **472** | **66** | **3,839** | **4,925** | **Proposal target:** >= 6,000 verified unique images across all 6 classes. |

---

## 3. Reproducible CLI Workflow

1. **Generate the 20 Nov 6-class manifest** (writes to gitignored `Data/ecosort_manifest_2026-11-20.csv`):
   `py -3.11 -m Src.build_manifest`
2. **Audit schema, SHA256 uniqueness, `approve`-only gating, and zero group leakage**:
   `py -3.11 -m Src.dataset_audit Data\ecosort_manifest_2026-11-20.csv`
3. **Run unit & regression tests**:
   `py -3.11 -m unittest discover -s tests -v`
