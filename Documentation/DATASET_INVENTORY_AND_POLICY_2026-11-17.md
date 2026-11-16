# EcoSort-AI — Dataset Inventory, Audit & Class-Policy Questions (17 November 2026)

**Owner:** Prince (`dataset-preparation` branch)  
**Target Demo:** 5 December 2026 (Official Deadline: 15 December 2026)

## 1. Local & Repository Asset Inventory (Verified 17 Nov 2026)

| Asset / Path | Location | Status | Verified Count / Details |
|---|---|---|---|
| `EcoSort-AI` Repo (`dataset-preparation`) | `C:\Users\asus\EcoSort-AI` | Synced with `origin/main` (`10ef90b`) | Contains `Documentation/CROSS_ACCOUNT_HANDOFF_2026-11-02.md`, `NoteBook/`, `Results/`, `Src/predict.py` |
| `additional_training/` + `manifest.csv` | `C:\Users\asus\github\React\EcoSort_AI\additional_training` | Local only (excluded from Git) | **20 images** (`wrapper_001.jpg`–`wrapper_020.jpg`) in `plastic/` subfolder; noted as `metallized_plastic_snack_wrapper` (`unknown_brand`). **Status: Deferred** pending class-policy confirmation (wrapper-only work paused; no silent relabeling). |
| `EcoSort_Wrappers_Training_20.zip` | `C:\Users\asus\github\React\EcoSort_AI\EcoSort_Wrappers_Training_20.zip` | Local archive (`406,823` bytes) | Archive of the 20 wrapper photos above. |
| Historical TrashNet Baseline Report | `Results/EcoSort_TrashNet_Test_Results.txt` | Verified in repo | ResNet18 test accuracy **78.9%** on **383** test images (`trash`: 21 support, 0.619 recall). |
| Historical TrashNet + Wrappers Report | `Results/EcoSort_Updated_Test_Results.txt` | Verified in repo | ResNet18 test accuracy **84.3%** on **383** test images (`trash`: 21 support, 0.714 recall; 15/21 correct). |
| Cross-Account Handoff Summary | `Documentation/CROSS_ACCOUNT_HANDOFF_2026-11-02.md` | Verified in repo | Records 28 Oct experiment splits (TrashNet 2,521; TACO plastic crops 309; RealWaste subset 3,587; Combined 6,417) and 1 Nov screening notes (5,244 candidates -> 5,072 label-review candidates; 7,463 decoded originals vs 7,462 source-class total). |
| Latest Reviewed-Data Checkpoint | Google Drive (`1UqeAxAmnx1X7js6QUFJOwtUEnMjQ4J5J`) | **Missing locally / Access needed** | Contains reported 7,463 recovered originals, 4,905 restored review records, and 990 visual decisions (548 approved, 518 deferred, 3,839 pending). |
| Full Handoff Bundle & Proposal Docs | Google Drive (`1xmmvWpjzQ2GN3CuFYzLudfJ5VZ93mwfW`) | **Access needed** (Local proposal PDF/DOCX found in `Downloads`) | Needed to cross-check exact proposal definitions and prior review sheets. |
| Trained Model Weights (`*.pth`) | `Models/` (gitignored) | **Not present locally** | Needed if running local inference checks. |

## 2. Class-Policy Questions for Prateek / Supervisor (Hold & Defer Until Confirmed)

Per instructions, uncertain/composite/wrapper items remain **deferred** (`review_decision = deferred`) with **no automatic relabeling**:

1. **Lined Drink Cartons (Tetra Pak / Juice & Milk Cartons):** Many source `cardboard` candidates are multi-layer foil/plastic-lined drink cartons. Should clean drink cartons remain `cardboard` (following TrashNet/RealWaste precedent), move to `trash` (non-recyclable composite in many local streams), or be `deferred` until E5/E6?
2. **Residual `trash` Scope (Masks, Diapers, Toothbrushes, Tissues, Sweepings):** Currently only **10 approved additions** are `trash`. Confirm which non-recyclable items (e.g., disposable masks, hygiene waste, toothbrushes, soiled food paper, multi-material broken items) are approved as `trash` so we can prioritize local capture.
3. **Metallized & Multi-Layer Snack Wrappers:** Earlier wrapper tests used `trash`, while our local 20-image batch (`wrapper_001`–`wrapper_020`) was placed under `plastic/` (`metallized_plastic_snack_wrapper`). Should all metallized/multi-layer wrappers be standardized under `trash` (with disposal-ambiguity documented) or kept `deferred` while six-class baselines are prioritized?
4. **Mixed-Scene & Composite Photos:** Confirm that any image containing multiple distinct waste classes in one frame is excluded/deferred from single-label 6-class classification manifests and reserved strictly for the Nov 23 YOLO detection dataset.
5. **7,463 vs. 7,462 Original Count Discrepancy:** Confirm whether the 1-image difference in the 1 Nov audit (`7,463` decoded originals vs `7,462` source-class sum) is already logged in the Drive checkpoint or if our audit script should flag the exact orphan/corrupt file upon Drive access.

## 3. Milestone Plan & Shortfall Honesty

- **17 Nov 2026:** Deliver this inventory, policy questions, availability, and Drive access request.
- **18–20 Nov 2026:** Once Drive checkpoint access is granted, run `Src/dataset_audit.py` to compute SHA256 hashes, verify exact counts, enforce group-aware train/val/test splits (excluding existing validation/test references), and deliver the 6-class CSV manifest (`relative_path, sha256, source_or_license, label, review_decision, group_id, split`).
- **21–23 Nov 2026:** Deliver TACO + local YOLO conversion, class mapping, group-aware splits, and visual bounding-box checks.
- **Target Reality Check:** Proposal-wide targets (600 local classification photos at 100/class, 300 mixed scenes with >=800 boxes, 200 fine/unknown) require active photo collection with Rahul over the coming days; current local fresh photos stand at 20 deferred wrapper images, so shortfall against the 600+300+200 target is flagged early.
