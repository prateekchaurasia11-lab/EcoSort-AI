# EcoSort-AI — Cross-Account Project Status (2 November 2026)

This file records a verified status checkpoint for continuing the minor project across accounts. The full detailed handoff is in the team's restricted Google Drive folder and should be used with the approved proposal and progress report:
- [Project handoff folder](https://drive.google.com/drive/folders/1xmmvWpjzQ2GN3CuFYzLudfJ5VZ93mwfW)
- [Complete handoff document](https://docs.google.com/document/d/17C0dFjvdFMylfsRNkjtg1I3jK3QoJft3YEbr8aH1aCs/edit?usp=drivesdk)

## Current direction
**Improve all six classes, not only plastic wrappers.** Collect/review clean, independently sourced examples; fix labels, item/scene overlap and class imbalance before six-class model training. Focus on validation macro-F1 and per-class recall. Do not repeatedly optimize on known 30-photo diagnostic examples.

The user's latest reported 1 November Garbage Dataset screening has not yet produced a verified new training checkpoint. Latest reported stages include 5,244 candidate images after initial exclusions, 172 additional near-duplicate removals, 5,072 label-review candidates. Another prior review stage reported 4,939 candidates, 298 conflicts and 208 review sheets; compare actual local audit manifests before interpreting these as one final count. Reported 7,463 decoded originals vs a source-class total of 7,462 is unresolved. Preserve original data and verify.

## Experiment evidence through 28 October
Six labels in checkpoint order: `cardboard, glass, metal, paper, plastic, trash`.
- Clean TrashNet 2,521 = train 1,762, val 377, test 382 (6 conflicting copies removed).
- TACO compatible plastic packaging crops 309 = train 218, val 45, test 46.
- RealWaste compatible subset 3,587 = train 2,490, val 575, test 522.
- Combined 6,417 = train 4,470, val 997, test 950.
- ResNet18 test: TrashNet 324/382, RealWaste 389/522, TACO plastic recall 45/46, combined 758/950 (79.79%).
- EfficientNet-B0 test: TrashNet 329/382, RealWaste 386/522, TACO plastic recall 45/46, combined 760/950 (80.00%).
- Separate, small, inspected 30-image web diagnostic: ResNet18 14/30 (macro-F1 .4436); EfficientNet 23/30 (macro-F1 .7604). Not representative final accuracy; both missed two snack packets and EfficientNet confused a clear plastic bottle with glass.
- Keep old checkpoint versions, metrics and source manifests. Model weights, ZIPs and datasets are not stored in the repository.

## Approved project vs exploratory prototypes
Approved E1 Custom CNN, E2 MobileNetV2, E3 balancing, E4 targeted augmentation, E5 calibrated confidence/unknown rejection, E6 lightweight object detector + best crop classifier. Streamlit UI and disposal guidance remain planned. Current ResNet18 / EfficientNet-B0 are supplementary exploratory comparisons, not proof the approved E1–E6 stages have been completed. Approved proposal discusses TensorFlow/Keras while current work uses PyTorch; supervisor alignment needed.

## Git status and team branches
Three staged draft PRs need review and dependency-aware integration: #1 `codex/taco-wrapper-training` -> `main`; #2 `codex/realwaste-training` -> #1 branch; #3 `codex/efficientnet-comparison` -> #2 branch. Do not assume their contents exist on main. Work branches: Prateek `model-training` (model/reviews), Prince `dataset-preparation` (images/labels), Aditya `app-interface` (UI), Rahul `integration-testing` (independent metrics/integration). Work completion is not verified merely because branches exist.

## Next actions
1. Inspect local dataset audit outputs and reconcile reported counts/labels; no silent material relabeling.
2. Freeze clean split with item/scene grouping and untouched local evaluation.
3. Train controlled, approved Custom CNN and MobileNetV2 baselines; report F1, recalls and errors versus previous checkpoints using matching evaluation.
4. Preserve new model/checkpoints in restricted Drive, small scripts/results in PRs, then proceed to calibrated rejection, detector and application.

Full original research proposal, October 28 progress report, and October 29 slides are provided in the handoff bundle; consult them before altering scope.