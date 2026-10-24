# EfficientNet run review — 24 October 2026

The uploaded EfficientNet result bundle passed ZIP integrity, strict CPU checkpoint loading, class/preprocessing validation, finite-weight and inference checks. Selected epoch 15 agrees with training history. The computed dataset membership signature matches the previously verified ResNet18 experiment; no manifest source-group or exact-pixel cross-split overlap was found. Saved test accuracy agrees with the confusion matrices. Dataset test predictions were not independently reproduced without the original dataset images.

| Recorded test result | ResNet18 | EfficientNet-B0 |
|---|---:|---:|
| TrashNet | 324/382 (84.82%) | 329/382 (86.13%) |
| RealWaste | 389/522 (74.52%) | 386/522 (73.95%) |
| TACO packaging recall | 45/46 (97.83%) | 45/46 (97.83%) |
| Combined accuracy | 758/950 (79.79%) | 760/950 (80.00%) |
| RealWaste plastic recall | 97/121 (80.17%) | 81/121 (66.94%) |

The overall accuracy change is two additional correct cases on the reused 950-image test set, with a meaningful plastic-recall tradeoff. This does not establish an overall deployment improvement.

Four previously used diagnostic image URLs were downloaded again and visually checked. EfficientNet classified all four as plastic: frosty-ground packet 93.39%, beach packet 38.00%, green Lay's 88.90%, white Cheetos wrapper 34.76%. The previous ResNet18 called the Cheetos photo paper. Two candidate decisions are close: beach plastic 38.00% vs paper 37.44%; Cheetos plastic 34.76% vs cardboard 29.97%. Scores are not calibrated correctness probabilities. Original historical image-byte hashes were not retained, so an exact byte-for-byte historical image comparison is unavailable. These four known examples are diagnostic, not an independent benchmark.

## Team next steps

- Prateek: preserve all three result bundles and present the data/architecture comparisons and limitations. Do not declare a universal winner from the four photos or the highest model score.
- Prince: prepare a fresh balanced evaluation set across all six classes, with varied backgrounds and item/session identifiers. Include white/printed wrappers, paper and cardboard; keep every view of a physical item in the same split. Freeze labels before predictions.
- Rahul: evaluate the candidate and previous checkpoints on that identical frozen set; report per-class recall, F1, false plastic predictions and model disagreement. Existing dataset tests are now repeatedly inspected diagnostics.
- Aditya: retain a configurable model path and architecture-aware loader; no automatic app switch is authorized by this report. Add a review flow for uncertain predictions only after choosing and validating its decision rule.

Full verification details and all class scores for the four photos are in `Results/EfficientNet_Run_Verification.json`. No photographs or checkpoint weights were uploaded to GitHub.
