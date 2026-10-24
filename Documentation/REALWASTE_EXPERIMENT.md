# Next EcoSort experiment: add RealWaste

## Why add data first

The present ResNet18 classified four downloaded packaging photos as plastic with top scores of roughly 49–61%. That small positive-only check cannot establish general accuracy, plastic precision or calibrated confidence. Increasing the number of epochs alone may fit the same data more closely without improving performance on new photos.

This experiment keeps ResNet18 and adds examples from all six compatible RealWaste categories. It compares the previous TACO checkpoint and the new model on exactly the same source-specific test sets after validation-based selection. The previous model remains available. Improvement is an experimental outcome, not a guarantee.

## Data and labels

Source: https://archive.ics.uci.edu/dataset/908/realwaste

Citation: Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G. License: CC BY 4.0.

| RealWaste category | EcoSort class | Source images |
|---|---|---:|
| Cardboard | cardboard | 461 |
| Glass | glass | 420 |
| Metal | metal | 790 |
| Paper | paper | 500 |
| Plastic | plastic | 921 |
| Miscellaneous Trash | trash | 495 |
| Total used | | 3,587 |

Food Organics, Textile Trash and Vegetation are excluded because the current six-class taxonomy has no explicit matching classes. They are not silently merged into trash. Existing packaging labels remain the project convention: TACO film/wrappers/crisp packets map to plastic, including metallized snack packets. These are not verified composition or recycling instructions. RealWaste's miscellaneous trash category is broad and should be audited against the project's intended meaning of trash.

## Split and actual preparation check

Existing TrashNet and TACO splits are preserved. New RealWaste splitting uses seed 2026 and groups each category's consecutive filename numbers into blocks of 20, then merges near-identical images with whole-image dHash distance <= 4. Groups stay together and are stratified by majority class before approximate 70/15/15 assignment. Exact copies with conflicting labels are excluded; groups containing exact pixel matches to any old source are excluded, and same-class exact copies are reduced to one.

This is a conservative filename-order heuristic, not verified capture-session grouping. Neither it nor perceptual hashes can detect every related scene. Cross-dataset screening uses exact decoded-pixel hashes, so transformed overlaps may remain. RealWaste represents one landfill environment, not all deployment settings. Source-specific and independent fresh-photo results remain necessary.

Local preparation on 24 October 2026 checked the downloaded official archive and produced:

| Source | Training | Validation | Test |
|---|---:|---:|---:|
| Clean TrashNet | 1,762 | 377 | 382 |
| TACO packaging | 218 | 45 | 46 |
| New RealWaste | 2,490 | 575 | 522 |
| Combined | 4,470 | 997 | 950 |

These are actual preparation counts, not newly trained GPU results. See `Results/RealWaste_Preparation_Report.json`. Later downloads may vary if upstream files change. No dataset images or model weights are committed to GitHub.

## Current-model validation check

The actual previously uploaded TACO checkpoint was evaluated on the new RealWaste validation split only: **245/575 correct (42.61% accuracy), macro F1 0.4091**. This does not imply that a newly trained model will improve. The 522 new RealWaste test images have not been evaluated here. Full per-class validation results and checkpoint identity are recorded in `Results/RealWaste_Current_Model_Validation.json`.

## How to run

Open `NoteBook/EcoSort_TrashNet_TACO_RealWaste.ipynb` in Google Colab with a T4 GPU and run its single code cell.

The notebook tries to reuse the TrashNet ZIP and current TACO checkpoint in the existing Colab session. If either is missing, it asks for the required files. Keep these two ZIPs available and select the missing ones together in the upload dialog:

- `EcoSort_TrashNet_split.zip`
- `EcoSort_TACO_Plastic_Result.zip` (the previous trained result bundle)

The notebook reuses existing TACO crops where available and automatically downloads RealWaste (about 657 MB). It prints the counts and shows training-only examples. It trains a **fresh ImageNet-initialized ResNet18** for up to 15 epochs (3 head, up to 12 layer4/head), with early stopping after 5 epochs without validation selection improvement during fine-tuning. Frozen layers and their batch-normalization statistics stay frozen. Class-weighted loss and augmentation are retained. The selected checkpoint is the highest combined validation macro F1, accuracy breaking ties.

After checkpoint selection, the previous and new models are both evaluated on the same test images, separately for TrashNet, TACO and RealWaste. A wrapper-only result is recall, not six-class accuracy or plastic precision. The previously inspected TrashNet and TACO holdouts are diagnostic. Do not use the four online example photos as a new untouched benchmark or add test/validation photos to training.

The result ZIP is `EcoSort_RealWaste_Result.zip`, with new weights `Models/ecosort_resnet18_realwaste.pth`, metrics/baseline comparison, training history, manifests, class counts, source attribution and package versions. The previous model is not overwritten. The compatible CLI is `Src/predict_realwaste.py`.

## Team tasks

- Prateek: run the new experiment, preserve both result bundles, and review the source-specific comparison before choosing the deployed model. Explain the data/split/model-selection process in the presentation.
- Prince: review training-only RealWaste categories, especially miscellaneous trash, and build a separate labelled local-photo evaluation set covering all six classes and difficult packaging. Freeze labels before evaluation.
- Aditya: keep the current app model until the new checkpoint is verified. Make the model path/version configurable; load checkpoint class order and preprocessing. The new filename differs; inference does not require retraining.
- Rahul: evaluate both checkpoints on the identical frozen local-photo set; report per-class recall/F1, wrapper recall and false plastic predictions, plus app/CLI consistency. Preserve test labels and model SHA-256 checksums.

After measuring this data experiment, a separate architecture comparison (for example EfficientNet-B0) can use the same frozen data protocol. Do not change the dataset and architecture simultaneously and attribute the result to one change. Any new architecture also needs a matching inference loader; changing a model filename is insufficient.
