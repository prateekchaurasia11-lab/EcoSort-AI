# Improve single-item classification: EfficientNet-B0 experiment

## Scope

EcoSort predicts one class for the whole photo: cardboard, glass, metal, paper, plastic or trash. This experiment improves that classifier; it does not add object boxes or reliably separate multiple materials in one image. An improvement is not guaranteed merely by changing the architecture or increasing its score.

The latest verified ResNet18 checkpoint selected epoch 14 and its saved results report 324/382 TrashNet, 389/522 RealWaste and 45/46 TACO packaging examples correct. On the four previously downloaded diagnostic photos it classified three as plastic and incorrectly classified the white Cheetos packet as paper with a 95.18% model score. The old TACO model classified all four as plastic. The four photos are deliberately excluded from this training experiment and cannot serve as an untouched final benchmark.

## What changes

The candidate is Torchvision's ImageNet-initialized EfficientNet-B0. The existing three datasets and their exact class/split memberships stay fixed. Class-weighted loss, image augmentation and the project's RGB / 224×224 resize / ImageNet-normalization preprocessing remain explicit, so earlier and candidate predictions receive the same images. This is the project's preprocessing protocol, not a claim to use the pretrained weights' default resize-and-center-crop recipe.

Training has 3 classifier-only epochs followed by up to 12 epochs training the classifier and EfficientNet feature stages 6–8. Other layers and their batch-normalization statistics remain frozen. Learning rates are 1e-3 for head warmup, then 1e-4 for the classifier and 1e-5 for the late feature stages. Early stopping allows 5 consecutive non-improving epochs in the fine-tuning stage. Selection uses combined validation macro F1, then validation accuracy for ties.

All previous ResNet checkpoints and predictors remain available. Candidate weights have architecture metadata and a distinct filename: `ecosort_efficientnet_b0.pth`. `efficientnet_model.py` builds the matching architecture; an EfficientNet checkpoint must not be loaded into a ResNet18 object.

## Run in Google Colab

Open `NoteBook/EcoSort_EfficientNet_Comparison.ipynb`, select a T4 GPU, then run its one code cell. The notebook reuses available inputs where possible. If prompted, select BOTH required ZIPs together:

1. `EcoSort_TrashNet_split.zip`
2. `EcoSort_RealWaste_Result.zip` — the latest completed ResNet18 result bundle, not the earlier TACO-only result.

TACO and RealWaste data caches from the earlier notebook are reused if available. Otherwise the same official data is downloaded. RealWaste is about 657 MB. The notebook reconstructs the dataset, verifies actual cached TACO crop pixels and compares source, decoded-pixel fingerprint, class and split against the previous result bundle. Any membership mismatch stops training instead of silently changing the comparison. Original input image files are never overwritten by model training. Source availability or decoding differences may cause the strict check to stop; preserve earlier source images for exact reproduction.

The expected previous split has 4,470 training, 997 validation and 950 test images. Test metrics are calculated only after validation selects a checkpoint. Both candidate and latest ResNet18 are evaluated on identical source-specific test images. These test sets have already been inspected, so results are diagnostic; collect a new independent evaluation set before deployment claims. A wrapper-only set measures recall and cannot establish plastic precision by itself.

## Results and next decision

Download `EcoSort_EfficientNet_Result.zip`. It includes candidate weights, baseline comparison, selected epoch, training history, package versions, frozen membership signature, source manifests, attribution and matching inference code. The model is not automatically promoted into the app.

Compare per-class errors, macro F1 and source-specific results. Test the candidate and both previous models on the same separately collected six-class photo set. A higher model score on one known example is not sufficient evidence for promotion. Keep ResNet18 if the candidate fails the intended use case.

In the inference environment after extracting the bundle into the project:

```bat
python Src\predict_efficientnet.py "C:\path\to\fresh-photo.jpg"
```

## How more data should be collected

Additional images must contain labelled garbage objects. Background-only carpet photos do not teach a six-class waste classifier which material an object belongs to. For a useful local dataset, photograph each of the six classes against several backgrounds, distances, viewing angles and lighting conditions. Include clean and crumpled objects, printed wrappers, white wrappers, transparent packaging, foil and paper as visually similar positive/negative examples. Avoid assigning every plastic example one background and every paper example another background.

Keep all photographs of the same physical item or capture session together in one split. Do not treat crops, flips or augmented copies as independent new observations. Reserve a balanced evaluation set before looking at either model's predictions, and do not train on that set after seeing failures. Keep the current four internet photos diagnostic-only.

Team responsibilities:

- Prateek: run the controlled architecture comparison, preserve results and explain selection/limits in the presentation.
- Prince: audit training labels and collect 20–30 new original photos per class with varied objects/backgrounds. Keep source/item/session IDs and freeze an independent evaluation set. Flag ambiguous mixed-material objects rather than guessing their composition.
- Aditya: keep the app model configurable and use an architecture-aware loader. Wait for the chosen checkpoint to be verified before switching the app.
- Rahul: compare all candidate checkpoints on the frozen local evaluation set, including false plastic predictions, per-class F1 and app/CLI consistency.

## Validation status and sources

This notebook is prepared code, not a completed GPU training run. See the PR for tests actually executed. Synthetic CPU tests do not establish trained accuracy. The previously uploaded ResNet18 checkpoint was independently loaded and its saved metrics/history checked; test scores were not independently recomputed without the original TrashNet test images.

- EfficientNet-B0: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html
- Tan and Le (2019), EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks: https://arxiv.org/abs/1905.11946
- TACO: Proença and Simões (2020), https://github.com/pedropro/TACO. Source categories and per-image URLs/licenses remain in the TACO manifest; toolkit licensing is not a blanket license for photographs.
- RealWaste: Single, Iranmanesh and Raad (2023), https://doi.org/10.24432/C5SS4G, CC BY 4.0.
- Existing label/split conventions: `Documentation/REALWASTE_EXPERIMENT.md` and `Documentation/TACO_PLASTIC_TRAINING.md`.
