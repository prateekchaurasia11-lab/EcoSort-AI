# EcoSort-AI team update — 24 October 2026

Training has finished in Prateek's Colab session. The results below were read from his completion screenshot; the trained checkpoint and full metrics ZIP have not yet been provided to this workspace for independent verification.

## Current result

| Item | Result |
|---|---|
| Model | Fresh ImageNet-initialized ResNet18, six classes |
| Training hardware | Tesla T4 in Google Colab |
| Epochs | 10; selected epoch 10 by validation macro F1 |
| Selected validation macro F1 | 0.850 (rounded notebook output) |
| TrashNet train / validation / test | 1,762 / 377 / 382 after duplicate exclusions |
| New TACO packaging train / validation / test | 218 / 45 / 46 |
| TrashNet test accuracy | 83.2% (rounded notebook output) |
| New packaging test recall | 100.0%: all 46 plastic-labelled holdout examples predicted plastic |
| New model filename | `ecosort_resnet18_taco_plastic.pth` |

The new data contains 309 usable crops: 186 Plastic film, 96 Other plastic wrapper and 27 Crisp packet objects. It is broader than snack packets alone. Wrappers, including metallized snack packets, map to the project's `plastic` packaging class. These labels do not establish chemical composition or recyclability.

The six-class order remains `cardboard, glass, metal, paper, plastic, trash`. The positive-only wrapper test measures recall; it does not establish plastic precision or overall six-class accuracy. The TrashNet test was inspected during earlier experiments. This run does not establish a general improvement over the earlier model without a comparable evaluation on fresh photos.

## Where the work is

- Pull request: https://github.com/prateekchaurasia11-lab/EcoSort-AI/pull/1
- Branch: `codex/taco-wrapper-training`. The PR is still a draft; these files are not yet on `main`.
- Notebook: `NoteBook/EcoSort_NEW_TACO_Plastic.ipynb`
- Data and training code: `Src/taco_data.py`, `Src/train_taco.py`
- New standalone predictor: `Src/predict_taco.py`
- Detailed method: `Documentation/TACO_PLASTIC_TRAINING.md`
- Screenshot-transcribed run summary: `Results/TACO_Colab_Run_Summary.json`

Six workflow tests passed, including an actual synthetic CPU training/checkpoint smoke test. Prateek's screenshot separately confirms completion of the real Colab GPU run. The previously committed data-preparation report describes the pre-training preparation check; it is not a training-results file.

## Tasks by member

| Member | Branch / responsibility | Next work | Deliverable |
|---|---|---|---|
| Prateek | `model-training`; model lead and presentation | Download and preserve `EcoSort_TACO_Plastic_Result.zip` from Colab. Share the checkpoint, metrics, history, package versions and manifests with the team through an agreed file-sharing location. Keep originals. Review PR #1 and coordinate integration after testing. | Shared trained-model bundle, reproducible results and presentation explaining dataset, labels, selection and limits. |
| Prince | `dataset-preparation`; dataset quality | Audit training-only crops and label policy. Collect a separate set of fresh local packaging photos plus examples from all six classes, with varied brands, lighting and backgrounds. Remove duplicates and related scenes across splits. Do not add the existing held-out examples to training. Freeze the new evaluation set before Rahul evaluates it. | A labelled, versioned evaluation manifest and image collection, source notes and any label issues. Do not silently relabel the existing experiment. |
| Aditya | `app-interface`; app integration | Use the new checkpoint after Prateek shares it. Keep `class_names` order, RGB conversion, 224×224 resize, ImageNet mean/std and ResNet18 architecture compatible with `Src/predict_taco.py`. Wire upload/prediction into the interface; run inference without retraining. Avoid hardcoding the old wrappers checkpoint. | Working image-upload prediction flow with the selected model version and clear class scores. Model scores must not be presented as guaranteed correctness. |
| Rahul | `integration-testing`; verification | Verify the shared checkpoint loads and matches its SHA-256 checksum. Run the same frozen fresh-image evaluation through the command-line predictor and app. Cover all six classes, wrappers, invalid uploads and expected CPU use. Preserve expected labels and outputs; compare older and new models on the same images if reporting improvement. | Test report, per-class errors/confusion matrix, wrapper recall and precision on a set with both positive and negative examples, integration bugs and demo checklist. |

## Setup and integration

Use the existing EcoSort-AI checkout at `C:\Users\Prateek\Desktop\EcoSort-AI` on Prateek's machine, or each member's own clone. Keep the main `.venv`, annotation `.labelenv`, and inference `.inferenv` separate. Training runs in Colab; teammates doing app/data/testing work do not need to retrain the model.

To refresh remote branch information without switching branches or modifying working files, type these in Command Prompt from the existing repository folder:

```bat
git status
git fetch origin
```

Review PR #1 or read this handoff directly from the remote branch:

```bat
git show origin/codex/taco-wrapper-training:Documentation/TEAM_UPDATE.md
```

Continue separate responsibilities on the assigned branches. Commit code and small reports there and open PRs to `main`. Coordinate with Prateek before merging; do not force-push teammates' branches. Pulling `main` alone will not include the new pipeline while PR #1 remains unmerged.

After the pipeline is integrated and the trained ZIP is shared, extract its `Models` and `Src` folders into the project without replacing the existing virtual environments or datasets. In the active inference environment, run:

```bat
python Src\predict_taco.py "C:\path\to\fresh-photo.jpg"
```

The predictor defaults to `Models/ecosort_resnet18_taco_plastic.pth`. The checkpoint exposes `class_names`, `model_state_dict`, `image_size`, `mean` and `std`. Preserve that inference contract in the app. The output ZIP records Colab package versions; do not assume Windows CPU and Colab CUDA package versions must be identical.

Do not commit `.venv`, `.inferenv`, `.labelenv`, image collections, ZIP archives or `.pth` weights. Share large artifacts separately and record their filenames/checksums. GitHub currently contains the code and reports; it does not contain the trained model from this Colab run.
