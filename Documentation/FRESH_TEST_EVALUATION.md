# Compare saved models on the 30-photo sample

`Src/evaluate_fresh_test.py` evaluates the latest saved ResNet18 and EfficientNet checkpoints. It never trains or saves model weights. Run it from the EcoSort-AI project folder using the existing inference environment:

```bat
.inferenv\Scripts\python.exe Src\evaluate_fresh_test.py
```

The script can also live in Downloads; invoke its absolute path while the current directory remains the project root. `--project` accepts another explicit project folder if needed.

Required paths:

- `data/fresh_test/{cardboard,glass,metal,paper,plastic,trash}`: exactly five images per class.
- `evaluation_models/resnet18/Models/ecosort_resnet18_realwaste.pth`
- `evaluation_models/efficientnet/Models/ecosort_efficientnet_b0.pth`

The two model folders come from extracting their respective result ZIPs. Original datasets, model files and environments remain unchanged. Dependencies are the existing PyTorch, torchvision and Pillow inference packages; scikit-learn is not needed. Checkpoints use weights-only loading, class/preprocessing checks and their matching architecture.

Exact duplicated evaluation photos and wrong image counts stop evaluation. Candidate folder labels are the expected labels; they do not establish material composition. All trash samples in this starter pack are cigarette butts, and some other photos show clean packaging. Five examples per class are insufficient for a representative final benchmark. Keep all these photos out of training and validation.

A new timestamped `Results/fresh_test_...` folder contains `comparison.json` (accuracy, per-class precision/recall/F1, confusion matrices, checkpoint hashes and image fingerprints) and `predictions.csv` (both predictions and all six model scores for every image). Scores are not calibrated correctness probabilities. The script does not automatically select or promote a model.

Validation: five unit checks cover metric arithmetic, perfect classification, invalid prediction counts, wrong extraction layout and duplicate rejection. A complete 30-image CPU smoke run used the actual uploaded EfficientNet checkpoint and a synthetic ResNet18 baseline to verify both loaders and CSV/JSON output. The synthetic baseline's score is not a real ResNet18 comparison; use the user's actual extracted checkpoint for the comparison.
