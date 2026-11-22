# Train EcoSort on new plastic packaging

Open `NoteBook/EcoSort_NEW_TACO_Plastic.ipynb` in Google Colab, select a T4 GPU, and run the single code cell. Upload **EcoSort_TrashNet_split.zip**, containing the existing six-class train/val/test folders. Do not upload an old `.pth` model. Keep the notebook open while the runtime works. Download the output ZIP when prompted.

The notebook contains the preparation and training source, so it can run without GitHub credentials. It automatically downloads TACO annotations and source images, extracts labelled plastic-packaging objects, combines them with TrashNet, and trains a new ResNet18 from ImageNet weights. It does not use the three hand-picked photos or earlier wrapper-labelled checkpoints. Default training: 3 head epochs, then 7 epochs updating layer4 and the head. The six output classes remain cardboard, glass, metal, paper, plastic, trash.

## Labels and source

Source: https://github.com/pedropro/TACO, pinned annotation commit `70479ad9adc3da971a8ae75ac97eb1026910eec7`.

The official annotations contain 451 Plastic film, 260 Other plastic wrapper and 39 Crisp packet objects (750 candidates, not 750 training images). Only successfully downloaded, unique crops with both sides at least 48 pixels enter the experiment. Actual counts and failed downloads are printed and saved. The workflow refuses to train with fewer than 100 new training examples or fewer than 15 validation or test examples.

**Project convention:** these three packaging categories map to plastic, including metallized snack packets. This is an appearance/packaging label, not verified chemical composition or recycling guidance. The broader category counts do not mean hundreds of snack-packet photographs.

Images are downloaded from the URLs provided by TACO. Source URLs, bounding boxes, annotation IDs, category names, image-license metadata and annotation hash are preserved in the manifest. A missing image license remains missing; the toolkit's MIT license should not be assumed to license every photograph. Review original source permissions before redistributing image collections. No images are committed to this repository.

## Splits and measurement

- All crops from the same source image stay in the same split. Exact duplicate crops and whole scenes within dHash distance 4 are grouped before splitting. This is a heuristic, not a guarantee that every related scene is detected.
- Source groups are split approximately 70/15/15 with seed 42, stratified on presence of crisp packets when enough groups exist.
- TrashNet copies crossing splits or class labels are all excluded from this experiment; other identical copies within one class/split are reduced to one. Uploaded originals are preserved. Exact decoded-pixel collisions with TrashNet exclude the whole matching TACO source group.
- Selection uses combined validation macro F1, with accuracy breaking ties. Test inference happens only after selecting the final checkpoint.
- Report TrashNet six-class test metrics and new-wrapper plastic recall separately. The TACO subset contains only plastic-labelled examples, so it cannot establish plastic precision or six-class accuracy by itself.
- Earlier EcoSort experiments already examined the TrashNet test set. Its results are diagnostic rather than an untouched final benchmark. TACO holdout results also do not establish performance on independently collected local packet photographs.
- TACO object crops have different backgrounds/framing from TrashNet whole images. Check fresh, unused real photos before claiming an improvement.

## Outputs

`EcoSort_TACO_Plastic_Result.zip` includes `Models/ecosort_resnet18_taco_plastic.pth`, metrics, training history, dataset records, excluded-image list, TACO provenance/download errors, the training-only preview, a compatible prediction script, documentation and the Colab package versions. It does not include the downloaded image collection. Keep all output files together for reproducibility. Output names differ from earlier models.

After extracting the ZIP into your project, predict in the inference environment with:

```bat
python Src\predict_taco.py "C:\path\to\new-photo.jpg"
```

Reproduction outside Colab requires Python, PyTorch/torchvision, Pillow, NumPy and scikit-learn with a CUDA GPU for the training function. Exact versions are captured at run time, rather than assuming the Windows CPU package list matches Colab.

## Dataset preparation check (24 October 2026)

All 475 selected source images were downloaded. Filtering left **309 new crops: 218 train, 45 validation, 46 test**. Categories: 186 plastic film, 96 other wrappers and 27 crisp packets. Another 441 candidate crops were too small. Source-group and exact-crop overlap checks passed. These are preparation counts, not trained accuracy. Availability changes may affect later downloads.

## Validation of the workflow

See the pull request for checks actually run. Preparing a notebook or passing a synthetic CPU smoke test does not mean the real dataset has been GPU-trained. Real training scores are produced only by running the Colab notebook.

Please cite: Pedro F. Proença and Pedro Simões, *TACO: Trash Annotations in Context for Litter Detection*, 2020, https://arxiv.org/abs/2003.06975.
