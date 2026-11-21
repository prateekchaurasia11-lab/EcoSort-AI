import argparse
import csv
import hashlib
import json
import time
import zipfile
from collections import defaultdict
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

if __package__:
    from .e2_preprocessing import decode_pixels
else:
    from e2_preprocessing import decode_pixels

CLASSES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]

EXPECTED_SHA256 = (
    "466d5514de963ccbb3606ff3cbe39c77c42fde2442ab3f108d555722e408a27e"
)

EXPECTED_PREPROCESSING = (
    "RGB 0..255; embedded Rescaling(1/127.5, offset=-1)"
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(manifest_csv):
    with Path(manifest_csv).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        required = {"split", "label", "source_archive_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    if not rows:
        raise ValueError("Manifest is empty")

    records = []
    for line_number, row in enumerate(rows, start=2):
        split = (row.get("split") or "").strip().lower()
        label = (row.get("label") or "").strip().lower()
        source = (row.get("source_archive_path") or "").strip().replace("\\", "/")

        if split not in {"train", "val", "test"}:
            raise ValueError(
                f"Invalid split at CSV line {line_number}: {split!r}"
            )
        if label not in CLASSES:
            raise ValueError(
                f"Invalid label at CSV line {line_number}: {label!r}"
            )

        source_path = PurePosixPath(source)
        if (
            not source
            or source_path.is_absolute()
            or ".." in source_path.parts
            or source_path.name in {"", ".", ".."}
        ):
            raise ValueError(
                f"Unsafe or empty source_archive_path at line {line_number}"
            )
        if source_path.parent.name.lower() != label:
            raise ValueError(
                f"Source path class does not match label at line {line_number}"
            )

        member = f"split/{split}/{label}/{source_path.name}"
        group_id = (
            (row.get("group_id") or row.get("source_group_id") or "").strip()
        )
        records.append({
            "split": split,
            "label": label,
            "label_id": CLASSES.index(label),
            "source_archive_path": source,
            "archive_path": member,
            "group_id": group_id,
            "line_number": line_number,
        })

    return records, fields


def audit_dataset_manifest(manifest_csv, dataset_zip):
    records, fields = read_manifest(manifest_csv)
    seen_paths = set()
    hashes = defaultdict(list)
    groups = defaultdict(set)
    split_counts = defaultdict(int)
    label_counts = defaultdict(int)
    missing_group_ids = 0
    missing_files = []
    decode_errors = []

    with zipfile.ZipFile(dataset_zip) as archive:
        members = set(archive.namelist())

        for record in records:
            member = record["archive_path"]
            if member in seen_paths:
                raise ValueError(f"Duplicate manifest archive path: {member}")
            seen_paths.add(member)

            if member not in members:
                missing_files.append(member)
                continue

            image_bytes = archive.read(member)
            digest = sha256_bytes(image_bytes)
            record["sha256"] = digest

            try:
                with Image.open(BytesIO(image_bytes)) as image:
                    image.verify()
            except Exception as exc:
                decode_errors.append({
                    "path": member,
                    "error": str(exc),
                })

            hashes[digest].append(record)
            if record["group_id"]:
                groups[record["group_id"]].add(record["split"])
            else:
                missing_group_ids += 1

            split_counts[record["split"]] += 1
            label_counts[record["label"]] += 1

    if missing_files:
        raise ValueError(
            f"{len(missing_files)} manifest images are missing from ZIP; "
            f"examples: {missing_files[:5]}"
        )
    if decode_errors:
        raise ValueError(
            f"{len(decode_errors)} images failed PIL verification; "
            f"examples: {decode_errors[:3]}"
        )

    cross_split_duplicates = []
    conflicting_labels = []
    within_split_duplicate_groups = []

    for digest, items in hashes.items():
        item_splits = {item["split"] for item in items}
        item_labels = {item["label"] for item in items}

        if len(item_labels) > 1:
            conflicting_labels.append({
                "sha256": digest,
                "paths": [item["archive_path"] for item in items],
                "labels": sorted(item_labels),
            })
        if len(item_splits) > 1:
            cross_split_duplicates.append({
                "sha256": digest,
                "paths": [item["archive_path"] for item in items],
                "splits": sorted(item_splits),
            })
        elif len(items) > 1:
            within_split_duplicate_groups.append({
                "sha256": digest,
                "paths": [item["archive_path"] for item in items],
                "split": items[0]["split"],
            })

    group_leaks = [
        {"group_id": group_id, "splits": sorted(splits)}
        for group_id, splits in groups.items()
        if len(splits) > 1
    ]

    if conflicting_labels:
        raise ValueError(
            "Exact duplicate image bytes have conflicting labels: "
            f"{conflicting_labels[:3]}"
        )
    if cross_split_duplicates:
        raise ValueError(
            "Exact duplicate image bytes cross dataset splits: "
            f"{cross_split_duplicates[:3]}"
        )
    if group_leaks:
        raise ValueError(
            f"Capture groups cross dataset splits: {group_leaks[:5]}"
        )

    split_summary = {
        split: int(split_counts.get(split, 0))
        for split in ("train", "val", "test")
    }
    label_summary = {
        label: int(label_counts.get(label, 0))
        for label in CLASSES
    }

    return records, {
        "manifest_rows": len(records),
        "split_counts": split_summary,
        "label_counts": label_summary,
        "unique_content_hashes": len(hashes),
        "missing_files": len(missing_files),
        "decode_errors": len(decode_errors),
        "cross_split_exact_duplicates": 0,
        "conflicting_label_duplicates": 0,
        "within_split_duplicate_groups": within_split_duplicate_groups,
        "group_id_column_present": (
            "group_id" in fields or "source_group_id" in fields
        ),
        "nonempty_group_ids": len(groups),
        "rows_without_group_id": missing_group_ids,
        "cross_split_group_leaks": 0,
        "group_leakage_check": (
            "performed" if missing_group_ids == 0 else
            f"not available for all rows: {missing_group_ids} rows lack group IDs"
        ),
    }


def classification_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)

    if labels.ndim != 1 or predictions.shape != labels.shape:
        raise ValueError("Labels and predictions must be matching 1-D arrays")
    if labels.size == 0:
        raise ValueError("Cannot calculate metrics for an empty split")
    if (
        (labels < 0).any() or (labels >= len(CLASSES)).any()
        or (predictions < 0).any()
        or (predictions >= len(CLASSES)).any()
    ):
        raise ValueError("Labels or predictions contain invalid class IDs")

    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)

    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)

    precision = np.divide(
        true_positive, predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0
    )
    recall = np.divide(
        true_positive, support,
        out=np.zeros_like(true_positive),
        where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0
    )

    per_class = {}
    for index, class_name in enumerate(CLASSES):
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "sample_count": int(labels.size),
        "accuracy": float(true_positive.sum() / labels.size),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate(checkpoint, protocol_path, manifest_csv, dataset_zip,
             split, batch_size=32, device="auto",
             output_json=None, output_npz=None, decoder="tensorflow"):
    if decoder not in {"tensorflow", "pil"}:
        raise ValueError("Decoder must be tensorflow or pil")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val or test")
    if device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be auto, cpu or gpu")

    checkpoint = Path(checkpoint)
    protocol_path = Path(protocol_path)
    manifest_csv = Path(manifest_csv)
    dataset_zip = Path(dataset_zip)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("classes") != CLASSES:
        raise ValueError("Protocol class order does not match E2 class order")
    if protocol.get("architecture") != "MobileNetV2":
        raise ValueError("Unexpected model architecture")
    if protocol.get("input_size") != [224, 224]:
        raise ValueError("Unexpected protocol input size")
    if protocol.get("preprocessing") != EXPECTED_PREPROCESSING:
        raise ValueError("Unexpected model preprocessing protocol")

    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != EXPECTED_SHA256:
        raise ValueError("Checkpoint SHA-256 mismatch")

    records, audit = audit_dataset_manifest(manifest_csv, dataset_zip)
    selected = [row for row in records if row["split"] == split]
    if not selected:
        raise ValueError(f"No rows found for split {split!r}")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise ValueError("GPU was requested but TensorFlow reports no GPU")

    device_name = (
        "/CPU:0" if device == "cpu"
        else "/GPU:0" if device == "gpu"
        else None
    )
    device_context = tf.device(device_name) if device_name else nullcontext()

    load_start = time.perf_counter()
    model = tf.keras.models.load_model(checkpoint, compile=False)
    model_load_seconds = time.perf_counter() - load_start

    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError("Checkpoint input shape is not 224x224x3")
    if model.output_shape[-1] != len(CLASSES):
        raise ValueError("Checkpoint output shape is not six classes")

    rescaling_layers = [
        layer for layer in model.layers
        if layer.__class__.__name__ == "Rescaling"
    ]
    if len(rescaling_layers) != 1:
        raise ValueError("Expected exactly one embedded Rescaling layer")
    rescaling = rescaling_layers[0]
    if (
        not np.isclose(float(rescaling.scale), 1 / 127.5, atol=1e-9)
        or not np.isclose(float(rescaling.offset), -1.0, atol=1e-9)
    ):
        raise ValueError("Embedded Rescaling layer does not match E2 protocol")

    labels = np.asarray([row["label_id"] for row in selected], dtype=np.int64)
    paths = np.asarray([row["archive_path"] for row in selected], dtype=str)
    probabilities = []
    preprocessing_seconds = 0.0
    inference_seconds = 0.0
    warmed_up = False

    with zipfile.ZipFile(dataset_zip) as archive, device_context:
        for start in range(0, len(selected), batch_size):
            batch_rows = selected[start:start + batch_size]
            prep_start = time.perf_counter()
            batch_images = []

            for row in batch_rows:
                image_bytes = archive.read(row["archive_path"])
                pixels = decode_pixels(image_bytes, decoder)
                batch_images.append(pixels.astype(np.float32))

            batch = np.stack(batch_images).astype(np.float32)
            preprocessing_seconds += time.perf_counter() - prep_start

            if not warmed_up:
                model.predict(batch[:min(8, len(batch))], verbose=0)
                warmed_up = True

            inference_start = time.perf_counter()
            batch_scores = model.predict(
                batch, batch_size=batch_size, verbose=0
            )
            inference_seconds += time.perf_counter() - inference_start

            batch_scores = np.asarray(batch_scores, dtype=np.float32)
            if batch_scores.shape != (len(batch_rows), len(CLASSES)):
                raise ValueError("Model returned an unexpected score shape")
            if (
                not np.isfinite(batch_scores).all()
                or (batch_scores < 0).any()
                or not np.allclose(
                    batch_scores.sum(axis=1), 1.0, atol=1e-4
                )
            ):
                raise ValueError("Model returned invalid probability scores")

            probabilities.append(batch_scores)

    probabilities = np.concatenate(probabilities, axis=0)
    predictions = probabilities.argmax(axis=1)
    metrics = classification_metrics(labels, predictions)

    report = {
        "model": {
            "architecture": "MobileNetV2",
            "checkpoint_sha256": checkpoint_hash,
            "classes": CLASSES,
            "input_size": [224, 224],
            "preprocessing": EXPECTED_PREPROCESSING,
            "scores_calibrated": False,
            "decoder": decoder,
            "preprocessing_version": "e2-tf-bilinear-224-" + decoder,
        },
        "dataset": {
            "manifest_path": str(manifest_csv),
            "manifest_sha256": sha256_file(manifest_csv),
            "archive_path": str(dataset_zip),
            "archive_sha256": sha256_file(dataset_zip),
            "selected_split": split,
            "audit": audit,
            "split_use_note": (
                "The test split is diagnostic only; do not tune model settings "
                "or thresholds against it."
                if split == "test" else
                "Record this split's role and keep test data untouched."
            ),
        },
        "runtime": {
            "tensorflow_version": tf.__version__,
            "keras_version": getattr(tf.keras, "__version__", "unknown"),
            "available_devices": [
                str(item) for item in tf.config.list_physical_devices()
            ],
            "requested_device": device,
            "batch_size": batch_size,
            "warmup_images": min(8, len(selected)),
            "model_load_seconds": model_load_seconds,
            "preprocessing_seconds": preprocessing_seconds,
            "inference_seconds": inference_seconds,
            "preprocessing_ms_per_image": (
                preprocessing_seconds / len(selected) * 1000
            ),
            "inference_ms_per_image": (
                inference_seconds / len(selected) * 1000
            ),
        },
        "metrics": metrics,
    }

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if output_npz:
        output_npz = Path(output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_npz,
            probabilities=probabilities,
            labels=labels,
            files=paths,
            checkpoint_sha256=np.asarray(checkpoint_hash),
            decoder=np.asarray(decoder),
            preprocessing_version=np.asarray("e2-tf-bilinear-224-" + decoder),
        )

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the verified E2 MobileNetV2 checkpoint on a manifest split."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--dataset-zip", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--output-json")
    parser.add_argument("--output-npz")
    parser.add_argument("--decoder", choices=["tensorflow", "pil"], default="tensorflow")
    args = parser.parse_args()

    report = evaluate(
        checkpoint=args.checkpoint,
        protocol_path=args.protocol,
        manifest_csv=args.manifest_csv,
        dataset_zip=args.dataset_zip,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        output_json=args.output_json,
        output_npz=args.output_npz,
        decoder=args.decoder,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
