"""Reproducible dataset inventory, SHA256 hashing, and split-leakage checker for EcoSort-AI."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED_LABELS = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
REQUIRED_COLUMNS = [
    "relative_path",
    "sha256",
    "source_or_license",
    "label",
    "review_decision",
    "group_id",
    "split",
]


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path: Path, dataset_root: Path | None = None) -> int:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or any(col not in reader.fieldnames for col in REQUIRED_COLUMNS):
            print(f"ERROR: Missing required columns. Expected: {REQUIRED_COLUMNS}")
            return 1

        rows = list(reader)

    split_counts: Counter[str] = Counter()
    approved_label_counts: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    seen_hashes: dict[str, str] = {}
    errors: list[str] = []

    for idx, row in enumerate(rows, start=2):
        rel_path = row["relative_path"].strip()
        sha = row["sha256"].strip().lower()
        label = row["label"].strip()
        decision = row["review_decision"].strip().lower()
        group_id = row["group_id"].strip()
        split = row["split"].strip().lower()

        if label not in ALLOWED_LABELS:
            errors.append(f"Row {idx}: invalid label '{label}'")

        if decision == "approved" and split in {"train", "val", "test"}:
            approved_label_counts[label] += 1
            split_counts[split] += 1
            if group_id:
                group_splits[group_id].add(split)
        elif decision in {"pending", "deferred"} and split == "train":
            errors.append(f"Row {idx}: {decision} item '{rel_path}' cannot be in train split")

        if sha:
            if sha in seen_hashes:
                errors.append(f"Row {idx}: duplicate SHA256 with '{seen_hashes[sha]}' ({rel_path})")
            seen_hashes[sha] = rel_path

        if dataset_root is not None:
            full_path = dataset_root / rel_path
            if not full_path.is_file():
                errors.append(f"Row {idx}: file missing on disk '{full_path}'")
            elif sha and compute_sha256(full_path) != sha:
                errors.append(f"Row {idx}: SHA256 mismatch for '{rel_path}'")

    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"Group leakage: group '{group_id}' appears across splits {sorted(splits)}")

    print(f"Total manifest rows: {len(rows)}")
    print(f"Approved split counts: {dict(split_counts)}")
    print(f"Approved class counts: {dict(approved_label_counts)}")

    if errors:
        print(f"\nFAILED with {len(errors)} issue(s):")
        for err in errors[:25]:
            print(f"  - {err}")
        return 1

    print("\nPASSED: Manifest schema, SHA256 uniqueness, review gating, and group split integrity verified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EcoSort-AI dataset manifest integrity.")
    parser.add_argument("manifest", type=Path, help="Path to CSV manifest")
    parser.add_argument("--root", type=Path, default=None, help="Optional dataset root to verify files & hashes")
    args = parser.parse_args()
    return verify_manifest(args.manifest, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
