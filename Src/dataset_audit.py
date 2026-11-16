"""Reproducible dataset inventory, SHA256 hashing, and split-leakage checker for EcoSort-AI."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ALLOWED_LABELS = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
ALLOWED_DECISIONS = {"approved", "deferred", "pending", "rejected"}
ALLOWED_SPLITS = {"train", "val", "test", "unassigned"}
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

    if not rows:
        print("ERROR: Manifest is empty.")
        return 1

    split_counts: Counter[str] = Counter()
    approved_label_counts: Counter[str] = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    seen_hashes: dict[str, str] = {}
    seen_paths: set[str] = set()
    errors: list[str] = []

    for idx, row in enumerate(rows, start=2):
        rel_path = (row.get("relative_path") or "").strip()
        sha = (row.get("sha256") or "").strip().lower()
        source = (row.get("source_or_license") or "").strip()
        label = (row.get("label") or "").strip()
        decision = (row.get("review_decision") or "").strip().lower()
        group_id = (row.get("group_id") or "").strip()
        split = (row.get("split") or "").strip().lower()

        if not rel_path:
            errors.append(f"Row {idx}: relative_path cannot be empty")
        elif rel_path in seen_paths:
            errors.append(f"Row {idx}: duplicate relative_path '{rel_path}'")
        seen_paths.add(rel_path)

        if label not in ALLOWED_LABELS:
            errors.append(f"Row {idx}: invalid label '{label}'")

        if decision not in ALLOWED_DECISIONS:
            errors.append(f"Row {idx}: invalid review_decision '{decision}'")

        if split not in ALLOWED_SPLITS:
            errors.append(f"Row {idx}: invalid split '{split}'")

        if decision == "approved":
            if not sha or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                errors.append(f"Row {idx}: approved item '{rel_path}' must have a valid 64-char SHA256")
            if not source:
                errors.append(f"Row {idx}: approved item '{rel_path}' missing source_or_license")
            if not group_id:
                errors.append(f"Row {idx}: approved item '{rel_path}' missing group_id")
            if split not in {"train", "val", "test"}:
                errors.append(f"Row {idx}: approved item '{rel_path}' must be assigned to train, val, or test")
            else:
                approved_label_counts[label] += 1
                split_counts[split] += 1
                if group_id:
                    group_splits[group_id].add(split)
        elif decision in {"pending", "deferred", "rejected"} and split in {"train", "val", "test"}:
            errors.append(f"Row {idx}: {decision} item '{rel_path}' cannot be placed in '{split}' split")

        if sha:
            if sha in seen_hashes:
                errors.append(f"Row {idx}: duplicate SHA256 with '{seen_hashes[sha]}' ({rel_path})")
            seen_hashes[sha] = rel_path

        if dataset_root is not None and rel_path:
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
