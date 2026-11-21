"""Reproducible dataset inventory, SHA256 hashing, and split-leakage checker for EcoSort-AI."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

ALLOWED_LABELS = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
APPROVED_DECISIONS = {"approved", "approve"}
HOLD_DECISIONS = {"deferred", "defer_ambiguous", "defer_wrapper", "pending", "rejected", "reject", "reject_wrong_label", "reject_bad_image"}
ALLOWED_DECISIONS = APPROVED_DECISIONS | HOLD_DECISIONS
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


def resolve_dataset_path(relative_path: str, root: Path | None = None) -> Path:
    """Reject portable absolute/traversal paths and resolved symlink escapes."""
    portable = relative_path.replace("\\", "/")
    relative = PurePosixPath(portable)
    windows = PureWindowsPath(relative_path)
    if not portable or relative.is_absolute() or windows.drive or ".." in relative.parts:
        raise ValueError(f"unsafe relative path: {relative_path!r}")
    if root is None:
        return Path(relative.as_posix())
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"path escapes dataset root: {relative_path!r}")
    return resolved


def verify_manifest(
    manifest_path: Path,
    dataset_root: Path | None = None,
    protected_manifest: Path | None = None,
) -> int:
    protected_hashes: set[str] = set()
    if protected_manifest is not None:
        with protected_manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "sha256" not in reader.fieldnames:
                print("ERROR: Protected-reference CSV requires a sha256 column.")
                return 1
            for row in reader:
                sha = (row.get("sha256") or "").strip().lower()
                if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                    print("ERROR: Invalid protected-reference SHA256.")
                    return 1
                protected_hashes.add(sha)
        if not protected_hashes:
            print("ERROR: Protected-reference CSV is empty.")
            return 1
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
    decision_counts: Counter[str] = Counter()
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
        full_path = None
        try:
            full_path = resolve_dataset_path(rel_path, dataset_root)
            path_key = str(full_path)
            if path_key in seen_paths:
                errors.append(f"Row {idx}: duplicate relative_path '{rel_path}'")
            seen_paths.add(path_key)
        except ValueError as exc:
            errors.append(f"Row {idx}: {exc}")

        if label not in ALLOWED_LABELS:
            errors.append(f"Row {idx}: invalid label '{label}'")

        if decision not in ALLOWED_DECISIONS:
            errors.append(f"Row {idx}: invalid review_decision '{decision}'")
        else:
            decision_counts[decision] += 1

        if split not in ALLOWED_SPLITS:
            errors.append(f"Row {idx}: invalid split '{split}'")

        if decision in APPROVED_DECISIONS:
            if sha in protected_hashes:
                errors.append(f"Row {idx}: protected-reference collision for '{rel_path}'")
            if not sha or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                errors.append(f"Row {idx}: approved item '{rel_path}' must have a valid 64-char SHA256")
            if not source:
                errors.append(f"Row {idx}: approved item '{rel_path}' missing source_or_license")
            if not group_id:
                errors.append(f"Row {idx}: approved item '{rel_path}' missing group_id")
            if split in {"train", "val", "test"}:
                approved_label_counts[label] += 1
                split_counts[split] += 1
                group_splits[group_id].add(split)
            elif split == "unassigned":
                # Approved candidate awaiting group-freeze before split admission
                approved_label_counts[label] += 1
                split_counts["unassigned"] += 1
        elif decision in HOLD_DECISIONS and split in {"train", "val", "test"}:
            errors.append(f"Row {idx}: {decision} item '{rel_path}' cannot be placed in '{split}' split")

        if sha:
            if sha in seen_hashes:
                errors.append(f"Row {idx}: duplicate SHA256 with '{seen_hashes[sha]}' ({rel_path})")
            seen_hashes[sha] = rel_path

        if dataset_root is not None and full_path is not None:
            if not full_path.is_file():
                errors.append(f"Row {idx}: file missing on disk '{full_path}'")
            elif sha and compute_sha256(full_path) != sha:
                errors.append(f"Row {idx}: SHA256 mismatch for '{rel_path}'")

    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"Group leakage: group '{group_id}' appears across splits {sorted(splits)}")

    print(f"Total manifest rows: {len(rows)}")
    print(f"Decision counts: {dict(decision_counts)}")
    print(f"Approved split counts: {dict(split_counts)}")
    print(f"Approved class counts: {dict(approved_label_counts)}")

    if errors:
        print(f"\nFAILED with {len(errors)} issue(s):")
        for err in errors[:25]:
            print(f"  - {err}")
        return 1

    mode = "metadata and source bytes" if dataset_root is not None else "metadata only"
    print(f"\nPASSED: {mode}; schema, uniqueness, review gating and assigned group splits checked.")
    if not any(split_counts[s] for s in ("train", "val", "test")):
        print("All candidates are unassigned; split-leakage verification awaits split assignment.")
    print("Protected-reference hashes checked." if protected_manifest else "Protected-reference exclusion was not requested.")
    print("Pixel/perceptual independence requires separate review.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EcoSort-AI dataset manifest integrity.")
    parser.add_argument("manifest", type=Path, help="Path to CSV manifest")
    parser.add_argument("--root", type=Path, default=None, help="Optional dataset root to verify files & hashes")
    parser.add_argument("--protected-manifest", type=Path, help="CSV of forbidden reference SHA256 values")
    args = parser.parse_args()
    return verify_manifest(args.manifest, args.root, args.protected_manifest)


if __name__ == "__main__":
    raise SystemExit(main())
