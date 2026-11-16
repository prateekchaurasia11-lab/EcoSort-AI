"""Build the unified 6-class EcoSort-AI manifest from the 16 Nov reviewed checkpoint & local captures."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from Src.dataset_audit import compute_sha256

FIELDNAMES = [
    "relative_path",
    "sha256",
    "source_or_license",
    "label",
    "review_decision",
    "group_id",
    "split",
    "reviewer",
    "notes",
]


def build_manifest(
    checkpoint_csv: Path,
    output_csv: Path,
    local_wrapper_dir: Path | None = None,
) -> int:
    rows_out: list[dict[str, str]] = []
    seen_sha: set[str] = set()

    with checkpoint_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sha = row["image_id"].strip().lower()
            if sha in seen_sha:
                continue
            seen_sha.add(sha)
            rows_out.append(
                {
                    "relative_path": row["relative_file"].strip(),
                    "sha256": sha,
                    "source_or_license": "kaggle_six_class_reviewed_2026-11-16",
                    "label": row["source_label"].strip(),
                    "review_decision": row["decision"].strip(),
                    "group_id": row["source_group_id"].strip(),
                    # Keep unassigned until item/session grouping is verified before split freeze
                    "split": "unassigned",
                    "reviewer": row.get("reviewer", "").strip(),
                    "notes": row.get("notes", "").strip(),
                }
            )

    if local_wrapper_dir and local_wrapper_dir.is_dir():
        for img_path in sorted(local_wrapper_dir.glob("*.jpg")):
            sha = compute_sha256(img_path)
            if sha in seen_sha:
                continue
            seen_sha.add(sha)
            rows_out.append(
                {
                    "relative_path": f"local_wrappers/plastic/{img_path.name}",
                    "sha256": sha,
                    "source_or_license": "local_capture_prince_2026-11",
                    "label": "plastic",
                    "review_decision": "defer_wrapper",
                    "group_id": f"local_wrapper_{img_path.stem}",
                    "split": "unassigned",
                    "reviewer": "Prince Verma (dataset-preparation)",
                    "notes": "Deferred per wrapper-only pause & composite/film material policy.",
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {output_csv}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EcoSort-AI 6-class manifest CSV.")
    parser.add_argument(
        "--checkpoint-csv",
        type=Path,
        default=Path("Data/drive_checkpoint/extracted/review_decisions.csv"),
        help="Path to extracted review_decisions.csv",
    )
    parser.add_argument(
        "--local-wrappers",
        type=Path,
        default=Path(r"C:\Users\asus\github\React\EcoSort_AI\additional_training\plastic"),
        help="Optional local wrapper capture folder (kept defer_wrapper)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Data/ecosort_manifest_2026-11-20.csv"),
        help="Output manifest CSV path (gitignored)",
    )
    args = parser.parse_args()
    return build_manifest(args.checkpoint_csv, args.out, args.local_wrappers)


if __name__ == "__main__":
    raise SystemExit(main())
