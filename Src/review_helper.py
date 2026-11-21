"""Audit-logged CLI review helper for updating pending items in the EcoSort-AI manifest."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from Src.dataset_audit import verify_manifest

VALID_NEW_DECISIONS = {"approve", "defer_ambiguous", "defer_wrapper", "reject"}


def summarize(manifest_path: Path) -> int:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_class: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    for r in rows:
        lbl = r["label"]
        dec = r["review_decision"]
        by_class.setdefault(lbl, Counter())[dec] += 1
        totals[dec] += 1

    print(f"Manifest: {manifest_path} ({len(rows)} rows)")
    print(f"Totals: {dict(totals)}\n")
    print(f"{'Class':<12} {'approve':>8} {'defer_amb':>10} {'defer_wrap':>11} {'pending':>9} {'reject':>8}")
    print("-" * 62)
    for cls in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
        c = by_class.get(cls, Counter())
        print(
            f"{cls:<12} {c['approve']:>8} {c['defer_ambiguous']:>10} "
            f"{c['defer_wrapper']:>11} {c['pending']:>9} {c['reject']:>8}"
        )
    return 0


def apply_batch_decision(
    manifest_path: Path,
    changelog_path: Path,
    sha256_prefixes: list[str],
    new_decision: str,
    reviewer: str,
    note: str,
) -> int:
    """Serialize edits and preserve both files on validation/write failures."""
    manifest_path = manifest_path.resolve()
    changelog_path = changelog_path.resolve()
    if manifest_path == changelog_path:
        print("ERROR: Manifest and changelog must be different files.")
        return 1
    lock = manifest_path.with_suffix(manifest_path.suffix + ".review-lock")
    try:
        lock.mkdir()
    except FileExistsError:
        print(f"ERROR: Another edit or interrupted edit holds {lock}. Inspect before removing it.")
        return 1
    try:
        return _apply_batch_decision(manifest_path, changelog_path, sha256_prefixes,
                                     new_decision, reviewer, note)
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: Review edit failed: {exc}")
        return 1
    finally:
        lock.rmdir()


def _apply_batch_decision(
    manifest_path: Path,
    changelog_path: Path,
    sha256_prefixes: list[str],
    new_decision: str,
    reviewer: str,
    note: str,
) -> int:
    if new_decision not in VALID_NEW_DECISIONS:
        print(f"ERROR: decision must be one of {sorted(VALID_NEW_DECISIONS)}")
        return 1

    if not reviewer.strip():
        print("ERROR: A reviewer is required.")
        return 1
    if verify_manifest(manifest_path):
        return 1

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    prefixes = [p.strip().lower() for p in sha256_prefixes if p.strip()]
    if not prefixes or any(len(p) < 8 or len(p) > 64 or any(c not in "0123456789abcdef" for c in p) for p in prefixes):
        print("ERROR: Supply hexadecimal SHA256 prefixes between 8 and 64 characters.")
        return 1
    selected: set[str] = set()
    for prefix in prefixes:
        matches = [row["sha256"].lower() for row in rows if row["sha256"].lower().startswith(prefix)]
        if len(matches) != 1:
            print(f"ERROR: Prefix {prefix!r} matches {len(matches)} rows; supply a unique prefix.")
            return 1
        selected.add(matches[0])
    existing: list[dict] = []
    if changelog_path.is_file():
        existing = json.loads(changelog_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list) or any(not isinstance(item, dict) for item in existing):
            raise ValueError("Changelog must be a JSON array of review records.")
    for column in ("reviewer", "notes"):
        if column not in fieldnames:
            fieldnames.append(column)
    changes: list[dict[str, str]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for row in rows:
        sha = row["sha256"].lower()
        if sha not in selected:
            continue
        old_dec = row["review_decision"]
        if old_dec == new_decision:
            continue
        row["review_decision"] = new_decision
        row["reviewer"] = reviewer
        previous_note = row.get("notes", "").strip()
        row["notes"] = "\n".join(part for part in (previous_note, f"{note} ({now_iso})".strip()) if part)
        changes.append(
            {
                "timestamp": now_iso,
                "sha256": sha,
                "relative_path": row["relative_path"],
                "label": row["label"],
                "before_decision": old_dec,
                "after_decision": new_decision,
                "reviewer": reviewer,
                "note": note,
            }
        )

    if not changes:
        print("No matching rows changed.")
        return 0

    existing.extend(changes)
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    # Staging beside each destination keeps replacements on the same filesystem.
    # Retain backups until both replacements complete so caught failures roll back.
    with tempfile.TemporaryDirectory(prefix=".review-", dir=manifest_path.parent) as manifest_tmp, \
         tempfile.TemporaryDirectory(prefix=".review-log-", dir=changelog_path.parent) as log_tmp:
        staged_manifest = Path(manifest_tmp) / "candidate.csv"
        with staged_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if verify_manifest(staged_manifest):
            print("ERROR: Proposed decisions failed validation; saved files were preserved.")
            return 1
        staged_log = Path(log_tmp) / "candidate.json"
        staged_log.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        backup = Path(manifest_tmp) / "original.csv"
        backup.write_bytes(manifest_path.read_bytes())
        os.replace(staged_manifest, manifest_path)
        try:
            os.replace(staged_log, changelog_path)
        except OSError:
            os.replace(backup, manifest_path)
            raise

    print(f"Updated {len(changes)} item(s) -> '{new_decision}' (logged to {changelog_path})")
    return 0


def list_pending(manifest_path: Path, label: str | None, limit: int) -> int:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            r
            for r in csv.DictReader(handle)
            if r["review_decision"] == "pending" and (not label or r["label"] == label)
        ]
    print(f"Matching pending rows: {len(rows)} (showing first {min(limit, len(rows))})\n")
    for r in rows[:limit]:
        print(f"  {r['sha256'][:12]}  {r['label']:<10}  {r['group_id']:<28}  {r['relative_path']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="EcoSort-AI manifest review progress & audit helper.")
    parser.add_argument("--manifest", type=Path, default=Path("Data/ecosort_manifest_2026-11-20.csv"))
    parser.add_argument("--changelog", type=Path, default=Path("Data/review_changes_local.json"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="Show class-by-class review counts")

    p_list = sub.add_parser("list-pending", help="List pending items by class")
    p_list.add_argument("--label", default=None, help="Filter by class name")
    p_list.add_argument("--limit", type=int, default=20)

    p_set = sub.add_parser("decide", help="Update decision by SHA256 prefix with audit log")
    p_set.add_argument("--decision", required=True, choices=sorted(VALID_NEW_DECISIONS))
    p_set.add_argument("--reviewer", default="Prince Verma (manual visual review)")
    p_set.add_argument("--note", default="Manual visual inspection")
    p_set.add_argument("sha_prefixes", nargs="+", help="One or more SHA256 prefixes (min 8 chars)")

    args = parser.parse_args()
    if args.cmd == "summary":
        return summarize(args.manifest)
    if args.cmd == "list-pending":
        return list_pending(args.manifest, args.label, args.limit)
    if args.cmd == "decide":
        return apply_batch_decision(
            args.manifest,
            args.changelog,
            args.sha_prefixes,
            args.decision,
            args.reviewer,
            args.note,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
