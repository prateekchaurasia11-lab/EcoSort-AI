"""Protect review records from ambiguous selections and failed updates."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Src.dataset_audit import REQUIRED_COLUMNS
from Src.review_helper import apply_batch_decision


class ReviewHelperChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "manifest.csv"
        self.log = self.root / "log.json"
        self.rows = [dict(relative_path="a.jpg", sha256="a" * 64, source_or_license="fixture",
                          label="glass", review_decision="pending", group_id="g1", split="unassigned",
                          reviewer="previous", notes="previous note"),
                     dict(relative_path="b.jpg", sha256="a" * 8 + "b" * 56, source_or_license="fixture",
                          label="paper", review_decision="pending", group_id="g2", split="unassigned",
                          reviewer="previous", notes="keep this")]
        self.write()
        self.log.write_text("[]")

    def write(self):
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS + ["reviewer", "notes"])
            writer.writeheader()
            writer.writerows(self.rows)

    def apply(self, prefixes=None, decision="approve"):
        return apply_batch_decision(self.manifest, self.log, prefixes or ["a" * 9],
                                    decision, "fixture reviewer", "new note")

    def assert_preserved(self, before, log_before):
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual(self.log.read_bytes(), log_before)

    def test_valid_edit_preserves_other_rows_and_notes(self):
        self.assertEqual(self.apply(), 0)
        with self.manifest.open() as handle:
            updated = list(csv.DictReader(handle))
        self.assertEqual(updated[0]["review_decision"], "approve")
        self.assertIn("previous note", updated[0]["notes"])
        self.assertEqual(updated[1], self.rows[1])
        changes = json.loads(self.log.read_text())
        self.assertEqual(changes[0]["before_decision"], "pending")
        self.assertEqual(changes[0]["after_decision"], "approve")

    def test_short_invalid_missing_and_ambiguous_prefixes(self):
        before, log_before = self.manifest.read_bytes(), self.log.read_bytes()
        for prefixes in (["a"], ["z" * 8], ["c" * 8], ["a" * 8], ["a" * 9, "c" * 8]):
            with self.subTest(prefixes=prefixes):
                self.assertEqual(self.apply(prefixes), 1)
                self.assert_preserved(before, log_before)

    def test_proposed_invalid_split_edit_leaves_files_unchanged(self):
        self.rows[0].update(review_decision="approve", split="train")
        self.write()
        before, log_before = self.manifest.read_bytes(), self.log.read_bytes()
        self.assertEqual(self.apply(decision="reject"), 1)
        self.assert_preserved(before, log_before)

    def test_malformed_changelog_preserves_manifest(self):
        self.log.write_text("malformed")
        before, log_before = self.manifest.read_bytes(), self.log.read_bytes()
        self.assertEqual(self.apply(), 1)
        self.assert_preserved(before, log_before)

    def test_log_replace_failure_rolls_back_manifest(self):
        import os
        real_replace = os.replace
        def replace(source, destination):
            if Path(destination) == self.log:
                raise OSError("fixture log write failure")
            return real_replace(source, destination)
        before, log_before = self.manifest.read_bytes(), self.log.read_bytes()
        with patch("Src.review_helper.os.replace", side_effect=replace):
            self.assertEqual(self.apply(), 1)
        self.assert_preserved(before, log_before)

    def test_existing_lock_blocks_update(self):
        lock = self.manifest.with_suffix(".csv.review-lock")
        lock.mkdir()
        before, log_before = self.manifest.read_bytes(), self.log.read_bytes()
        self.assertEqual(self.apply(), 1)
        self.assert_preserved(before, log_before)
        self.assertTrue(lock.exists())

    def test_repeated_same_decision_does_not_duplicate_log(self):
        self.assertEqual(self.apply(), 0)
        before, log_before = self.manifest.read_bytes(), self.log.read_bytes()
        self.assertEqual(self.apply(), 0)
        self.assert_preserved(before, log_before)

    def test_manifest_log_alias_rejected(self):
        before = self.manifest.read_bytes()
        self.assertEqual(apply_batch_decision(self.manifest, self.manifest,
                                             ["a" * 9], "approve", "reviewer", "note"), 1)
        self.assertEqual(self.manifest.read_bytes(), before)
