"""Regression tests for opt-in, private live-caption segment audits."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.live_caption_segment_audit import (
    build_segment_audit, capture_segment_audit, write_private_segment_audit,
)


def outside_private_temporary_directory():
    """Create an E-drive project-local fixture outside the private export root.

    A Windows test harness can deliberately redirect TEMP/TMP beneath
    benchmark/fixed195. A default TemporaryDirectory() would then be an
    ALLOWED export destination, invalidating tests that expect rejection.
    Placing this fixture under the sibling benchmark directory makes the
    negative path test independent of the machine's temporary directory.
    """
    benchmark = Path(__file__).resolve().parents[2] / "benchmark"
    benchmark.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(
        prefix="outside-private-audit-", dir=benchmark,
    )


def segment(sid, text, speaker=1, start=0, end=1000, name=None):
    return {
        "id": sid, "text": text, "speaker_id": speaker,
        "speaker_name": name, "start_ms": start, "end_ms": end,
    }


def state():
    return {
        "segment_audit_enabled": True,
        "segment_audit_segments": {},
        "segment_audit_order": [],
        "segment_audit_events": [],
        "segment_order": [],
    }


def summary():
    return {
        "source_audio": "test.wav",
        "source_audio_duration_s": 2,
        "final_cluster_diagnostics": {
            "anonymous_timeline": {
                "truncated": False,
                "segments": [
                    {"segment_id": 10, "embedding_available": False},
                    {"segment_id": 2, "embedding_available": True},
                ],
            },
        },
    }


class LiveCaptionSegmentAuditTests(unittest.TestCase):
    def test_existing_private_output_rejected_before_replay(self):
        from argparse import Namespace
        from tools.live_caption_ws_replay import preflight_private_exports

        root = Path("benchmark/fixed195")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            existing = Path(temp) / "prior_transcript.txt"
            existing.write_text("Prior test evidence", encoding="utf-8")
            args = Namespace(
                transcript_output=existing, segment_audit_output=None,
                output=Path(temp) / "metrics.json", reference_text=None,
            )
            with self.assertRaisesRegex(
                FileExistsError, "No audio has been replayed"
            ):
                preflight_private_exports(args)
            self.assertEqual(
                existing.read_text(encoding="utf-8"),
                "Prior test evidence",
            )
            args.transcript_output = Path(temp) / "new_transcript.txt"
            preflight_private_exports(args)

    def test_export_preflight_detects_report_and_reference_collisions(self):
        from argparse import Namespace
        from tools.live_caption_ws_replay import preflight_private_exports

        root = Path("benchmark/fixed195")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            target = Path(temp) / "audit.json"
            args = Namespace(
                transcript_output=None, segment_audit_output=target,
                output=target, reference_text=None,
            )
            with self.assertRaisesRegex(ValueError, "must not share"):
                preflight_private_exports(args)
            args.output = Path(temp) / "other_metrics.json"
            preflight_private_exports(args)
            args.segment_audit_output = None
            args.transcript_output = Path(temp) / "hypothesis.txt"
            args.reference_text = args.transcript_output
            with self.assertRaisesRegex(ValueError, "must not share"):
                preflight_private_exports(args)

    def test_export_preflight_rejects_paths_outside_ignored_directory(self):
        from argparse import Namespace
        from tools.live_caption_ws_replay import preflight_private_exports

        with outside_private_temporary_directory() as temp:
            self.assertFalse(
                Path(temp).resolve().is_relative_to(
                    Path("benchmark/fixed195").resolve(),
                )
            )
            args = Namespace(
                transcript_output=Path(temp) / "speech.txt",
                segment_audit_output=None, output=None, reference_text=None,
            )
            with self.assertRaisesRegex(ValueError, "gitignored"):
                preflight_private_exports(args)

    def test_real_replay_capture_only_records_when_explicitly_enabled(self):
        from tools.live_caption_ws_replay import capture_event, new_capture

        entry = segment(10, "Alpha", 1)
        for enabled in (False, True):
            stream = new_capture(segment_audit_enabled=enabled)
            capture_event(
                stream, {"type": "transcript", "version": 1,
                         "segments": [entry]}, 2000, sent_samples=32000
            )
            capture_event(
                stream, {"type": "transcript", "version": 2,
                         "segments": [segment(10, "Beta", 2)]},
                25000, sent_samples=400000
            )
            self.assertEqual(
                len(stream["segment_audit_events"]), 2 if enabled else 0
            )
            self.assertEqual(
                len(stream["segment_audit_segments"]), 1 if enabled else 0
            )
            self.assertEqual(stream["confirmed_text_changes"], 1)
            self.assertEqual(stream["speaker_reassignments"], 1)


    def test_disabled_does_not_record_any_content(self):
        sample = state()
        sample["segment_audit_enabled"] = False
        capture_segment_audit(sample, [segment(1, "Private words")], 2000)
        self.assertEqual(sample["segment_audit_events"], [])
        self.assertEqual(sample["segment_audit_segments"], {})
        with self.assertRaisesRegex(ValueError, "not enabled"):
            build_segment_audit(sample, summary())

    def test_final_snapshot_preserves_audio_order_not_id_order(self):
        sample = state()
        incoming = [
            segment(10, "First", 1, 0, 900),
            segment(2, "Second", 2, 900, 1800),
        ]
        capture_segment_audit(sample, incoming, 3000, media_elapsed_ms=2000)
        sample["segment_order"] = [10, 2]
        result = build_segment_audit(sample, summary())
        self.assertEqual(
            [x["segment_id"] for x in result["final_segments"]], [10, 2]
        )
        self.assertEqual(result["segments_without_speaker_embedding"], [10])
        self.assertIn(
            {"segment_id": 10, "flag": "speaker_embedding_missing"},
            result["review_flags"],
        )

    def test_late_speaker_switch_records_exact_changed_segment(self):
        sample = state()
        capture_segment_audit(
            sample, [segment(10, "Hello", 1)], 1000, media_elapsed_ms=1000
        )
        capture_segment_audit(
            sample, [segment(10, "Hello", 2)], 25000,
            media_elapsed_ms=25000,
        )
        sample["segment_order"] = [10]
        result = build_segment_audit(sample, summary())
        event = result["revision_events"][0]
        self.assertEqual(event["kind"], "speaker_reassignment")
        self.assertEqual(event["transition"], "labeled_to_different_labeled")
        self.assertEqual(event["segment_age_s"], 24)
        self.assertEqual((event["from_speaker_id"], event["to_speaker_id"]), (1, 2))
        self.assertIn(
            {"segment_id": 10, "flag": "late_labeled_speaker_change"},
            result["review_flags"],
        )

    def test_unidentified_to_labeled_is_not_a_late_labeled_switch(self):
        sample = state()
        capture_segment_audit(sample, [segment(10, "Hello", None)], 1000)
        capture_segment_audit(
            sample, [segment(10, "Hello", 1)], 25000,
            media_elapsed_ms=25000,
        )
        sample["segment_order"] = [10]
        result = build_segment_audit(sample, summary())
        self.assertEqual(
            result["revision_events"][0]["transition"], "unidentified_to_labeled"
        )
        self.assertNotIn(
            {"segment_id": 10, "flag": "late_labeled_speaker_change"},
            result["review_flags"],
        )

    def test_text_changes_are_inspectable_and_disappearances_recorded(self):
        sample = state()
        capture_segment_audit(
            sample, [segment(10, "Old"), segment(2, "Other")], 1000
        )
        capture_segment_audit(sample, [segment(10, "Corrected")], 2000)
        sample["segment_order"] = [10]
        result = build_segment_audit(sample, summary())
        kinds = {e["kind"] for e in result["revision_events"]}
        self.assertEqual(
            kinds, {"confirmed_text_revision", "confirmed_segment_disappearance"}
        )
        edit = next(e for e in result["revision_events"]
                    if e["kind"] == "confirmed_text_revision")
        self.assertEqual((edit["previous_text"], edit["revised_text"]),
                         ("Old", "Corrected"))

    def test_mismatched_snapshot_rejected(self):
        sample = state()
        capture_segment_audit(sample, [segment(10, "Hi")], 2000)
        sample["segment_order"] = [2]
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_segment_audit(sample, summary())

    def test_invalid_export_path_rejected(self):
        with outside_private_temporary_directory() as temp:
            self.assertFalse(
                Path(temp).resolve().is_relative_to(
                    Path("benchmark/fixed195").resolve(),
                )
            )
            with self.assertRaisesRegex(ValueError, "benchmark/fixed195"):
                write_private_segment_audit(Path(temp) / "outside.json", {})

    def test_export_is_exclusive_and_within_gitignored_directory(self):
        root = Path("benchmark/fixed195")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            destination = Path(temp) / "review.json"
            write_private_segment_audit(destination, {"segments": []})
            self.assertIn('"segments"', destination.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_private_segment_audit(destination, {"segments": []})


if __name__ == "__main__":
    unittest.main()
