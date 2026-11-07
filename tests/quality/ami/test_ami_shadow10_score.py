"""Offline AMI reference-audit regression tests. No real audio or GPU required."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.evaluation.ami.ami_shadow10_prepare import SELECTED, checksum
from tools.evaluation.ami.ami_shadow10_score import (
    score_case, reference_union_overlap, unidentified_segments,
    missed_speaker_time_breakdown, reference_speech_without_captions,
    reference_speaker_coverage, reference_speaker_context_breakdown,
)


def prepare(root: Path, *, diagnostics: bool = True):
    meeting, offset = "TS3003b", dict(SELECTED)["TS3003b"]
    (root / "cases").mkdir(parents=True)
    (root / "audio").mkdir(parents=True)
    (root / "rttm").mkdir(parents=True)
    source = root / "audio" / "ami-TS3003b-1500-1740.wav"
    source.write_bytes(b"synthetic test only; load_wav mocked")
    ref = [
        {"start_ms": i * 12500, "end_ms": (i + 1) * 12500, "speaker": "A"}
        for i in range(8)
    ] + [
        {"start_ms": 100000 + i * 10000,
         "end_ms": 100000 + (i + 1) * 10000, "speaker": "B"}
        for i in range(8)
    ]
    rttm = root / "rttm" / (meeting + ".rttm")
    rttm.write_text(
        "".join(
            f"SPEAKER {meeting} 1 {offset + turn['start_ms'] / 1000:.3f} "
            f"{(turn['end_ms'] - turn['start_ms']) / 1000:.3f} "
            f"<NA> <NA> {turn['speaker']} <NA> <NA>\n"
            for turn in ref
        ),
        encoding="utf-8",
    )
    case = {
        "id": f"ami-{meeting}-{offset}",
        "source_meeting": meeting, "start_seconds": offset,
        "end_seconds": offset + 240,
        "audio": str(source.relative_to(root)).replace("\\", "/"),
        "sha256": checksum(source), "reference_turns": ref,
        "reference_source_sha256": checksum(rttm),
        "reference_kind": "human-derived only_words",
    }
    (root / "cases" / (meeting + ".json")).write_text(
        json.dumps(case), encoding="utf-8",
    )
    timeline = {
        "truncated": False,
        "total_segments": 4,
        "segments": [
            {"segment_id": 1, "start_ms": 0, "end_ms": 100000,
             "published_speaker_id": 5, "embedding_available": True},
            {"segment_id": 2, "start_ms": 100000, "end_ms": 120000,
             "published_speaker_id": -1, "speaker_uncertain": True},
            {"segment_id": 3, "start_ms": 120000, "end_ms": 180000,
             "published_speaker_id": 8, "embedding_available": True},
            {"segment_id": 4, "start_ms": 200000, "end_ms": 210000,
             "published_speaker_id": None, "embedding_available": False},
        ],
    }
    replay = {
        "source_audio": str(source),
        "source_audio_sha256": case["sha256"],
        "source_audio_duration_s": 240.0,
        "loops": 1, "playback_speed": 1.0,
        "delivery_matched": True, "server_closed": True,
        "errors": [], "flags": [],
        "final_confirmed_segments": 4,
        "final_unidentified_segment_count": 2,
        "final_cluster_diagnostics": (
            {"anonymous_timeline": timeline} if diagnostics else None
        ),
        "speaker_shadow_diagnostics": {"submitted": 5, "completed": 5},
    }
    replay_path = root / "replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    return replay_path


class AmiShadowScoreTests(unittest.TestCase):
    def test_union_of_overlapping_reference_intervals_not_double_counted(self):
        reference = [
            {"start_ms": 0, "end_ms": 1000, "speaker": "A"},
            {"start_ms": 500, "end_ms": 1500, "speaker": "B"},
            {"start_ms": 2000, "end_ms": 2800, "speaker": "A"},
        ]
        self.assertEqual(reference_union_overlap(250, 2300, reference), 1550)
        self.assertEqual(reference_union_overlap(1500, 1900, reference), 0)

    @patch("tools.evaluation.ami.ami_shadow10_score.load_wav")
    def test_der_and_unknown_segments_are_reference_backed(self, fake_audio):
        fake_audio.return_value = np.zeros(240 * 16000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            replay = prepare(root)
            result = score_case(root, "TS3003b", replay)
            self.assertAlmostEqual(result["metrics"]["der"], 20 / 180, places=4)
            self.assertEqual(result["metrics"]["reference_speaker_count"], 2)
            self.assertEqual(result["metrics"]["predicted_speaker_count"], 2)
            self.assertEqual(result["unidentified"]["count"], 2)
            self.assertEqual(result["unidentified"]["with_reference_speech"], 1)
            self.assertEqual(result["unidentified"]["outside_reference_speech"], 1)
            per_voice = result["reference_speaker_coverage"]["speakers"]
            self.assertEqual([x["reference_speaker"] for x in per_voice], ["A", "B"])
            contexts = result["reference_speaker_context"]["speakers"]
            self.assertEqual(contexts[0]["solo"]["reference_speech_seconds"], 100.0)
            self.assertEqual(contexts[1]["solo"]["reference_speech_seconds"], 80.0)
            self.assertEqual(contexts[1]["overlap"]["reference_speech_seconds"], 0.0)
            self.assertEqual(per_voice[0]["reference_speech_seconds"], 100.0)
            self.assertEqual(per_voice[0]["positive_caption_seconds"], 100.0)
            self.assertEqual(per_voice[1]["reference_speech_seconds"], 80.0)
            self.assertEqual(per_voice[1]["unknown_only_caption_seconds"], 20.0)
            self.assertEqual(per_voice[1]["positive_caption_seconds"], 60.0)
            self.assertEqual(result["missed_speaker_time_breakdown"]["unidentified_transcript_s"], 20.0)
            self.assertEqual(result["missed_speaker_time_breakdown"]["total_missed_speaker_seconds"], 20.0)
            self.assertEqual(
                result["unidentified"]["intervals"][0]["reference_speech_overlap_ms"],
                20000,
            )
            self.assertEqual(
                result["unidentified"]["intervals"][1]["reference_speech_overlap_ms"],
                0,
            )
            self.assertNotIn("speaker_mapping", result["metrics"])
            self.assertNotIn('"text":', json.dumps(result))

    def test_locate_reference_speech_gaps_without_fabricating_words(self):
        reference = [{"start_ms": 0, "end_ms": 3000, "speaker": "A"}]
        timeline = {"segments": [
            {"start_ms": 0, "end_ms": 1000, "published_speaker_id": 1},
            {"start_ms": 1700, "end_ms": 3000, "published_speaker_id": -1},
        ]}
        gaps = reference_speech_without_captions(
            timeline, reference, 3000,
        )
        self.assertEqual(gaps["gap_count"], 1)
        self.assertEqual(
            gaps["reference_speech_without_captions_seconds"], 0.7
        )
        self.assertEqual(gaps["longest_gaps"][0], {
            "start_ms": 1000, "end_ms": 1700, "duration_ms": 700,
        })
        self.assertTrue(gaps["requires_audio_review"])

    def test_per_speaker_coverage_preserves_overlap_and_missing_emb_status(self):
        timeline = {"segments": [
            {"start_ms": 0, "end_ms": 1000, "published_speaker_id": 5,
             "embedding_available": True},
            {"start_ms": 1000, "end_ms": 1800, "published_speaker_id": None,
             "embedding_available": False},
            {"start_ms": 1800, "end_ms": 2400, "published_speaker_id": 7,
             "embedding_available": False},
            {"start_ms": 2600, "end_ms": 3000, "published_speaker_id": 7,
             "embedding_available": True},
        ]}
        reference = [
            {"start_ms": 0, "end_ms": 3000, "speaker": "A"},
            {"start_ms": 1000, "end_ms": 2000, "speaker": "B"},
            {"start_ms": 2200, "end_ms": 2800, "speaker": "B"},
        ]
        result = reference_speaker_coverage(timeline, reference, 3000)
        self.assertEqual(result["reference_speaker_count"], 2)
        a, b = result["speakers"]
        self.assertEqual(a["reference_speech_seconds"], 3.0)
        self.assertEqual(a["without_caption_seconds"], 0.2)
        self.assertEqual(a["unknown_only_caption_seconds"], 0.8)
        self.assertEqual(a["positive_caption_seconds"], 2.0)
        self.assertEqual(a["caption_without_embedding_seconds"], 1.4)
        self.assertEqual(a["caption_with_embedding_seconds"], 1.4)
        self.assertEqual(a["overlap_reference_seconds"], 1.6)
        self.assertEqual(b["reference_speech_seconds"], 1.6)
        self.assertEqual(b["annotated_intervals"], 2)
        self.assertEqual(b["short_intervals_under_1_2s"], 2)
        self.assertEqual(b["without_caption_seconds"], 0.2)
        self.assertEqual(b["unknown_only_caption_seconds"], 0.8)
        self.assertEqual(b["positive_caption_seconds"], 0.6)
        self.assertEqual(
            b["positive_label_contacts"],
            [{"published_label": 7, "temporal_overlap_seconds": 0.6}],
        )
        self.assertNotIn("person_name", json.dumps(result))
        with self.assertRaisesRegex(ValueError, "100ms"):
            reference_speaker_coverage(timeline, reference, 3000,
                                       frame_step_ms=50)

    def test_context_breakdown_keeps_solo_and_overlap_evidence_separate(self):
        reference = [
            {"start_ms": 0, "end_ms": 4000, "speaker": "A"},
            {"start_ms": 1000, "end_ms": 2000, "speaker": "B"},
            {"start_ms": 3000, "end_ms": 6000, "speaker": "B"},
        ]
        timeline = {"segments": [
            {"start_ms": 0, "end_ms": 2000,
             "published_speaker_id": 1, "embedding_available": True},
            {"start_ms": 3000, "end_ms": 4000,
             "published_speaker_id": -1, "embedding_available": False},
            {"start_ms": 4000, "end_ms": 5000,
             "published_speaker_id": 2, "embedding_available": False},
            {"start_ms": 5000, "end_ms": 6000,
             "published_speaker_id": 2, "embedding_available": True},
        ]}
        data = reference_speaker_context_breakdown(
            timeline, reference, 6000,
        )
        a, b = data["speakers"]
        self.assertEqual([x["reference_speaker"] for x in (a, b)],
                         ["A", "B"])
        self.assertEqual(a["solo"]["reference_speech_seconds"], 2.0)
        self.assertEqual(a["solo"]["without_caption_seconds"], 1.0)
        self.assertEqual(a["overlap"]["reference_speech_seconds"], 2.0)
        self.assertEqual(a["overlap"]["unknown_only_caption_seconds"], 1.0)
        self.assertEqual(a["overlap"]["positive_label_contacts"], [
            {"published_label": 1, "temporal_overlap_seconds": 1.0},
        ])
        self.assertEqual(b["solo"]["reference_speech_seconds"], 2.0)
        self.assertEqual(b["solo"]["caption_without_embedding_seconds"], 1.0)
        self.assertEqual(b["solo"]["caption_with_embedding_seconds"], 1.0)
        self.assertEqual(b["overlap"]["reference_speech_seconds"], 2.0)
        self.assertEqual(b["longest_solo_stretches"], [{
            "start_ms": 4000, "end_ms": 6000, "duration_ms": 2000,
        }])
        self.assertEqual(a["longest_solo_stretches"], [
            {"start_ms": 0, "end_ms": 1000, "duration_ms": 1000},
            {"start_ms": 2000, "end_ms": 3000, "duration_ms": 1000},
        ])
        self.assertNotIn("utterance", json.dumps(data))
        with self.assertRaisesRegex(ValueError, "100ms"):
            reference_speaker_context_breakdown(
                timeline, reference, 6000, frame_step_ms=50,
            )

    def test_missed_speaker_time_breakdown_separates_three_causes(self):
        timeline = {"segments": [
            {"start_ms": 0, "end_ms": 1000, "published_speaker_id": -1},
            {"start_ms": 1000, "end_ms": 2000, "published_speaker_id": 7},
        ]}
        reference = [
            {"start_ms": 0, "end_ms": 2000, "speaker": "A"},
            {"start_ms": 1000, "end_ms": 2000, "speaker": "B"},
            {"start_ms": 2000, "end_ms": 3000, "speaker": "B"},
        ]
        result = missed_speaker_time_breakdown(timeline, reference, 3000)
        self.assertEqual(result["unidentified_transcript_s"], 1.0)
        self.assertEqual(result["simultaneous_speaker_shortfall_s"], 1.0)
        self.assertEqual(result["without_transcript_s"], 1.0)
        self.assertEqual(result["total_missed_speaker_seconds"], 3.0)
        self.assertEqual(result["reference_speaker_seconds"], 4.0)

    @patch("tools.evaluation.ami.ami_shadow10_score.load_wav")
    def test_missing_diagnostics_requires_new_replay(self, fake_audio):
        fake_audio.return_value = np.zeros(240 * 16000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            replay = prepare(root, diagnostics=False)
            with self.assertRaisesRegex(ValueError, "MEETING_DIARIZATION_DIAGNOSTICS=1"):
                score_case(root, "TS3003b", replay)

    @patch("tools.evaluation.ami.ami_shadow10_score.load_wav")
    def test_wrong_audio_hash_rejected(self, fake_audio):
        fake_audio.return_value = np.zeros(240 * 16000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            replay_path = prepare(root)
            data = json.loads(replay_path.read_text())
            data["source_audio_sha256"] = "0" * 64
            replay_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "frozen AMI WAV"):
                score_case(root, "TS3003b", replay_path)

    @patch("tools.evaluation.ami.ami_shadow10_score.load_wav")
    def test_tampered_reference_rttm_is_rejected(self, fake_audio):
        fake_audio.return_value = np.zeros(240 * 16000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            replay_path = prepare(root)
            rttm = root / "rttm" / "TS3003b.rttm"
            rttm.write_text(rttm.read_text() + "tampered\n")
            with self.assertRaisesRegex(ValueError, "RTTM"):
                score_case(root, "TS3003b", replay_path)

    @patch("tools.evaluation.ami.ami_shadow10_score.load_wav")
    def test_truncated_timeline_never_scored(self, fake_audio):
        fake_audio.return_value = np.zeros(240 * 16000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            replay_path = prepare(root)
            data = json.loads(replay_path.read_text())
            data["final_cluster_diagnostics"]["anonymous_timeline"]["truncated"] = True
            replay_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "timeline"):
                score_case(root, "TS3003b", replay_path)

    @patch("tools.evaluation.ami.ami_shadow10_score.load_wav")
    def test_unidentified_count_mismatch_is_rejected(self, fake_audio):
        fake_audio.return_value = np.zeros(240 * 16000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            replay_path = prepare(root)
            data = json.loads(replay_path.read_text())
            data["final_unidentified_segment_count"] = 18
            replay_path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Unidentified timeline"):
                score_case(root, "TS3003b", replay_path)


if __name__ == "__main__":
    unittest.main()
