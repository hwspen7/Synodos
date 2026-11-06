"""Synthetic anonymous AMI replay comparisons; no private speech required."""
import copy
import unittest

from tools.evaluation.ami.ami_replay_delta import compare_replays


def replay(rows):
    return {
        "source_audio": "E:\\frozen\\TS3003b.wav",
        "source_audio_sha256": "frozen-wav",
        "source_audio_duration_s": 4.0,
        "loops": 1,
        "playback_speed": 1.0,
        "final_confirmed_segments": len(rows),
        "final_cluster_diagnostics": {
            "anonymous_timeline": {
                "truncated": False,
                "total_segments": len(rows),
                "segments": rows,
            }
        },
    }


def segment(a, b, label, emb=True):
    return {"start_ms": a, "end_ms": b,
            "published_speaker_id": label, "embedding_available": emb}


class ReplayDeltaTests(unittest.TestCase):
    def test_reports_lost_captions_and_unidentified_turns_separately(self):
        before = replay([segment(0, 1000, 1), segment(1200, 2000, 2)])
        after = replay([segment(100, 1000, None, False),
                        segment(1200, 1900, 2)])
        x = compare_replays(before, after)
        self.assertEqual(x["changes"]["lost_caption"]["total_seconds"], 0.2)
        self.assertEqual(x["changes"]["lost_positive_speaker"]["total_seconds"], 1.1)
        self.assertEqual(x["changes"]["caption_to_unidentified"]["total_seconds"], 0.9)
        self.assertEqual(x["changes"]["lost_embedding"]["total_seconds"], 1.1)
        self.assertEqual(x["changes"]["gained_caption"]["total_seconds"], 0)
        self.assertEqual(x["changes"]["lost_caption"]["interval_count"], 2)

    def test_renumbering_speakers_is_not_a_regression(self):
        a = replay([segment(0, 3000, 1)])
        b = replay([segment(0, 3000, 99)])
        result = compare_replays(a, b)
        self.assertEqual(result["changes"]["lost_positive_speaker"]["total_seconds"], 0)

    def test_permuted_or_overlapping_segments_are_counted_once(self):
        a = replay([segment(0, 2000, 1), segment(500, 1500, 2)])
        b = replay([segment(0, 2000, 1)])
        result = compare_replays(a, b)
        self.assertEqual(result["changes"]["lost_caption"]["total_seconds"], 0)
        self.assertEqual(result["changes"]["lost_positive_speaker"]["total_seconds"], 0)

    def test_mismatched_recording_and_incomplete_data_fail_closed(self):
        original = replay([segment(0, 1000, 1)])
        other = copy.deepcopy(original)
        other["source_audio_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "paired source_audio_sha256"):
            compare_replays(original, other)
        other = copy.deepcopy(original)
        other["final_cluster_diagnostics"]["anonymous_timeline"]["truncated"] = True
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            compare_replays(original, other)
        other = copy.deepcopy(original)
        other["final_confirmed_segments"] = 2
        with self.assertRaisesRegex(ValueError, "Final transcript"):
            compare_replays(original, other)
        other = copy.deepcopy(original)
        other["final_cluster_diagnostics"]["anonymous_timeline"]["segments"][0]["end_ms"] = 5000
        with self.assertRaisesRegex(ValueError, "timestamps"):
            compare_replays(original, other)


if __name__ == "__main__":
    unittest.main()
