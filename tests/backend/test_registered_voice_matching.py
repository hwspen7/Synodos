"""Regression tests for persistent cross-meeting speaker identity.

Run with: python -m unittest tests.backend.test_registered_voice_matching -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sessions.identity.registered_identity import match_saved_voice, resolved_speaker_identity
from storage.speakers import SpeakerStore


def unit(index: int, size: int = 192) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[index] = 1.0
    return out


def profile(name: str, idx: int, *, duration: float = 8.0,
            short: bool = False, model: str = "camplus") -> dict:
    return {
        "id": name,
        "name": name,
        "vectors": [{
            "emb": unit(idx), "embedding_model": model,
            "duration_s": duration, "short": short,
        }],
    }


class RegisteredVoiceMatchingTests(unittest.TestCase):
    def test_known_voice_recognized_in_later_meeting(self):
        who, score = match_saved_voice(unit(1), [profile("Spencer", 1)])
        self.assertEqual(who, "Spencer")
        self.assertAlmostEqual(score, 1.0)

    def test_unfamiliar_voice_remains_anonymous(self):
        who, _ = match_saved_voice(unit(2), [profile("Spencer", 1)])
        self.assertIsNone(who)

    def test_competing_profiles_require_margin(self):
        same = [profile("Spencer", 1), profile("SomeoneElse", 1)]
        who, _ = match_saved_voice(unit(1), same)
        self.assertIsNone(who)

    def test_reject_short_or_incompatible_embedding(self):
        for bad in [
            profile("Spencer", 1, duration=0.8),
            profile("Spencer", 1, short=True),
            profile("Spencer", 1, model="ecapa"),
        ]:
            self.assertIsNone(match_saved_voice(unit(1), [bad])[0])

    def test_reject_mismatched_shapes_and_nan(self):
        invalid = profile("Spencer", 1)
        invalid["vectors"][0]["emb"] = unit(1, 128)
        self.assertIsNone(match_saved_voice(unit(1), [invalid])[0])
        probe = unit(1)
        probe[3] = float("nan")
        self.assertIsNone(match_saved_voice(probe, [profile("Spencer", 1)])[0])

    def test_registered_name_survives_anonymous_diarization(self):
        record = {"speaker_id": 2, "matched_profile_id": "person-1",
                  "matched_profile_name": "Spencer"}
        self.assertEqual(resolved_speaker_identity(record, 27),
                         (2, "Spencer", "registered_voiceprint"))

    def test_manual_correction_wins(self):
        record = {"speaker_override": True, "speaker_override_id": 3,
                  "speaker_override_name": "Corrected",
                  "speaker_id": 2, "matched_profile_id": "person-1",
                  "matched_profile_name": "Spencer"}
        self.assertEqual(resolved_speaker_identity(record, 27),
                         (3, "Corrected", "manual_override"))

    def test_no_known_identity_uses_anonymous_label(self):
        self.assertEqual(resolved_speaker_identity({"speaker_id": 1}, 5),
                         (5, "Speaker 5", "nemotron3"))

    def test_speaker_library_is_reused_and_not_changed_by_lookup(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SpeakerStore(Path(temp))
            person = store.create("Spencer")
            saved = store.add_vectors(person["id"], [{
                "emb": unit(1), "duration_s": 8.0, "source": "recording",
                "embedding_model": "camplus",
            }])
            self.assertEqual(len(saved), 1)
            records = store.get_auto_matching_profiles()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["name"], "Spencer")
            self.assertEqual(match_saved_voice(unit(1), records)[0], person["id"])
            again = SpeakerStore(Path(temp))
            self.assertEqual(again.list_speakers()[0]["vector_count"], 1)

    def test_short_samples_are_excluded_from_auto_lookup(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SpeakerStore(Path(temp))
            person = store.create("Short")
            store.add_vectors(person["id"], [{
                "emb": unit(1), "duration_s": 0.5,
                "short": True, "source": "recording",
                "embedding_model": "camplus",
            }])
            self.assertEqual(store.get_auto_matching_profiles(), [])


if __name__ == "__main__":
    unittest.main()
