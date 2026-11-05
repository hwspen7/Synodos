"""Speaker enrollment is validated, auditable and cannot erase earlier identities."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np

from storage.speakers import SpeakerStore


def vector(**overrides):
    return {
        "emb": np.ones(192, dtype=np.float32),
        "duration_s": 3.0,
        "embedding_model": "camplus",
        **overrides,
    }


class SpeakerStoreIntegrityTests(unittest.TestCase):
    def test_empty_batch_does_not_delete_existing_vectors(self):
        with TemporaryDirectory() as base:
            store = SpeakerStore(Path(base))
            person = store.create("A")
            first = store.add_vectors(person["id"], [vector()])
            self.assertEqual(len(first), 1)
            self.assertEqual(store.add_vectors(person["id"], []), [])
            fresh = SpeakerStore(Path(base))
            self.assertEqual(
                len(fresh.get_for_matching([person["id"]])[0]["vectors"]), 1
            )

    def test_all_vectors_are_validated_before_any_are_added(self):
        with TemporaryDirectory() as base:
            store = SpeakerStore(Path(base))
            person = store.create("A")
            with self.assertRaisesRegex(ValueError, "embedding"):
                store.add_vectors(person["id"], [
                    vector(), vector(emb=np.array([1.0, np.nan]))
                ])
            self.assertEqual(
                store.list_speakers()[0]["vector_count"], 0
            )

    def test_verified_provenance_is_not_just_a_boolean(self):
        with TemporaryDirectory() as base:
            store = SpeakerStore(Path(base))
            person = store.create("A")
            with self.assertRaisesRegex(ValueError, "provenance"):
                store.add_vectors(person["id"], [
                    vector(human_verified=True)
                ])
            added = store.add_vectors(person["id"], [
                vector(
                    human_verified=True,
                    verification_source="manual_profile_correction",
                    meeting_id="meeting1", segment_ids=[3],
                ),
            ])
            self.assertEqual(len(added), 1)
            self.assertTrue(added[0]["human_verified"])
            self.assertAlmostEqual(
                float(np.linalg.norm(store.get_for_matching(
                    [person["id"]]
                )[0]["vectors"][0]["emb"])), 1.0, places=6
            )

    def test_failed_save_does_not_leave_partial_vectors_in_memory(self):
        with TemporaryDirectory() as base:
            store = SpeakerStore(Path(base))
            person = store.create("A")
            with patch.object(store, "_save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.add_vectors(person["id"], [vector()])
            self.assertEqual(store.list_speakers()[0]["vector_count"], 0)

    def test_corrupt_library_is_not_silently_replaced_by_empty_database(self):
        with TemporaryDirectory() as base:
            target = Path(base) / "speakers.pkl"
            target.write_bytes(b"corrupt-data")
            with self.assertRaisesRegex(RuntimeError, "refusing to erase"):
                SpeakerStore(Path(base))
            self.assertEqual(target.read_bytes(), b"corrupt-data")


if __name__ == "__main__":
    unittest.main()
