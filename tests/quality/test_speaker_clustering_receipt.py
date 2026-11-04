"""Published speaker IDs must reflect actual clustering execution receipts."""
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

import numpy as np

from audio.buffer import AudioBuffer
from core.config import Settings
from sessions.processor import SessionProcessor


class ModelWithReceipt:
    def cluster_with_info(self, embeddings):
        labels = np.array([i // 3 for i in range(len(embeddings))], dtype=int)
        return labels, {
            "strategy_requested": "candidate",
            "strategy_executed": "spherical_candidate",
            "fallback_reason": None,
            "input_embeddings": len(embeddings),
            "cluster_count": len(set(labels)),
        }


class ProcessorReceiptTests(unittest.TestCase):
    def test_four_speakers_survive_published_identity_remap(self):
        with patch.dict(os.environ, {
            "MEETING_SPEAKER_LABEL_REVISABLE_S": "0",
        }):
            proc = SessionProcessor(
                "four-speaker-receipt",
                AudioBuffer(16000, max_seconds=30),
                Settings(), ModelWithReceipt(), asyncio.Queue(),
            )
        for i in range(12):
            speaker = i // 3
            vector = np.zeros(192, dtype=np.float32)
            vector[speaker] = 1.0
            proc.segments.append({
                "id": i + 1,
                "start_sample": i * 24000,
                "end_sample": i * 24000 + 16000,
                "emb": vector, "cluster": -1,
                "speaker_id": -1, "speaker_name": "Unidentified Speaker",
                "text": "test utterance", "language": "en",
                "speaker_uncertain": False,
            })
        proc._recluster()
        self.assertEqual(len({
            x["speaker_id"] for x in proc.segments
            if x["speaker_id"] > 0
        }), 4)
        self.assertEqual(
            proc._last_clustering_receipt["strategy_executed"],
            "spherical_candidate",
        )
        self.assertEqual(proc._last_clustering_receipt["input_embeddings"], 12)
        self.assertEqual(
            proc._last_clustering_receipt["cluster_count"], 4,
        )


if __name__ == "__main__":
    unittest.main()
