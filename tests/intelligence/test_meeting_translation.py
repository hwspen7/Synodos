import unittest
from unittest.mock import Mock, patch

from intelligence import meeting_translation


class MeetingTranslationTests(unittest.TestCase):
    def test_queues_all_saved_rows_and_skips_fully_cached_batches(self):
        rows = [{"id": index, "start_ms": index * 1000, "text": str(index)}
                for index in range(1, 52)]
        cached = [{"segment_id": index, "text": str(index)} for index in range(1, 5)]
        with patch.object(meeting_translation, "store") as store:
            store.meeting_segments.return_value = (7, list(reversed(rows)))
            store.translations.return_value = cached
            result = meeting_translation.queue_meeting("meeting", "zh")
        batches = [call.args[2] for call in store.enqueue.call_args_list]
        self.assertEqual(result["total"], 51)
        self.assertEqual([sid for batch in batches for sid in batch["segment_ids"]],
                         list(range(5, 52)))
        self.assertTrue(all(batch["workspace"] and batch["source_revision"] == 7
                            for batch in batches))
        self.assertTrue(all(call.kwargs["priority"] == 20
                            for call in store.enqueue.call_args_list))

    def test_retry_skips_saved_results_and_translates_entire_source_on_cpu(self):
        source = "A" * 1001 + " B" * 1200 + " final sentence"
        rows = [{"id": 1, "text": "Already translated"}, {"id": 2, "text": source}]
        payload = {"segment_ids": [1, 2], "source_revision": 7}
        model = Mock()
        model.model_for.return_value = "translator"
        model.chat.return_value = "Translated"
        with patch.object(meeting_translation, "store") as store:
            store.translations.return_value = [{"segment_id": 1, "text": "Cached"}]
            result = meeting_translation.translate_batch("meeting", 7, rows, "zh",
                                                         payload, model)
        chunks = [call.args[1].split("<source>\n", 1)[1].rsplit("\n</source>", 1)[0]
                  for call in model.chat.call_args_list]
        self.assertEqual("".join(chunks), source)
        self.assertTrue(all(call.kwargs["cpu_translation"] for call in model.chat.call_args_list))
        self.assertEqual(result["translations"][0], {"segment_id": 1, "text": "Cached"})
        store.save_translation.assert_called_once_with(
            "meeting", 2, source, "zh", "\n".join(["Translated"] * len(chunks)), "translator",
        )

    def test_changed_revision_does_not_translate_outdated_jobs(self):
        model = Mock()
        with self.assertRaisesRegex(ValueError, "transcript changed"):
            meeting_translation.translate_batch("meeting", 8, [], "zh", {
                "segment_ids": [1], "source_revision": 7,
            }, model)
        model.chat.assert_not_called()
