"""Regression tests for timebase, speech support and hallucination safeguards."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from speech.captions.quality_guard import speech_intervals, verify_utterances, voice_ms


def segment(start, end, text, *, words=(), **metadata):
    return SimpleNamespace(start=start, end=end, text=text, words=list(words), **metadata)


def word(start, end, text):
    return SimpleNamespace(start=start, end=end, word=text)


def region(start, end):
    return {"start": round(start * 16000), "end": round(end * 16000)}


class SpeechEvidenceTests(unittest.TestCase):
    def test_no_vad_speech_rejects_output(self):
        output = verify_utterances([segment(0, 2, "Thanks for watching")], [], 4000, "en")
        self.assertEqual(output, ())

    def test_long_hallucination_with_just_half_second_of_voice_is_rejected(self):
        output = verify_utterances([segment(0, 31, "Thanks for watching")],
                                   [region(1, 1.5)], 32000, "en")
        self.assertEqual(output, ())

    def test_language_neutral_real_chinese_sentence_is_preserved(self):
        output = verify_utterances([segment(0.3, 2.4, "\u4eca\u5929\u6211\u4eec\u6d4b\u8bd5\u591a\u8bed\u8a00\u7cfb\u7edf")],
                                   [region(.25, 2.5)], 3000, "zh")
        self.assertEqual(output[0].text, "\u4eca\u5929\u6211\u4eec\u6d4b\u8bd5\u591a\u8bed\u8a00\u7cfb\u7edf")
        self.assertEqual(output[0].language, "zh")

    def test_words_without_speech_support_are_removed(self):
        output = verify_utterances([segment(0, 6, "Hello Thanks for watching",
            words=(word(.4, 1.3, " Hello"), word(4, 4.8, " Thanks"),
                   word(4.8, 5.3, " for"), word(5.3, 6, " watching")))],
            [region(.2, 1.6), region(2, 3), region(3, 4)], 7000, "en")
        # The 6 s transcript has 3.4 s of VAD speech, but only Hello
        # has actual word-level evidence. Missing speech means no word.
        self.assertEqual([u.text for u in output], ["Hello"])
        self.assertEqual((output[0].start_ms, output[0].end_ms), (400, 1300))

    def test_words_split_at_long_silent_gap(self):
        output = verify_utterances([segment(.1, 5, "Hello world",
            words=(word(.3, 1, " Hello"), word(3.7, 4.5, " world")))],
            [region(.1, 2.0), region(3, 5)], 6000, "en")
        self.assertEqual([u.text for u in output], ["Hello", "world"])
        self.assertEqual([u.timestamp_quality for u in output], ["word", "word"])

    def test_zero_duration_is_rejected(self):
        result = verify_utterances([segment(1, 1, "Okay")], [region(0, 2)], 3000, "en")
        self.assertEqual(result, ())

    def test_segment_after_recording_duration_is_rejected(self):
        result = verify_utterances([segment(2, 31, "Ghost")], [region(1, 4)], 5000, "en")
        self.assertEqual(result, ())

    def test_high_no_speech_probability_and_low_logprob_rejected(self):
        result = verify_utterances([segment(.2, 1.2, "Ghost", no_speech_prob=.98,
                                                  avg_logprob=-1.6)],
                                   [region(0, 2)], 3000, "en")
        self.assertEqual(result, ())

    def test_missing_confidence_metadata_does_not_drop_speech(self):
        result = verify_utterances([segment(.2, 1.2, "Valid")],
                                   [region(0, 2)], 3000, "en")
        self.assertEqual(result[0].text, "Valid")

    def test_vad_intervals_are_clipped_and_union_not_double_counted(self):
        intervals = speech_intervals([region(-1, 1.0), region(.5, 2.5),
                                      region(50, 55)], 3000)
        self.assertEqual(intervals, ((0, 2500),))
        self.assertEqual(voice_ms(0, 3000, intervals), 2500)

    def test_invalid_model_timestamps_are_ignored(self):
        result = verify_utterances([segment(float('nan'), 1, "Bad"),
                                   segment(-1, 1, "Bad"), segment(0, 2, "Valid")],
                                  [region(0, 2)], 3000, "en")
        self.assertEqual([x.text for x in result], ["Valid"])

    def test_opt_in_short_reply_requires_strong_acoustic_support(self):
        reply = segment(.20, .32, "yeah")
        evidence = [region(.20, .32)]
        self.assertEqual(
            verify_utterances([reply], evidence, 1000, "en"), ()
        )
        retained = verify_utterances(
            [reply], evidence, 1000, "en",
            allow_short_vad_supported=True,
        )
        self.assertEqual([item.text for item in retained], ["yeah"])
        self.assertEqual(retained[0].timestamp_quality, "segment")

        weak = verify_utterances(
            [reply], [region(.20, .25)], 1000, "en",
            allow_short_vad_supported=True,
        )
        self.assertEqual(weak, ())
        suspicious = segment(
            .20, .32, "imagined reply",
            no_speech_prob=.98, avg_logprob=-1.6,
        )
        self.assertEqual(
            verify_utterances(
                [suspicious], evidence, 1000, "en",
                allow_short_vad_supported=True,
            ), (),
        )

    def test_short_valid_utterance_remains(self):
        result = verify_utterances([segment(.15, .6, "Hi")],
                                   [region(.10, .7)], 1000, "en")
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
