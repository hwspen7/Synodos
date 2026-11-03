"""Evidence-preserving conversational splitting for Whisper word timestamps."""
import unittest

from speech.contracts import Utterance, Word
from speech.utterance_units import split_utterance, split_utterances


def utterance(pieces, *, text=None, language="en"):
    words = []
    clock = 0
    for token, duration, gap in pieces:
        start = clock
        clock += duration
        words.append(Word(token, start, clock))
        clock += gap
    value = " ".join(w.text.strip() for w in words) if text is None else text
    return Utterance(value, words[0].start_ms, words[-1].end_ms, language, tuple(words), "word")


class ConversationalUnitsTests(unittest.TestCase):
    def test_real_user_complaint_becomes_short_timed_utterances(self):
        original = utterance([
            ("Really,", 400, 50), ("what", 240, 25), ("a", 130, 25),
            ("coincidence.", 650, 180),
            ("We", 200, 20), ("laughed", 350, 20), ("a", 140, 20),
            ("lot.", 350, 150),
            ("Absolutely.", 650, 400),
            ("Hey,", 230, 30), ("how", 180, 25), ("are", 180, 25),
            ("you?", 300, 300),
            ("Hey,", 200, 25), ("what's", 240, 25), ("man?", 300, 200),
            ("How", 180, 25), ("are", 160, 25), ("you", 170, 25),
            ("doing?", 280, 0),
        ])
        lines = split_utterance(original)
        self.assertEqual(
            [u.text for u in lines],
            [
                "Really, what a coincidence.", "We laughed a lot.",
                "Absolutely.", "Hey, how are you?", "Hey, what's man?",
                "How are you doing?",
            ],
        )
        self.assertTrue(all(u.start_ms < u.end_ms for u in lines))
        self.assertEqual(lines[0].start_ms, original.start_ms)
        self.assertEqual(lines[-1].end_ms, original.end_ms)
        self.assertEqual([word.text for unit in lines for word in unit.words],
                         [word.text for word in original.words])

    def test_silent_gap_creates_boundary_without_punctuation(self):
        original = utterance([
            ("Hello", 320, 1200), ("everyone", 360, 0),
        ])
        lines = split_utterance(original)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].text, "Hello")
        self.assertEqual(lines[1].text, "everyone")

    def test_abbreviations_do_not_force_splits(self):
        original = utterance([
            ("Dr.", 360, 20), ("Smith", 380, 20), ("arrived.", 600, 120),
            ("Hello.", 300, 0),
        ])
        lines = split_utterance(original)
        self.assertEqual([u.text for u in lines], ["Dr. Smith arrived.", "Hello."])

    def test_no_alignment_does_not_fabricate_timecodes(self):
        original = Utterance("Hi. How are you?", 0, 2000, "en")
        self.assertEqual(split_utterance(original), (original,))

    def test_mismatched_words_do_not_fabricate_text(self):
        original = Utterance("A completely different sentence.", 0, 2000, "en",
                             (Word("Hello.", 0, 600), Word("World.", 700, 1200)))
        self.assertEqual(split_utterance(original), (original,))

    def test_handles_chinese_punctuation_with_real_word_times(self):
        original = utterance([
            ("你好。", 500, 100), ("最近", 350, 20), ("怎么样？", 500, 0),
        ], text="你好。 最近 怎么样？", language="zh")
        lines = split_utterance(original)
        self.assertEqual([x.text for x in lines], ["你好。", "最近 怎么样？"])

    def test_output_order_is_stable(self):
        first = utterance([("Hello.", 500, 30), ("Welcome.", 500, 0)])
        second = Utterance("End", 1200, 1600, "en")
        actual = split_utterances((first, second))
        self.assertEqual([x.text for x in actual], ["Hello.", "Welcome.", "End"])


if __name__ == "__main__":
    unittest.main()
