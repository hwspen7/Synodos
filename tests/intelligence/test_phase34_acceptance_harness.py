"""Static checks for reproducible local twelve-language acceptance fixtures."""
import unittest

from tools.phase34_full_acceptance import fixtures, percentile


class Phase34AcceptanceHarnessTests(unittest.TestCase):
    def test_suite_covers_three_meetings_and_twelve_languages(self):
        meetings, questions, negatives = fixtures()
        self.assertEqual(set(meetings), {"atlas", "beacon", "cobalt"})
        self.assertEqual(len(questions), 36)
        self.assertEqual(len(negatives), 6)
        self.assertEqual(
            {value.split("-")[0] for _, _, value in questions if value.endswith("-731")},
            {"EN", "ZH", "FR", "ES", "AR", "ID", "JP", "RU",
             "DE", "HI", "BN", "PT"},
        )

    def test_every_expected_answer_occurs_in_its_scoped_fixture(self):
        meetings, questions, _ = fixtures()
        for name, question, expected in questions:
            with self.subTest(name=name, question=question):
                self.assertTrue(question.strip())
                self.assertIn(expected, "\n".join(meetings[name]))

    def test_no_untracked_or_non_fixture_meetings(self):
        meetings, questions, negatives = fixtures()
        ids = set(meetings)
        self.assertTrue(all(name in ids for name, _, _ in questions))
        self.assertTrue(all(name in ids for name, _ in negatives))

    def test_percentile_is_bounded(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.95), 4)
        self.assertEqual(percentile([], 0.5), None)


if __name__ == "__main__":
    unittest.main()
