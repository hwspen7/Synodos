"""Phase 07D references are bounded and measured conservatively."""
import unittest
from runtime_quality.evidence import evidence_audit, referenced_hits


class EvidenceTests(unittest.TestCase):
    def test_good_artifact(self):
        audit = evidence_audit({"actions": [{"title": "A", "source_ids": [4]}]}, [4])
        self.assertEqual(audit["citation_coverage"], 1)
        self.assertFalse(audit["independent_fact_check_performed"])

    def test_invalid_reference(self):
        audit = evidence_audit({"decisions": [{"title": "A", "source_ids": [999]}]}, [4])
        self.assertEqual(audit["citation_coverage"], 0)
        self.assertEqual(audit["invalid_reference_claims"], 1)

    def test_duplicate_claim(self):
        audit = evidence_audit({"actions": [{"title": "Do it", "source_ids": [1]},
                                           {"title": "do it", "source_ids": [1]}]}, [1])
        self.assertEqual(audit["duplicate_titles"], 1)

    def test_boolean_id_not_valid(self):
        audit = evidence_audit({"actions": [{"title": "A", "source_ids": [True]}]}, [1])
        self.assertEqual(audit["invalid_reference_claims"], 1)

    def test_references_deduplicated_and_embedding_removed(self):
        hits = [{"id": 1, "embedding": [1, 2]}, {"id": 2, "embedding": [3, 4]}]
        self.assertEqual(referenced_hits(hits, [2, 2, 1, 0, False]), [{"id": 2}, {"id": 1}])

    def test_absent_sources(self):
        self.assertEqual(referenced_hits([{"id": 1}], "1"), [])
