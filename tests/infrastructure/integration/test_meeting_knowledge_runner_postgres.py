"""Full synthetic QA harness round trip on the PostgreSQL CI service.

This uses a deterministic quote selector rather than calling Saluki. It
tests the local 42-case runner, source contracts and cleanup with real SQL.
"""
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from tools.maintenance import meeting_knowledge_synthetic as acceptance


@unittest.skipUnless(os.environ.get("TEST_DATABASE_URL"),
                     "TEST_DATABASE_URL required for PostgreSQL integration")
class MeetingKnowledgeRunnerPostgresTests(unittest.TestCase):
    def test_full_42_case_runner_and_cleanup(self):
        import psycopg

        url = os.environ["TEST_DATABASE_URL"]
        with psycopg.connect(url) as conn:
            if conn.info.dbname != "meeting_test":
                raise RuntimeError("Refusing non-test database")
        meetings, checks, negatives = acceptance.fixtures()
        gold = {
            question: next(line for line in meetings[name] if
                           expected.casefold() in line.casefold())
            for name, question, expected in checks
        }
        refused = {question for _, question in negatives}

        class DeterministicGateway:
            def __init__(self, config):
                self.cfg = config

            def model_for(self, _task):
                return self.cfg.text_model

            def embed(self, _text):
                return [1.0, 0.0] if self.cfg.enable_embeddings else None

            def structured(self, _system, prompt, **_kwargs):
                question = prompt.split("Question: ", 1)[1].split(
                    "\n<evidence>", 1
                )[0]
                if question in refused:
                    return {"source_numbers": []}
                expected_sentence = gold.get(question)
                if expected_sentence is None:
                    return {"source_numbers": []}
                excerpt = prompt.split("<evidence>\n", 1)[1].split(
                    "\n</evidence>", 1
                )[0]
                positions = list(re.finditer(r"(?m)^\[(\d+)\] \[\d+\] ", excerpt))
                for i, match in enumerate(positions):
                    right = (positions[i + 1].start()
                             if i + 1 < len(positions) else len(excerpt))
                    block = excerpt[match.start():right]
                    if expected_sentence in block:
                        return {"source_numbers": [int(match.group(1))]}
                return {"source_numbers": []}

        for mode in ("lexical", "hybrid"):
            config = SimpleNamespace(
                enabled=True, provider="ollama",
                text_model="deterministic-ci-stub",
                embed_model="fixture-vector-v1",
                enable_embeddings=mode == "hybrid",
            )
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / (mode + ".json")
                environ = {
                    "DATABASE_URL": url,
                    "MEETING_STORAGE_BACKEND": "postgres",
                    "MEETING_AI_ENABLED": "1",
                }
                with patch.dict(os.environ, environ), patch(
                    "tools.maintenance.meeting_knowledge_synthetic.guard_environment"
                ), patch(
                    "intelligence.config.load_config", return_value=config
                ), patch(
                    "intelligence.model_gateway.ModelGateway",
                    DeterministicGateway,
                ):
                    code = acceptance.run(SimpleNamespace(
                        mode=mode, output=str(output),
                    ))
                self.assertEqual(code, 0)
                data = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(data["passed"])
                self.assertEqual(data["counts"]["answerable_total"], 36)
                self.assertEqual(data["counts"]["abstention_total"], 6)
                self.assertEqual(data["counts"]["answerable_pass"], 36)
                self.assertEqual(data["counts"]["abstention_pass"], 6)
                with psycopg.connect(url) as conn:
                    leaked = conn.execute(
                        "SELECT COUNT(*) FROM meeting_records "
                        "WHERE id LIKE 'knowledge_eval_%'"
                    ).fetchone()[0]
                self.assertEqual(leaked, 0)


if __name__ == "__main__":
    unittest.main()
