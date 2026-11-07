"""Regression tests for the default meeting model profile and worker dispatch."""

import os
import unittest
from unittest.mock import patch

from intelligence.config import load_config
from intelligence.model_gateway import AIModelError, ModelGateway
from intelligence.worker import process


class ModelProfile06aTests(unittest.TestCase):
    @staticmethod
    def gateway():
        with patch.dict(os.environ, {
            'MEETING_AI_PROVIDER': 'ollama',
            'MEETING_AI_TEXT_MODEL': 'fallback-model',
            'MEETING_AI_ENDPOINT': 'http://127.0.0.1:11434',
        }):
            return ModelGateway(load_config())

    def test_default_meeting_profile_resolves(self):
        with patch.dict(os.environ, {'MEETING_AI_MODEL_MEETING': 'saluki-test'}, clear=False):
            self.assertEqual(self.gateway().model_for('meeting'), 'saluki-test')

    def test_meeting_uses_fallback_when_override_missing(self):
        with patch.dict(os.environ, {'MEETING_AI_MODEL_MEETING': ''}, clear=False):
            self.assertEqual(self.gateway().model_for('meeting'), '')
        with patch.dict(os.environ, {}, clear=False):
            previous = os.environ.pop('MEETING_AI_MODEL_MEETING', None)
            try:
                self.assertEqual(self.gateway().model_for('meeting'), 'fallback-model')
            finally:
                if previous is not None:
                    os.environ['MEETING_AI_MODEL_MEETING'] = previous

    def test_other_modes_unchanged(self):
        gateway = self.gateway()
        for mode in ('research', 'classroom', 'interview', 'standup', 'sales', 'ask', 'translate', 'vision', 'media'):
            self.assertIsInstance(gateway.model_for(mode), str)

    def test_unknown_profile_rejected(self):
        with self.assertRaisesRegex(AIModelError, 'Invalid model profile'):
            self.gateway().model_for('not-a-task')

    def test_default_meeting_summary_worker_dispatch(self):
        segments = [{'id': 1, 'start_ms': 0, 'end_ms': 1000,
                     'text': 'A team member agreed to complete a model benchmark.'}]
        parsed = {'summary': 'The team agreed to benchmark.', 'actions': [
            {'title': 'Benchmark a model', 'source_ids': [1], 'owner': None}],
            'decisions': [], 'open_questions': []}
        job = {'id': 'test-job', 'kind': 'summary', 'meeting_id': 'meeting-1', 'payload': {}}
        gateway = self.gateway()
        with patch.dict(os.environ, {'MEETING_AI_MODEL_MEETING': 'saluki-test'}):
            with patch('intelligence.worker._context', return_value=(1, segments)), \
                 patch('intelligence.worker.store.save_artifact', return_value='test-artifact') as save, \
                 patch.object(gateway, 'structured', return_value=parsed) as structured:
                result = process(job, gateway)
        self.assertEqual(result['artifact_id'], 'test-artifact')
        self.assertEqual(result['data']['actions'][0]['source_ids'], [1])
        self.assertEqual(structured.call_args.kwargs['model'], 'saluki-test')
        self.assertEqual(save.call_args.args[3], 'meeting')


if __name__ == '__main__':
    unittest.main()
