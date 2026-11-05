"""Regression coverage for bounded Ollama inference and actionable errors."""

import io
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from intelligence.model_gateway import AIModelError, ModelGateway, _request


class FakeResponse:
    def __init__(self, body):
        self.body = io.BytesIO(json.dumps(body).encode('utf-8'))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, limit):
        return self.body.read(limit)


class OllamaRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(provider='ollama', endpoint='http://127.0.0.1:11434',
                                   timeout_s=600, api_key='', text_model='saluki-test')
        self.gateway = ModelGateway(self.cfg)
        self.base_env = {
            'MEETING_AI_OLLAMA_NUM_CTX': '',
            'MEETING_AI_OLLAMA_NUM_PREDICT': '',
            'MEETING_AI_OLLAMA_THINK': '',
        }

    def test_default_parameters_bound_generation(self):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append((json.loads(req.data.decode('utf-8')), timeout))
            return FakeResponse({'message': {'content': '{"summary":"ok"}'}, 'done_reason': 'stop'})

        with patch.dict(os.environ, self.base_env), patch('intelligence.model_gateway.urlopen', fake_urlopen):
            result = self.gateway.structured('system', 'user', model='saluki-test')
        self.assertEqual(result, {'summary': 'ok'})
        body, timeout = calls[0]
        self.assertEqual(timeout, 600)
        self.assertEqual(body['options']['num_ctx'], 8192)
        self.assertEqual(body['options']['num_predict'], 1536)
        self.assertEqual(body['keep_alive'], '10m')
        self.assertEqual(body['format'], 'json')
        self.assertNotIn('think', body)

    def test_custom_parameters_and_optional_thinking(self):
        calls = []
        def fake_urlopen(req, timeout):
            calls.append(json.loads(req.data.decode('utf-8')))
            return FakeResponse({'message': {'content': 'ok'}, 'done_reason': 'stop'})

        env = {'MEETING_AI_OLLAMA_NUM_CTX': '4096',
               'MEETING_AI_OLLAMA_NUM_PREDICT': '512',
               'MEETING_AI_OLLAMA_THINK': '0'}
        with patch.dict(os.environ, env), patch('intelligence.model_gateway.urlopen', fake_urlopen):
            self.assertEqual(self.gateway.chat('system','hello'), 'ok')
        self.assertEqual(calls[0]['options']['num_ctx'], 4096)
        self.assertEqual(calls[0]['options']['num_predict'], 512)
        self.assertIs(calls[0]['think'], False)
        self.assertNotIn('format', calls[0])

    def test_evidence_qa_disables_thinking_and_uses_bounded_schema(self):
        calls = []
        def fake_urlopen(req, timeout):
            calls.append(json.loads(req.data.decode('utf-8')))
            return FakeResponse({
                'message': {'content': '{"source_numbers":[1]}'},
                'done_reason': 'stop',
            })
        with patch.dict(os.environ, self.base_env | {
            'MEETING_AI_OLLAMA_NUM_PREDICT': '4096',
            'MEETING_AI_OLLAMA_THINK': '1',
        }), patch('intelligence.model_gateway.urlopen', fake_urlopen):
            result = self.gateway.structured(
                'system', 'select quote', evidence_only=True,
            )
        self.assertEqual(result['source_numbers'], [1])
        body = calls[0]
        self.assertIs(body['think'], False)
        self.assertEqual(body['options']['num_predict'], 1024)
        self.assertEqual(body['format']['properties']['source_numbers']['maxItems'], 5)
        self.assertEqual(body['format']['required'], ['source_numbers'])
        self.assertTrue(body['stream'] is False)

    def test_evidence_qa_respects_tighter_user_token_budget(self):
        calls = []
        def fake_urlopen(req, timeout):
            calls.append(json.loads(req.data.decode('utf-8')))
            return FakeResponse({
                'message': {'content': '{"source_numbers":[]}'},
                'done_reason': 'stop',
            })
        with patch.dict(os.environ, self.base_env | {
            'MEETING_AI_OLLAMA_NUM_PREDICT': '512',
        }), patch('intelligence.model_gateway.urlopen', fake_urlopen):
            self.gateway.structured('system', 'question', evidence_only=True)
        self.assertEqual(calls[0]['options']['num_predict'], 512)

    def test_evidence_qa_still_refuses_truncated_model_response(self):
        response = {
            'message': {'content': '{"source_numbers":[1'},
            'done_reason': 'length',
        }
        with patch.dict(os.environ, self.base_env), patch(
            'intelligence.model_gateway.urlopen', return_value=FakeResponse(response),
        ):
            with self.assertRaisesRegex(AIModelError, 'output token limit'):
                self.gateway.structured('system', 'prompt', evidence_only=True)

    def test_bad_configuration_fails_before_request(self):
        for field, value in [('MEETING_AI_OLLAMA_NUM_CTX', '1'),
                             ('MEETING_AI_OLLAMA_NUM_PREDICT', 'abc'),
                             ('MEETING_AI_OLLAMA_THINK', 'sometimes')]:
            with self.subTest(field=field), patch.dict(os.environ, self.base_env | {field: value}):
                with patch('intelligence.model_gateway.urlopen') as send:
                    with self.assertRaisesRegex(AIModelError, field):
                        self.gateway.chat('system', 'user')
                    send.assert_not_called()

    def test_timeout_is_identified(self):
        with patch('intelligence.model_gateway.urlopen', side_effect=TimeoutError('timed out')):
            with self.assertRaisesRegex(AIModelError, 'timed out after 600s') as cm:
                _request(self.cfg, '/api/chat', {'model': 'saluki-test'})
        self.assertIsInstance(cm.exception.__cause__, TimeoutError)

    def test_http_error_is_identified_without_exposing_server_body(self):
        http_error = HTTPError('http://127.0.0.1:11434/api/chat', 400, 'invalid prompt', {}, None)
        with patch('intelligence.model_gateway.urlopen', side_effect=http_error):
            with self.assertRaisesRegex(AIModelError, 'HTTP 400'):
                _request(self.cfg, '/api/chat', {})

    def test_connection_error_is_identified(self):
        with patch('intelligence.model_gateway.urlopen', side_effect=URLError('refused')):
            with self.assertRaisesRegex(AIModelError, 'connection failed'):
                _request(self.cfg, '/api/chat', {})

    def test_incomplete_json_response_is_rejected(self):
        with patch('intelligence.model_gateway.urlopen', return_value=FakeResponse({'message': {'content': '{'}, 'done_reason': 'stop'})):
            with self.assertRaisesRegex(AIModelError, 'valid JSON'):
                self.gateway.structured('system', 'user')

    def test_generation_limit_is_reported(self):
        with patch('intelligence.model_gateway.urlopen', return_value=FakeResponse({'message': {'content': '{"x":1}'}, 'done_reason': 'length'})):
            with self.assertRaisesRegex(AIModelError, 'output token limit'):
                self.gateway.structured('system', 'user')

    def test_meeting_profile_still_supported(self):
        with patch.dict(os.environ, {'MEETING_AI_MODEL_MEETING': 'saluki-test'}):
            self.assertEqual(self.gateway.model_for('meeting'), 'saluki-test')


if __name__ == '__main__':
    unittest.main()
