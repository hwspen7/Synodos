"""Model-agnostic local HTTP transport for Ollama or OpenAI-compatible servers."""
import base64
import json
import os
import logging
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class AIModelError(RuntimeError):
    pass


logger = logging.getLogger('meeting.intelligence.gateway')


def _bounded_int(name, default, minimum, maximum):
    raw = os.getenv(name, str(default))
    if not raw.strip():
        raw = str(default)
    try:
        number = int(raw)
    except (TypeError, ValueError) as exc:
        raise AIModelError(f'{name} must be an integer') from exc
    if not minimum <= number <= maximum:
        raise AIModelError(f'{name} must be between {minimum} and {maximum}')
    return number


def _optional_think():
    value = os.getenv('MEETING_AI_OLLAMA_THINK')
    if value is None or not value.strip():
        return None
    if value.strip().lower() in ('0', 'false', 'no', 'off'):
        return False
    if value.strip().lower() in ('1', 'true', 'yes', 'on'):
        return True
    raise AIModelError('MEETING_AI_OLLAMA_THINK must be 0 or 1')


def _request(cfg, path, payload):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    if cfg.api_key and cfg.provider == 'openai_compatible':
        headers['Authorization'] = 'Bearer ' + cfg.api_key
    req = Request(cfg.endpoint + path, data=data, headers=headers, method='POST')
    started = time.monotonic()
    try:
        with urlopen(req, timeout=cfg.timeout_s) as response:
            result = response.read(16 * 1024 * 1024 + 1)
        if len(result) > 16 * 1024 * 1024:
            raise AIModelError('Model response exceeded size limit')
        parsed = json.loads(result.decode('utf-8'))
        if not isinstance(parsed, dict):
            raise AIModelError('Model response must be a JSON object')
        logger.info('Model request %s completed in %.1fs (input_tokens=%s, output_tokens=%s, stop=%s)',
                    path, time.monotonic() - started, parsed.get('prompt_eval_count', 'n/a'),
                    parsed.get('eval_count', 'n/a'), parsed.get('done_reason', 'n/a'))
        return parsed
    except HTTPError as exc:
        logger.warning('Model request %s received HTTP %d after %.1fs',
                       path, exc.code, time.monotonic() - started)
        raise AIModelError(f'Model server returned HTTP {exc.code}') from exc
    except TimeoutError as exc:
        logger.warning('Model request %s timed out after %.1fs',
                       path, time.monotonic() - started)
        raise AIModelError(f'Model request timed out after {cfg.timeout_s}s') from exc
    except URLError as exc:
        logger.warning('Model request %s could not connect after %.1fs',
                       path, time.monotonic() - started)
        raise AIModelError('Model server connection failed') from exc
    except (ValueError, UnicodeError) as exc:
        raise AIModelError('Model server returned malformed JSON') from exc


class ModelGateway:
    def __init__(self, cfg):
        self.cfg = cfg

    def model_for(self, task):
        allowed = {'meeting', 'summary', 'extract', 'research', 'classroom', 'interview',
                   'standup', 'sales', 'ask', 'translate', 'vision', 'media'}
        if task not in allowed:
            raise AIModelError('Invalid model profile')
        return os.getenv('MEETING_AI_MODEL_' + task.upper(), self.cfg.text_model)

    def chat(self, system, prompt, *, model=None, json_mode=False, image_bytes=None):
        model = model or self.cfg.text_model
        if not model:
            raise AIModelError('No local text model configured')
        if image_bytes is not None and len(image_bytes) > 8 * 1024 * 1024:
            raise AIModelError('Image too large for vision request')
        if self.cfg.provider == 'ollama':
            user = {'role': 'user', 'content': prompt}
            if image_bytes is not None:
                user['images'] = [base64.b64encode(image_bytes).decode('ascii')]
            options = {
                'temperature': 0.1,
                'num_ctx': _bounded_int('MEETING_AI_OLLAMA_NUM_CTX', 8192, 2048, 32768),
                'num_predict': _bounded_int('MEETING_AI_OLLAMA_NUM_PREDICT', 1536, 128, 8192),
            }
            body = {'model': model, 'stream': False, 'messages': [
                {'role': 'system', 'content': system}, user],
                'options': options, 'keep_alive': '10m'}
            think = _optional_think()
            if think is not None:
                body['think'] = think
            if json_mode:
                body['format'] = 'json'
            out = _request(self.cfg, '/api/chat', body)
            if out.get('done_reason') == 'length':
                raise AIModelError('Model reached output token limit; increase MEETING_AI_OLLAMA_NUM_PREDICT')
            result = (out.get('message') or {}).get('content')
        else:
            content = prompt
            if image_bytes is not None:
                content = [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {
                    'url': ('data:image/png;base64,' if image_bytes.startswith(b'\x89PNG') else 'data:image/jpeg;base64,') + base64.b64encode(image_bytes).decode('ascii')}}]
            body = {'model': model, 'messages': [
                {'role': 'system', 'content': system}, {'role': 'user', 'content': content}], 'temperature': 0.1}
            if json_mode:
                body['response_format'] = {'type': 'json_object'}
            out = _request(self.cfg, '/chat/completions', body)
            result = ((out.get('choices') or [{}])[0].get('message') or {}).get('content')
        if not isinstance(result, str) or not result.strip():
            raise AIModelError('Model response is empty')
        return result.strip()

    def structured(self, system, prompt, model=None):
        raw = self.chat(system, prompt, json_mode=True, model=model)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AIModelError('Model did not return valid JSON') from exc
        if not isinstance(result, dict):
            raise AIModelError('Model JSON must be an object')
        return result

    def embed(self, text):
        if not self.cfg.enable_embeddings:
            return None
        model = self.cfg.embed_model
        if not model:
            return None
        if self.cfg.provider == 'ollama':
            data = _request(self.cfg, '/api/embed', {'model': model, 'input': text, 'truncate': True})
            items = data.get('embeddings') or []
            vec = items[0] if items else None
        else:
            data = _request(self.cfg, '/embeddings', {'model': model, 'input': text})
            rows = data.get('data') or []
            vec = rows[0].get('embedding') if rows else None
        if not isinstance(vec, list) or not 16 <= len(vec) <= 8192:
            raise AIModelError('Embedding model returned an invalid vector')
        return [float(x) for x in vec]
