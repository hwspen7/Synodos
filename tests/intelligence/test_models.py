import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from intelligence.model_gateway import ModelGateway,AIModelError
from intelligence.media import store_upload,extract
from intelligence.config import load_config
from intelligence.prompts import SYSTEM,answer_prompt
from intelligence.contracts import safe_job_kind

class Models(unittest.TestCase):
    def config(self):
        with patch.dict('os.environ',{'MEETING_AI_PROVIDER':'ollama','MEETING_AI_ENDPOINT':'http://127.0.0.1:11434'}):
            return load_config()
    def test_ollama_chat_mock(self):
        with patch('intelligence.model_gateway._request', return_value={'message':{'content':'Done.'}}) as request:
            out=ModelGateway(self.config()).chat('system','prompt')
        self.assertEqual(out,'Done.')
        self.assertEqual(request.call_args.args[1],'/api/chat')
    def test_ollama_json(self):
        with patch('intelligence.model_gateway._request', return_value={'message':{'content':'{"summary":"Done"}'}}):
            self.assertEqual(ModelGateway(self.config()).structured('system','prompt')['summary'],'Done')
    def test_reject_non_object_json(self):
        with patch('intelligence.model_gateway._request', return_value={'message':{'content':'[]'}}):
            with self.assertRaises(AIModelError):ModelGateway(self.config()).structured('s','p')
    def test_chat_model_error(self):
        with patch('intelligence.model_gateway._request', return_value={'message':{'content':''}}):
            with self.assertRaises(AIModelError):ModelGateway(self.config()).chat('s','p')
    def test_embedding_disabled(self):
        self.assertIsNone(ModelGateway(self.config()).embed('hello'))
    def test_invalid_task(self):
        with self.assertRaises(ValueError):safe_job_kind('DROP TABLE')
    def test_prompt_contains_citations(self):
        self.assertIn('meeting:abc',answer_prompt('Where?', [{'source_kind':'meeting','source_id':'abc','evidence':{'segment_ids':[5]},'text':'Yes'}]))
    def test_text_media_extraction(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'data.txt';p.write_text('A meeting discussion')
            chunks=extract(p,self.config())
            self.assertEqual(chunks[0]['evidence']['section'],'document')
    def test_safe_media_upload(self):
        from io import BytesIO
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as t:
            cfg=replace(self.config(),media_root=Path(t),max_upload_bytes=512)
            mid,path,digest,size,name=store_upload(BytesIO(b'%PDF-1.7 sample'),'../../material.pdf',cfg)
            self.assertTrue(path.is_file())
            self.assertTrue(path.name.endswith('.pdf'))
            self.assertEqual(size,15)
    def test_reject_invalid_pdf_upload(self):
        from io import BytesIO
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as t:
            cfg=replace(self.config(),media_root=Path(t))
            with self.assertRaises(ValueError):store_upload(BytesIO(b'not a pdf'),'material.pdf',cfg)
            self.assertFalse(list(Path(t).glob('*.pdf')))
    def test_reject_oversize(self):
        from io import BytesIO
        from dataclasses import replace
        with tempfile.TemporaryDirectory() as t:
            cfg=replace(self.config(),media_root=Path(t),max_upload_bytes=4)
            with self.assertRaises(ValueError):store_upload(BytesIO(b'hello world'),'material.txt',cfg)
    def test_multimedia_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'audio.wav';p.write_bytes(b'RIFF'+b'0'*32)
            with self.assertRaises(RuntimeError):extract(p,self.config())
