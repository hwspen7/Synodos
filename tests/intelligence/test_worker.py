import unittest
from unittest.mock import patch
from intelligence.worker import process
from intelligence.config import load_config
from intelligence.model_gateway import ModelGateway

class FakeGateway:
    def __init__(self):
        self.cfg=load_config()
    def model_for(self,kind): return 'test-model'
    def structured(self,system,prompt,model=None):
        return {'summary':'Correct summary','actions':[{'title':'Ship feature','source_ids':[2], 'owner':None}],
                'decisions':[{'title':'Fabricated','source_ids':[999]}]}
    def chat(self,*args,**kwargs):return 'Sample translation'
    def embed(self,text):return None

class ProcessTests(unittest.TestCase):
    def test_extract_keeps_only_sourced_claims(self):
        with patch('intelligence.worker._context',return_value=(2,[{'id':2,'start_ms':0,'end_ms':1000,'text':'Plan release'}])):
            with patch('intelligence.worker.store.save_artifact',return_value='artifact'):
                r=process({'id':'job','kind':'extract','meeting_id':'a','payload':{}},FakeGateway())
        self.assertEqual(len(r['data']['decisions']),0)
        self.assertEqual(len(r['data']['actions']),1)
    def test_translation_needs_valid_saved_segment(self):
        with patch('intelligence.worker._context',return_value=(1,[{'id':2,'text':'hello'}])):
            with self.assertRaises(ValueError):
                process({'id':'job','kind':'translate','meeting_id':'a','payload':{'target':'zh','segment_ids':[5]}},FakeGateway())
    def test_translation_persisted(self):
        with patch('intelligence.worker._context',return_value=(1,[{'id':2,'text':'hello'}])):
            with patch('intelligence.worker.store.save_translation') as save:
                r=process({'id':'job','kind':'translate','meeting_id':'a','payload':{'target':'zh','segment_ids':[2]}},FakeGateway())
        self.assertEqual(r['translations'][0]['text'],'Sample translation')
        save.assert_called_once()
    def test_no_evidence_qa(self):
        with patch('intelligence.worker.retrieval.search',return_value=[]):
            r=process({'id':'job','kind':'ask','meeting_id':None,'payload':{'question':'Where?'}},FakeGateway())
        self.assertEqual(r['sources'],[])
    def test_ask_rejects_empty_question(self):
        with self.assertRaises(ValueError):
            process({'id':'job','kind':'ask','payload':{'question':''}},FakeGateway())
