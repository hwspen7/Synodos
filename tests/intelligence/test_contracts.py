import unittest
from intelligence.contracts import safe_artifact,build_chunks,evidence_ids,source_hash
from intelligence.retrieval import lexical_score,rank,_cosine
from intelligence.media import safe_filename,validate_magic,SUPPORTED
from intelligence.config import load_config
from unittest.mock import patch

class ContractTests(unittest.TestCase):
    def test_evidence_rejects_unreferenced_ids(self):
        self.assertEqual(evidence_ids([2,9,True,'3'],{2,3}),[2])
    def test_artifact_drops_ungrounded_decision(self):
        d=safe_artifact({'summary':'A','decisions':[{'title':'Invented','source_ids':[99]}, {'title':'Real','source_ids':[1]}]},[1])
        self.assertEqual([v['title'] for v in d['decisions']],['Real'])
    def test_no_due_date_invented(self):
        d=safe_artifact({'actions':[{'title':'Follow up','source_ids':[1],'owner':'','due_date':'tomorrow'}]},[1])
        self.assertIsNone(d['actions'][0]['due_date'])
    def test_chunk_provenance(self):
        r=build_chunks([{'id':1,'text':'Hello','start_ms':2,'end_ms':11},{'id':2,'text':'world','start_ms':12,'end_ms':20}],10)
        self.assertEqual(len(r),1)
        self.assertEqual(r[0]['evidence']['segment_ids'],[1,2])
    def test_long_multilingual_caption_is_split_without_content_loss(self):
        original = ("\u4e2d\u6587\u4f1a\u8bae\u8bb0\u5f55\u9700\u8981\u4fdd\u7559\u6240\u6709\u6587\u5b57" * 200)
        chunks = build_chunks([
            {'id': 42, 'text': original, 'start_ms': 100, 'end_ms': 60000},
        ], max_chars=120)
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(chunk['evidence']['segment_ids'] == [42]
                            for chunk in chunks))
        parts = [chunk['text'].removeprefix('[42] ') for chunk in chunks]
        self.assertEqual(''.join(parts), original)
        self.assertTrue(all(len(part) <= 120 for part in parts))

    def test_long_english_captions_keep_boundary_whitespace(self):
        original = 'a test recording with multiple natural phrase boundaries ' * 15
        chunks = build_chunks([
            {'id': 2, 'text': original, 'start_ms': 0, 'end_ms': 40000},
        ], max_chars=40)
        recovered = ''.join(chunk['text'].removeprefix('[2] ')
                            for chunk in chunks)
        self.assertEqual(recovered, original.strip())
        self.assertTrue(all(len(chunk['text'].removeprefix('[2] ')) <= 40
                            for chunk in chunks))

    def test_chunk_bounds_fail_closed(self):
        for bad in (0, 1, -3, 90000, True, 20.0):
            with self.assertRaises(ValueError):
                build_chunks([], max_chars=bad)

    def test_hash_deterministic(self):
        self.assertEqual(source_hash('test'),source_hash('test'))
    def test_lexical(self):
        self.assertGreater(lexical_score('GPU latency','GPU latency'),lexical_score('GPU latency','unrelated'))
    def test_cosine(self):
        self.assertAlmostEqual(_cosine([1,0],[1,0]),1)
    def test_cosine_wrong_dimensions(self):
        self.assertEqual(_cosine([1,0],[1]),0)
    def test_rank_no_match(self):
        self.assertEqual(rank('abracadabra',[{'text':'hello','embedding':None}]),[])
    def test_safe_filename(self):
        self.assertNotIn('/',safe_filename('../../secret.pdf'))
    def test_pdf_magic(self):
        self.assertTrue(validate_magic('.pdf',b'%PDF-1.4'))
        self.assertFalse(validate_magic('.pdf',b'NOT_PDF'))
    def test_unknown_extension(self):
        self.assertNotIn('.exe',SUPPORTED)
    def test_remote_ai_default_rejected(self):
        with patch.dict('os.environ',{'MEETING_AI_ENDPOINT':'https://example.com','MEETING_AI_ALLOW_REMOTE':'0'}):
            with self.assertRaises(ValueError):load_config()
    def test_local_provider_allowed(self):
        with patch.dict('os.environ',{'MEETING_AI_ENDPOINT':'http://127.0.0.1:11434','MEETING_AI_PROVIDER':'ollama'}):
            self.assertEqual(load_config().provider,'ollama')
