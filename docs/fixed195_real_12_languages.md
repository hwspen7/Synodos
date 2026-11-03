# Fixed195: 12-language benchmark with double English coverage

The primary 12-language benchmark contains **195 distinct recordings**:
130 single-speaker utterances and 65 authentic two-speaker conversations.
English has 20 single + 10 multi, twice the 10 single + 5 multi
allocated to each of the other eleven languages.

| Language | Old Fixed150 singles | New FLEURS singles | Genuine dual-speaker | Total |
|---|---:|---:|---:|---:|
| English (en) | 12 | 8 | 10 | 30 |
| Mandarin (zh) | 4 | 6 | 5 | 15 |
| Hindi (hi) | 4 | 6 | 5 | 15 |
| Spanish (es) | 4 | 6 | 5 | 15 |
| French (fr) | 4 | 6 | 5 | 15 |
| Arabic (ar) | 4 | 6 | 5 | 15 |
| Bengali (bn) | 4 | 6 | 5 | 15 |
| Portuguese (pt) | 4 | 6 | 5 | 15 |
| Russian (ru) | 4 | 6 | 5 | 15 |
| Indonesian (id) | 4 | 6 | 5 | 15 |
| German (de) | 4 | 6 | 5 | 15 |
| Japanese (ja) | 4 | 6 | 5 | 15 |
| **Total** | **56** | **74** | **65** | **195** |

## Frozen, non-duplicated single-speaker sources

English retains **all** original Fixed150 ids 009-020, representing
en-AU (009-012), en-GB (013-016), and en-US (017-020). English adds
eight FLEURS `en_us` validation clips at zero-based rows:
`113, 121, 129, 137, 145, 153, 161, 169`.

Each of the other 11 languages retains its exact 4 original Fixed150
single-speaker clips, with 6 new FLEURS validation rows:
`113, 121, 129, 137, 145, 153`.

Old audio bytes and original source SHA-256 values are copied after
verification of the existing Fixed150 lock. Every imported clip retains
its original case ID in `reused_fixed150_id` and its original provenance.
If a prior unfinished Fixed120 dataset has compatible FLEURS files
(same source revision, locale, and row), those can also be reused.
No original recordings are edited or counted twice in the 195.

## Real dual-speaker conversations

From gated `davidai-labs/DAI-ASR-I18N`, select 10 English and 5 in each
other language by deterministic overlap stratification. Require two
actual different speakers, original diarized word alignments, and
at least four valid words per speaker. English uses ten 5%-95%
overlap quantile positions; others use five 10%-90% quantiles.
Mixed audio is made locally from synchronized authentic speaker tracks.
Do not replace missing real recordings with synthetic combinations.

The DAI dataset requires explicit access approval and its evaluation-
only Data Use Agreement. Do not use these clips for model training,
optimization or distribution. The corpus is a blind independent test,
not a source for repeated model tuning.

## Local setup on Windows

    cd E:\Meeting-Fixed10-Test
    $env:HF_HOME = 'E:\HF_cache'
    .\.venv\Scripts\python.exe tools\fixed195_prepare.py --source singles

This checks and imports **56 historical recordings** and prepares
**74 new FLEURS recordings**, without requiring gated DAI access.

After DAI approval:

    hf auth login
    .\.venv\Scripts\python.exe tools\fixed195_prepare.py --source multi
    .\.venv\Scripts\python.exe tools\fixed195_prepare.py --verify

Files are private and Git-ignored under `benchmark/fixed195/`: WAVs,
`progress.json`, `manifest_single.jsonl` (130 cases),
`manifest_multi.jsonl` (65 cases), `lock.json` (only after all 195
are verified). Previous Fixed150 and Fixed120 outputs are untouched.

Scoring should report per-language CER/WER and macro-average language
results, separate historical reused clips from new ones, and measure
speaker attribution, overlap recall, missed/duplicated text, and
caption finalization latency. English has double raw weight, so
pooled language-unweighted quality must be reported separately.

Sources:
https://huggingface.co/datasets/google/fleurs
https://huggingface.co/datasets/davidai-labs/DAI-ASR-I18N
