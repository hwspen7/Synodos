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

The original 65-dialogue DAI-ASR-I18N proposal is blocked by manual
access approval and is no longer the default preparation source.
Current open-source replacement provides 25 authentic stereo audio excerpts
from DataHive (10 English, 5 Hindi, 5 Indonesian, 5 Russian). There are no
official human-verified transcripts or diarization time labels in the
public DataHive release. They remain explicitly UNSCORED until annotated.

Additional 40 real recordings for Mandarin, Spanish, French, Arabic,
Bengali, Portuguese, German and Japanese need independently verified
sources and transcripts. See docs/fixed195_public_dialogue_sources.md for
candidate sources and restrictions. Do not fabricate speaker references
or declare the 195-case corpus completed from audio-only clips.

## Local setup on Windows

    cd E:\Meeting-Fixed10-Test
    $env:HF_HOME = 'E:\HF_cache'
    .\.venv\Scripts\python.exe tools\fixed195_prepare.py --source singles

This checks and imports **56 historical recordings** and prepares
**74 new FLEURS recordings**, without requiring gated DAI access.

Ungated real-audio collection (no DAI approval required):

    .\.venv\Scripts\python.exe tools\fixed195_prepare.py --source multi

This prepares up to 25 public stereo conversation excerpts as separate
speaker WAVs plus a mono mix, with missing transcripts clearly flagged.
These are kept in public_multi_progress.json and are not counted in the
final scored manifest. Only after 65 real conversations are independently
annotated and validated can you run:

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
https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample
