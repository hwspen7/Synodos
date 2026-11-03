# Fixed180: twelve languages, ten single and five genuine dual-speaker clips each

Supersedes the preliminary Fixed120 protocol as the *balanced primary test*.
Old Fixed120 outputs and the original Fixed150 corpus remain untouched.

Total 180 unique test clips: 120 single-speaker and 60 real dual-speaker.
Each language has exactly 4 original, already-frozen Fixed150 single-speaker
clips, 6 new FLEURS validation clips, and 5 real dual-speaker conversations.
The added singles use FLEURS zero-based validation rows 113, 121, 129,
137, 145, 153. The first 5 may be copied from a compatible previous
Fixed120 local run instead of downloaded again; the sixth is new.

| Language | Code | Original Fixed150 single IDs | Six new FLEURS rows | Real dialogues | Total |
|---|---|---|---|---|---|
| English | en | 009, 013, 017, 018 | 6 | 5 | 15 |
| Mandarin Chinese | zh | 053-056 | 6 | 5 | 15 |
| Hindi | hi | 061-064 | 6 | 5 | 15 |
| Spanish | es | 021-024 | 6 | 5 | 15 |
| French | fr | 025-028 | 6 | 5 | 15 |
| Arabic | ar | 057-060 | 6 | 5 | 15 |
| Bengali | bn | 089-092 | 6 | 5 | 15 |
| Portuguese | pt | 045-048 | 6 | 5 | 15 |
| Russian | ru | 049-052 | 6 | 5 | 15 |
| Indonesian | id | 081-084 | 6 | 5 | 15 |
| German | de | 005-008 | 6 | 5 | 15 |
| Japanese | ja | 065-068 | 6 | 5 | 15 |

English Fixed150 has 12 old single-speaker clips. The balanced set keeps
one Australian-English clip (009), one British-English clip (013), and
two US-English clips (017, 018) as four frozen examples. The remaining
eight are retained only in Fixed150 historical tests and are not counted
in Fixed180. This deterministic selection does not use model error rates.

## Frozen sources

The old 48 samples are copied without changing audio bytes from
`benchmark/fixed150/audio/` after the entire old lock and SHA256s are
verified. Each imported record retains the source's transcript, locale,
dataset provenance, original SHA256, and `reused_fixed150_id`.

The 72 FLEURS samples use the six new validation rows, recorded with
FLEURS sample IDs, pinned dataset revision, transcript, locale and SHA256.
If the prior Fixed120 `progress.json` contains the same dataset revision
and selected FLEURS row, its local WAV is reused with hash checking.
Other rows are downloaded from the pinned Google FLEURS dataset.

The 60 genuine dual-speaker examples remain five per language, selected
deterministically by overlap fraction from the gated DAI-ASR-I18N
evaluation corpus. Each uses synchronized, real speaker tracks, original
speaker-attributed words, and explicit overlap/annotation provenance.
Do not substitute synthetic mixtures for unavailable real conversations.

DAI access must first be approved, and its Data Use Agreement applies.
Its benchmark content cannot be redistributed, used for training or model
optimization. Use these conversations as a blind evaluation, not a tuning
set. Fixed180 metadata, audio and transcripts are kept local and gitignored.

## Local Windows PowerShell

    cd E:\Meeting-Fixed10-Test
    $env:HF_HOME = 'E:\HF_cache'
    .\.venv\Scripts\python.exe tools\fixed180_prepare.py --source singles

This verifies and imports 48 original clips and prepares 72 new FLEURS
clips. It does NOT require DAI access. An earlier Fixed120 run is optional.

After gated DAI approval:

    hf auth login
    .\.venv\Scripts\python.exe tools\fixed180_prepare.py --source multi
    .\.venv\Scripts\python.exe tools\fixed180_prepare.py --verify

Local outputs: `benchmark/fixed180/lock.json` after all 180 pass;
`manifest_single.jsonl` (120 records), `manifest_multi.jsonl` (60 records),
`progress.json` (checkpoint), and WAVs under `audio/`. Previous benchmark
directories are not deleted, rewritten or used to inflate sample counts.

Pass criteria should include CER/WER, language-identification accuracy,
speaker-attributed accuracy, overlap recovery and live subtitle latency.
Baseline scores for reused Fixed150 recordings are development history,
not an independently blind new evaluation set. Report retained and truly
new subsets separately to prevent optimistic final quality claims.
Keep FLEURS reading-language/accent scope distinct from conversational
DAI performance; do not infer that either source covers all dialects.

Sources:
https://huggingface.co/datasets/google/fleurs
https://huggingface.co/datasets/davidai-labs/DAI-ASR-I18N
