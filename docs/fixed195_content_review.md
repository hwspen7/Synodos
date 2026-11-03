# Fixed195 evaluation: single-speaker scores first, real-dialogue review second

**Primary acceptance target:** all 130 reference-backed single-speaker clips,
20 English and 10 for each of the other eleven supported languages.
Compute character error rate (CER) per language, pooled weighted CER,
macro-average language CER (one vote per language), empty-transcript count
and per-case inference latency. Compare reused 56 old clips separately
from 74 genuinely new FLEURS examples to prevent inflated independent-test
claims. The model uses the existing production WhisperEngine pipeline,
not an oracle forced-language decode.

**Secondary, qualitative-only:** the 25 authentic stereo conversations
already downloaded from public DataHive (10 English, 5 Hindi,
5 Indonesian, 5 Russian). Run the same mixed-audio Whisper recognizer
and save candidate transcription, segment timestamps and machine-generated
review flags for missing text, possible repeated words, long paragraphs
and invalid timestamps. A human must listen to the actual recording
before declaring it intelligible, complete or correctly segmented.

These public multi recordings have NO human-verified transcript references
or speaker-timed alignments. Therefore no valid CER, WER, diarization DER
or voiceprint identity accuracy can be reported. WhisperEngine here is
ASR, **not** an implementation of the end-to-end diarization pipeline.
Speaker attribution is unmeasured, not assumed correct. A separate
full-session diarization replay is required to test voiceprints.

## Run on Windows

    cd E:\Meeting-Fixed10-Test
    git fetch origin feature/multilingual-fixed10-ui-refresh
    git switch --detach origin/feature/multilingual-fixed10-ui-refresh

First run a three-case smoke test of the gold singles:

    .\.venv\Scripts\python.exe tools\fixed195_content_review.py --mode singles --max-cases 3

Then complete the remaining 127 gold cases. The checkpoint is reused:

    .\.venv\Scripts\python.exe tools\fixed195_content_review.py --mode singles

Output files:
`benchmark/fixed195/content_review_singles.json`
`benchmark/fixed195/content_review_singles_summary.json`

Review authentic dialogue audio separately with a three-case smoke test:

    .\.venv\Scripts\python.exe tools\fixed195_content_review.py --mode multi --max-cases 3

Then finish the 25 available authentic conversations:

    .\.venv\Scripts\python.exe tools\fixed195_content_review.py --mode multi

Output files:
`benchmark/fixed195/content_review_multi.json`
`benchmark/fixed195/content_review_multi_summary.json`

Each run resumes previously completed cases after verifying source WAV
SHA-256s. A fingerprint of model/config/source/code refuses stale
checkpoint reuse. CLI default assumes local NVIDIA CUDA and
Faster-Whisper large-v3 float16. Scores are reference-based for singles
only. Audio and report contents remain local and are Git-ignored.

Human review checklist for multi-speaker audio:
- Does the transcript consist of intelligible, plausible speech?
- Are entire turns omitted when a second person interrupts?
- Is the same phrase repeated after overlapping speech?
- Are sentences grouped into readable conversational units?
- Does the beginning/end of the recording lose any dialogue?

The checklist is not an automatic claim of intelligibility. The 40
remaining unsourced dialogues across the other eight languages are
not required for this first qualitative pilot. The 65-dialogue
benchmark completion flag remains false until all data and verified
human references exist.
