# Fixed195 ungated conversation source transition

DAI-ASR-I18N access requires a manual Data Use Agreement review and the
user received a 403 while their request is pending. To avoid blocking
all experimental work, `tools/evaluation/corpora/fixed195_prepare.py --source multi` now uses
**public DataHive audio** and never attempts to access the gated DAI repo.

## What the public DataHive collector DOES provide

Source: https://huggingface.co/datasets/datahiveai/multilingual-conversations-sample
License: CC BY 4.0. Source has 61 genuine unscripted stereo conversations
recorded from distinct human callers, with left/right separate speaker
channels. It includes at least 10 English, 12 Hindi, 11 Indonesian and
20 Russian recordings, sufficient for the following first-stage allocation:

| Language | Authentic stereo excerpts |
|---|---:|
| English | 10 |
| Hindi | 5 |
| Indonesian | 5 |
| Russian | 5 |
| Total | 25 |

Selections are deterministic by pinned repo SHA and sorted original MP3
filenames. Extract a 75-second crop starting 15 seconds into each full
conversation, retaining synchronized speaker channels and a normalized
16kHz combined mono waveform. Acoustic validation rejects mono, duplicated
channels, non-finite audio or channels with too little signal.

**DataHive public preview has NO transcripts or human word timestamps.**
This first-stage dataset is audio-only, with `reference_status` set to
`missing_human_transcript` and `is_scoreable_asr=false`. It cannot be
legitimately counted as a transcript-verified Fixed195 completion.
Speaker channels are separate, but word-level diarization scores likewise
require independent human-labeled activity timelines. No fake reference
text, captions or diarization labels are generated.

    cd E:\Meeting-Fixed10-Test
    $env:HF_HOME = 'E:\HF_cache'
    .\.venv\Scripts\python.exe tools\fixed195_prepare.py --source multi

Outputs: `benchmark/fixed195/public_multi_progress.json`,
`benchmark/fixed195/public_multi_unscored.jsonl` and audio/
containing mixed stereo-derived mono WAV plus two isolated speaker WAVs.
The completed 130 single-speaker files remain unchanged.
Each source original MP3 is pinned by repo SHA and recorded by SHA256.

## Remaining languages require authentic, reference-ready material

The other 40 conversations (5 each) must come from separately verified
sources and must not be fabricated or marked complete automatically:

- Mandarin: https://huggingface.co/datasets/BAAI/CS-Dialogue
  Real Mandarin-English code-switching two-person conversation, manual
  utterance transcripts; CC BY-NC-SA 4.0. Mandarin-only sections may be
  selected without claiming they contain no code switching unless checked.
- Spanish: https://magichub.com/datasets/spanish-conversational-speech-corpus/
- French: https://magichub.com/datasets/french-conversational-speech-corpus-2/
- German: https://magichub.com/datasets/german-conversational-speech-corpus-2/
- Portuguese: https://magichub.com/datasets/brazilian-portuguese-conversational-speech-corpus/
- Arabic (Egyptian): https://magichub.com/datasets/egyptian-arabic-conversational-speech-corpus/
- Japanese: https://magichub.com/jp/datasets/japanese-duplex-conversation-training-dataset/
  MagicHub has real conversation samples but frequently requires website
  login to download and uses restrictive noncommercial/no-derivatives terms.
  None of these are an established automated, ungated, verified source for
  this script. Respect licenses before use.
- Bengali: https://huggingface.co/datasets/eQOURSE/multilingual-speech
  Ungated CC BY 4.0 paired dialogue with segment transcripts, but apparently
  only one full conversation per language, so five excerpts would not be
  five independent conversations. Do not count as five separate sessions.

Basis Conversations 1500 was also investigated. Although the dataset
has abundant conversation data, Hugging Face explicitly states that
requests are manually reviewed, so it is NOT a solution to the DAI issue.

## Status and release gates

Current original Fixed195 scoreable singles: 130/130.
Potential public two-person audio clips: 25/65, once downloaded and verified.
Fully transcript-verified dialogues: 0/65 until reference validation.
Unresolved language quota: 40/65, pending source and legal/format checks.

Original `benchmark/fixed195/lock.json` MUST NOT be created until 195
unique audio sources and 65 independent verified speaker references exist.
No scoring on pseudotranscripts as though they were human ground truth.
Keep historical Fixed150, Fixed120 and all existing singles untouched.

If strict no-registration-and-human-transcript coverage is mandatory for
all 12 languages, the 65-clip goal cannot currently be substantiated.
Do not replace unavailable sources with artificial mixes or mislabeled
broadcast single-speaker monologues.
