# Reproducible FLEURS multilingual dataset freeze

Google FLEURS (CC BY 4.0) supplies model-independent published reference
transcripts for the ten-language Meeting audio gate. It is read speech,
not a substitute for verified live meeting recordings. Never select
samples based on which model scored best.

A locked corpus manifest records the upstream source revision,
language, immutable case ID, WAV SHA256, transcript, development versus
sealed-test split, audio duration, official source and selection seed.

On Windows, all download caches, WAVs and manifests are constrained to E.
The command defaults to DRY RUN. Download only with --apply.

    cd E:\Meeting-Fixed10-Test
    & .\.venv\Scripts\python.exe -m pip install --no-deps "datasets==5.1.0"
    & .\.venv\Scripts\python.exe -m pip install "huggingface_hub" "soundfile>=0.13" scipy
    & .\.venv\Scripts\python.exe -m tools.freeze_fleurs_asr_corpus --split validation --per-language 20 --apply

Result manifest:
E:\AI-Cache\MeetingReference\ASRSelection\fleurs_validation_20\manifest.json

The short development set has 200 cases. It may be used for debugging
engine setup and decoding differences. It is not release test evidence.

After freezing and testing model configurations on validation data,
freeze the official sealed test set with an untouched fixed revision:

    & .\.venv\Scripts\python.exe -m tools.freeze_fleurs_asr_corpus --split test --per-language 100 --apply

The sealed test contains 1,000 cases, one hundred per target language.
Do not tune thresholds or optimize prompts against sealed test outcomes.
Published references are not independently human-listened audio gold.
The external corpus may be present in some pretrained model's training
data, so this is a blind-to-Meeting-tuning test, not proof of training
set isolation.

All selections are deterministic by hash of official source ID,
language and fixed seed. The inclusion policy excludes unusable files,
empty references and samples outside 0.8 to 30 seconds. A randomly
selected invalid case fails the freeze rather than being silently
replaced by an easier one. Any exception must be documented with a
new version of the benchmark, never by editing the frozen manifest.

Use a separate independently audio-reviewed AMI IHM clean-meeting
cohort for meeting-domain efficacy. Keep the existing three AMI mixed
headset sources as difficult audio; never relabel inaudible official
words as human-confirmed.


## Windows CUDA dependency safety (2026-11-05)

Do NOT run pip install datasets[audio] or pip install torchcodec in the
production Meeting virtual environment. Their latest resolver may replace
torch 2.7.0+cu128 with a newer incompatible version, breaking torchaudio
2.7.0 and potentially the existing Whisper/CUDA deployment.

Use the existing datasets package if it is installed. Audio records are
loaded with datasets.Audio(decode=False), then decoded via soundfile,
not TorchCodec. For 16 kHz WAV examples no FFmpeg is needed. For compressed
MP3 the installed libsndfile must support MP3 decoding. If this fails,
the tool raises an explicit error rather than modifying CUDA libraries.
The scipy resampler is used only when the source encoding differs
from 16 kHz.

To repair an environment affected by the original installation advice:

    & .\.venv\Scripts\python.exe -m pip uninstall -y torchcodec
    & .\.venv\Scripts\python.exe -m pip install "torch==2.7.0" "torchaudio==2.7.0" --index-url https://download.pytorch.org/whl/cu128
    & .\.venv\Scripts\python.exe -m pip check

Verify CUDA and test the actual audio decode before downloading an entire
cohort. Keep experimental qwen-asr packages in a separate E-drive venv.
