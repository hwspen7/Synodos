# Meeting ASR model-selection benchmark: full workflow

The goal is to select the best multilingual ASR engine by paired real WAV
inference, not by vendor claims or gaming public test-set statistics.

## Candidate models and scope

Model identifiers supported by the first-run inference harness:

- whisper-large-v3: current project WhisperEngine, float16 CUDA.
- qwen3-asr-1.7b: existing QwenEngine, CUDA, no forced aligner.
- qwen3-asr-0.6b: existing QwenEngine, CUDA, no forced aligner.

The first version uses existing production-style speech adapters, including
their VAD/windowing. All candidates receive exactly the same frozen WAVs,
sample rate, language-mode and reference texts held out from inference.
Known-language versus auto language ID are separate trials. No model
replacement or download occurs merely by scoring.

NVIDIA Parakeet v3 can be evaluated as an English/European-language
specialist in a later adapter, but it does not cover the full ten-language
policy and is not a candidate for the sole multilingual default.

## Dataset policy

The ten required languages are en, zh, fr, es, ar, id, ja, ru, de, hi.
Public Google FLEURS validation is used for setup/tuning; its untouched
official test set is an independent final read-speech comparison.
Use both multilingual published references and separately human-reviewed
AMI individual-headset real meeting speech. Existing AMI mono mixed-headset
recordings are kept as hard acoustic pressure tests. Published text alone
is NOT proof that faint overlapping speech is audible. No automatic
human labels or synthetic cases become real-meeting gold.

Freeze development FLEURS first (requires datasets, soundfile, scipy; NOT torchcodec):
    python -m tools.freeze_fleurs_asr_corpus --split validation --per-language 20 --apply

Set the E-drive development manifest:
    $manifest = "E:\AI-Cache\MeetingReference\ASRSelection\fleurs_validation_20\manifest.json"
    $results = "E:\AI-Cache\MeetingReference\ASRSelection\trials"
    New-Item -ItemType Directory -Force $results | Out-Null

Validate:
    python -m tools.asr_model_selection validate --manifest $manifest

Run three models against the EXACT same WAVs (model weights will be
retrieved to HF_HOME on the E drive if not already cached):
    python -m tools.asr_model_selection run --manifest $manifest --model whisper-large-v3 --output "$results\whisper.json"
    python -m tools.asr_model_selection run --manifest $manifest --model qwen3-asr-1.7b --output "$results\qwen17.json"
    python -m tools.asr_model_selection run --manifest $manifest --model qwen3-asr-0.6b --output "$results\qwen06.json"

Each trial is immutable and contains all case IDs, input audio SHA256,
recognized text, detected language, latency and explicit errors.
The first inference includes model loading and is excluded from the
reported warm inference latency, but still participates in accuracy.
A trial with errors is not an eligible model winner.

Score:
    python -m tools.asr_model_selection score --manifest $manifest --predictions "$results\whisper.json" --output "$results\whisper.score.json"
    python -m tools.asr_model_selection score --manifest $manifest --predictions "$results\qwen17.json" --output "$results\qwen17.score.json"
    python -m tools.asr_model_selection score --manifest $manifest --predictions "$results\qwen06.json" --output "$results\qwen06.score.json"

Compare:
    python -m tools.asr_model_selection compare --scores "$results\whisper.score.json" "$results\qwen17.score.json" "$results\qwen06.score.json" --output "$results\compare.json"

The scorer computes per-language WER and CER, omissions (empty
predictions), latency p95, real-time factor, language identification
accuracy for auto mode, the worst-language error and an equal-language
macro error. Chinese and Japanese use CER for the primary metric,
other languages use WER. The macro combines normalized error rates, not
raw token counts. The winner label is explicitly only for the
published read-speech cohort. Never promote production automatically.

## Sealed test

After development tuning, freeze the untouched FLEURS test set:

    python -m tools.freeze_fleurs_asr_corpus --split test --per-language 100 --apply

This produces 1,000 model-independent test samples with their own
immutable input manifest. Change only the manifest and output file
names for the final three-model repeat. Do not tune using the test results.

## Actual production selection

No model is accepted solely from FLEURS. For a complete decision,
also measure real AMI independent-headset speech, the original
three mixed recordings with verified audibility, own opt-in real
recordings, sentence preservation, end-of-speech loss, repeated
speech, multi-speaker segments, 10-language coverage and latency.
The release must still pass live caption, diarization, source evidence,
manual QA and PostgreSQL integration gates.

A reasonable decision rule: rank candidates by equal-language macro
errors, subject to no severe per-language regression and p95
latency meeting the live-caption objective. The numeric production
gate thresholds must be determined from product requirements and
frozen BEFORE examining the sealed test. Separate English-specialist
routing is a follow-on only if it improves end-to-end performance.


## Separate Windows Qwen environment and safe model dispatch

Do not install qwen-asr into the main Meeting .venv. The Meeting .venv
retains the validated torch 2.7.0+cu128, torchaudio 2.7.0 stack.
The Qwen models should use a separate E-drive environment; scoring
is Python-only and can run from the Meeting environment because it reads
only exported JSON predictions.

    py -3.12 -m venv E:\AI-Cache\Environments\meeting-qwen-asr
    $qpy = "E:\AI-Cache\Environments\meeting-qwen-asr\Scripts\python.exe"
    $env:PIP_CACHE_DIR = "E:\AI-Cache\Pip"
    $env:HF_HOME = "E:\AI-Cache\MeetingReference\ASRSelection\cache\huggingface"
    & $qpy -m pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    & $qpy -m pip install qwen-asr faster-whisper numpy
    & $qpy -m pip check
    & $qpy -c "import torch; print(torch.__version__,torch.cuda.is_available())"

Run the Whisper trial with the regular Meeting .venv Python interpreter.
Run the two Qwen trials with $qpy, e.g.:

    & $qpy -m tools.asr_model_selection run --manifest $manifest --model qwen3-asr-1.7b --language-mode auto --output "$results\qwen17.json"
    & $qpy -m tools.asr_model_selection run --manifest $manifest --model qwen3-asr-0.6b --language-mode auto --output "$results\qwen06.json"

If qwen-asr's installer updates torch incompatibly in the isolated
environment, repair only that experimental environment; do not edit
Meeting's working GPU stack. The Qwen inference runner imports the
local project source and does not require installing the full
Meeting backend into the isolated venv.

QwenEngine also requires faster-whisper for VAD and numpy in the isolated
runtime. Install these in the experimental venv, not in Meeting's
production environment. Do not run any automatic package upgrades in the
production environment.
