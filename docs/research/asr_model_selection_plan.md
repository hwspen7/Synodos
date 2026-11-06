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
    python -m tools.evaluation.corpora.freeze_fleurs_asr_corpus --split validation --per-language 20 --apply

Set the E-drive development manifest:
    $manifest = "E:\AI-Cache\MeetingReference\ASRSelection\fleurs_validation_20_sf_v3\manifest.json"
    $results = "E:\AI-Cache\MeetingReference\ASRSelection\trials"
    New-Item -ItemType Directory -Force $results | Out-Null

Validate:
    python -m tools.evaluation.asr.asr_model_selection validate --manifest $manifest

Run three models against the EXACT same WAVs (model weights will be
retrieved to HF_HOME on the E drive if not already cached):
    python -m tools.evaluation.asr.asr_model_selection run --manifest $manifest --model whisper-large-v3 --output "$results\whisper.json"
    python -m tools.evaluation.asr.asr_model_selection run --manifest $manifest --model qwen3-asr-1.7b --output "$results\qwen17.json"
    python -m tools.evaluation.asr.asr_model_selection run --manifest $manifest --model qwen3-asr-0.6b --output "$results\qwen06.json"

Each trial is immutable and contains all case IDs, input audio SHA256,
recognized text, detected language, latency and explicit errors.
The first inference includes model loading and is excluded from the
reported warm inference latency, but still participates in accuracy.
A trial with errors is not an eligible model winner.

Score:
    python -m tools.evaluation.asr.asr_model_selection score --manifest $manifest --predictions "$results\whisper.json" --output "$results\whisper.score.json"
    python -m tools.evaluation.asr.asr_model_selection score --manifest $manifest --predictions "$results\qwen17.json" --output "$results\qwen17.score.json"
    python -m tools.evaluation.asr.asr_model_selection score --manifest $manifest --predictions "$results\qwen06.json" --output "$results\qwen06.score.json"

Compare:
    python -m tools.evaluation.asr.asr_model_selection compare --scores "$results\whisper.score.json" "$results\qwen17.score.json" "$results\qwen06.score.json" --output "$results\compare.json"

The scorer computes per-language WER and CER, omissions (empty
predictions), latency p95, real-time factor, language identification
accuracy for auto mode, the worst-language error and an equal-language
macro error. Chinese and Japanese use CER for the primary metric,
other languages use WER. The macro combines normalized error rates, not
raw token counts. The winner label is explicitly only for the
published read-speech cohort. Never promote production automatically.

## Sealed test

After development tuning, freeze the untouched FLEURS test set:

    python -m tools.evaluation.corpora.freeze_fleurs_asr_corpus --split test --per-language 100 --apply

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

    & $qpy -m tools.evaluation.asr.asr_model_selection run --manifest $manifest --model qwen3-asr-1.7b --language-mode auto --output "$results\qwen17.json"
    & $qpy -m tools.evaluation.asr.asr_model_selection run --manifest $manifest --model qwen3-asr-0.6b --language-mode auto --output "$results\qwen06.json"

If qwen-asr's installer updates torch incompatibly in the isolated
environment, repair only that experimental environment; do not edit
Meeting's working GPU stack. The Qwen inference runner imports the
local project source and does not require installing the full
Meeting backend into the isolated venv.

QwenEngine also requires faster-whisper for VAD and numpy in the isolated
runtime. Install these in the experimental venv, not in Meeting's
production environment. Do not run any automatic package upgrades in the
production environment.


## 2026-11-05 FLEURS streaming recovery

The original eager loader downloaded all three Parquet splits for a
single language, although only validation was requested. It also
reused the same WAV folder after switching from TorchCodec decoding to
libsndfile. A previously exported en_1634.wav can therefore disagree
with the updated decoder's reproducible PCM output.

The updated freezer uses a new sf_v2 output namespace, a pinned
freeze_plan.json, stream-only loading of the selected split and
fail-closed SHA256 validation. Do not delete the old folder or
overwrite conflicting WAVs. The original download cache may be reused
where the datasets library supports it. Running the same command again
continues a partial sf_v2 freeze without changing its locked revision.


## sf_v3 source-row identity

FLEURS repeats some official \`id\` values within a split. Sampling still
uses the immutable per-record hash rank and original row order for ties,
but case IDs and WAV filenames are now based on both original ID and
stable row index. This avoids collisions and retains all selected data,
including duplicated official IDs. The sf_v3 folder is separate from
the earlier incomplete sf_v2 export, with no deletion or overwrite.


## Sealed-test objective exclusion correction (sf_v4)

The 200-case development set remains frozen at
fleurs_validation_20_sf_v3. A newly frozen 1,000-case test set
uses fleurs_test_100_sf_v4, carrying forward the official dataset
revision pinned in the failed sf_v3 test plan. Selection remains
independent of all model outputs. Technical-invalid 0.8-30s audio
samples and empty official references are audited and replaced
by the next pre-ranked source row without dropping a language quota.


## Fourth candidate: actual dual-engine inference

The model selector also supports \`hybrid-whisper-qwen17\`. Unlike an
"oracle" assembled from gold language labels, this candidate calls the
real \`HybridEngine\` and its isolated Qwen CUDA worker at inference
time. An \`auto\` trial first detects languages from Whisper acoustic
windows, and a \`known\` trial explicitly passes the user's chosen
language. These are different trial conditions and cannot be merged
in one compare report.

To run the fourth candidate against the existing 200-case validation:

    $env:MEETING_HYBRID_QWEN_PYTHON = "E:\AI-Cache\Environments\meeting-qwen-asr\Scripts\python.exe"
    $manifest = "E:\AI-Cache\MeetingReference\ASRSelection\fleurs_validation_20_sf_v3\manifest.json"
    $results = "E:\AI-Cache\MeetingReference\ASRSelection\trials\hybrid_development_v1"
    New-Item -ItemType Directory -Force $results | Out-Null
    & .\.venv\Scripts\python.exe -m tools.evaluation.asr.asr_model_selection run --manifest $manifest --model hybrid-whisper-qwen17 --language-mode auto --output "$results\hybrid.json"
    & .\.venv\Scripts\python.exe -m tools.evaluation.asr.asr_model_selection score --manifest $manifest --predictions "$results\hybrid.json" --output "$results\hybrid.score.json"

Once the sealed corpus passes integrity validation, compare all four
models on identical audio and the same language-mode setting. Never
auto-promote based on development or sealed FLEURS alone. Validate
the hybrid on real meetings, failure recovery and streaming p95.
