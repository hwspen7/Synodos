# Opt-in hybrid ASR: Whisper large-v3 + Qwen3-ASR-1.7B

This is a research candidate, not a production deployment. The default
MEETING_ASR_ENGINE remains whisper. A release needs sealed validation,
true-meeting evaluation, streaming-caption latency and speaker QA.

## Evidence-based first routing candidate

The Google FLEURS development cohort, 20 recordings per language
(200 total), returned the following lower error rates:

- Whisper large-v3: English, Mandarin Chinese, Arabic and Indonesian.
- Qwen3-ASR-1.7B: French, Spanish, Japanese, Russian, German and Hindi.

These are observed aggregate results, NOT a guarantee that an individual
sentence will be better. Japanese and Chinese use CER; the other languages
use WER. In Hindi Qwen 1.7B outperformed Whisper markedly, while English,
Chinese and Arabic favored Whisper.

With explicit language, send to that language's preferred backend. Retry
only after an exception or suspected missing text with audible activity.
Automatic language mode runs Whisper with its existing per-window language
detection and baseline text. Only Qwen-preferred language segments are
re-recognized by Qwen. If Qwen is unavailable, keep the Whisper baseline.
Low-confidence language assignments below 0.60 are not rerouted. This is
not a raw confidence contest between incomparable model probabilities.

## Separate Qwen runtime

The existing Meeting .venv uses its validated CUDA torch build.
The Qwen CUDA virtual environment is independent:

    E:\AI-Cache\Environments\meeting-qwen-asr\Scripts\python.exe

Do not install qwen-asr into the main Meeting .venv. Hybrid mode launches
one persistent Qwen CUDA worker through local private stdin/stdout. It
does not start a new process for each utterance, open any network port,
or send audio to an external service. Worker stderr logs remain on E:.

## Enabling in an acceptance test only

In PowerShell, after verifying the Qwen environment's CUDA=True and
GPU inference:

    $env:MEETING_ASR_ENGINE = "hybrid"
    $env:MEETING_ASR_MODEL = "large-v3"
    $env:MEETING_ASR_DEVICE = "cuda"
    $env:MEETING_ASR_COMPUTE = "float16"
    $env:MEETING_HYBRID_QWEN_PYTHON = "E:\AI-Cache\Environments\meeting-qwen-asr\Scripts\python.exe"
    $env:MEETING_HYBRID_QWEN_TIMEOUT_S = "180"

To switch back immediately:

    $env:MEETING_ASR_ENGINE = "whisper"

GPU warmup loads both models on the same GPU. On GPUs with 16 GB VRAM
this may not fit; test with the actual hardware and free memory headroom.
Hybrid auto mode has extra model inference and may fail a live-caption
latency budget even when its text is more accurate. Measure before
considering promotion.

The output Recognition engine is "hybrid" and includes per-call routing
audit in recovery_diagnostics["hybrid"]. All first-run routing rules
are fixed independently of the sealed FLEURS test cohort.

## Acceptance conditions before production

- Freeze and validate 100 test cases per language on an untouched
  official FLEURS test split; publish routing-versus-baseline macro error.
- Test explicit-language selection and auto-language detection separately.
- Require both backends to run on an actual GPU, no CPU fallback.
- Measure cold/warm behavior, memory, p50 and p95 per-utterance latency,
  empty captions and word timestamp quality.
- Evaluate AMI and user-consented real meeting speech for content
  retention, speakers, overlapping speech and tail truncation.
- Reject changes to the default production ASR until those pass.
