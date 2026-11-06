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


## Benchmark cold-start accounting

The paired ASR runner explicitly prewarms both hybrid backends before
the first measured clip. It records hybrid_preload_seconds separately.
Otherwise Qwen's first routed language (often French) would include
its entire model download and CUDA initialization as a misleading
inference latency while Whisper's startup is already excluded.
This is not permission to hide cold-start cost. Measure startup
readiness, VRAM use and per-window latency as separate product gates.


## Windows Qwen IPC failure policy

Windows CUDA extensions sometimes write non-UTF8 diagnostic bytes to
native stdout. A UTF8 text-mode subprocess pipe raised UnicodeDecodeError
in the reader thread, while Whisper fallback made the benchmark print OK.

The bridge now reads raw bytes and parses only tagged, ASCII-serialized
JSON frames. Untagged native bytes are ignored, corrupted tagged frames
fail closed, and a replacement worker uses a separate response queue.

The ASR benchmark separately stores the hybrid_routing audit including
qwen_attempts, qwen_accepted and qwen_failed. Any backend operational
exception stops the hybrid trial immediately, retains an intentionally
incomplete prediction file, and returns a failure status. This prevents
silent fallback from masquerading as healthy dual-model accuracy. In
live meetings Whisper still provides text fallback when Qwen fails.


## Full-context routing experiment (context-v2, NOT production default)

The first complete development run of hybrid-whisper-qwen17 scored a
macro error of 0.08676 on the 200-clip FLEURS validation cohort.
Qwen3-ASR-1.7B alone scored 0.07552 and Whisper alone 0.09179.
All six languages routed to Qwen scored worse in v1 than Qwen alone.
The suspected confound is double cropping from Whisper text bounds and
Qwen's second acoustic VAD, plus changed forced-language mode.

Keep the original hybrid-whisper-qwen17 model as a frozen comparison.
Use experimental hybrid-whisper-qwen17-context-v2 to test the hypothesis.
On an auto-language, <=30-second recording with one consistent,
confident Whisper language in fr, es, ja, ru, de, hi, pass the complete
original 16 kHz recording to Qwen using language=auto. This preserves
Qwen's independent acoustic context and mirrors its standalone decoder
input. Other audio follows the original per-Whisper-segment route,
which still requires live-multilingual quality verification.

The v2 audit records full_single_language_v2. Qwen errors and empty
replacement transcripts explicitly FAIL the research benchmark rather
than silently scoring Whisper fallback as Qwen output. Neither this
experimental v2 nor v1 changes Meeting's production default.

Run from the main Meeting .venv with a new immutable output path:

    & $py -m tools.evaluation.asr.asr_model_selection run --manifest $manifest --model hybrid-whisper-qwen17-context-v2 --language-mode auto --output "$results\hybrid-v2.json"
    & $py -m tools.evaluation.asr.asr_model_selection score --manifest $manifest --predictions "$results\hybrid-v2.json" --output "$results\hybrid-v2.score.json"

Full-window rerouting still depends on Whisper's language ID and adds
Whisper + Qwen compute. A higher score than Qwen alone or unacceptable
latency rejects this candidate. Do not tune on the sealed test cohort.


## Selected live Meeting mode: V2 (no V3 experiment)

The current benchmark-best V2 is selected for live Meeting operation.
The speech engine uses full_language_window=True when hybrid is chosen
and the Windows launcher uses MEETING_ASR_ENGINE=hybrid by default
unless explicitly overridden. It configures independent Qwen Python
on E: and automatic source-language recognition.

Run Start-Meeting.cmd in the updated checkout. A backend process already
running should be restarted with MEETING_FORCE_BACKEND_RESTART=1.
For rollback start with MEETING_ASR_ENGINE=whisper set.

This does not claim a real-meeting production release gate. It makes
the tested V2 available as the selected Meeting implementation without
changing its routing policy or starting V3 research.
