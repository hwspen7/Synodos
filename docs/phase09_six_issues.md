# Phase 09: Six-Issue Repair and Acceptance Register

Branch: refactor/phase09-six-issues

This is a draft repair branch. Never interpret green unit tests as proof
of real-world diarization or ASR quality. Only merge after the runtime and
recorded-audio acceptance conditions below have been met.

## Implementation status

| Requirement | Implemented and testable controls | Remaining acceptance blockers |
| --- | --- | --- |
| 1. Speaker-turn boundaries independent of ASR sentence boundaries | Continuous 2 s acoustic windows at 1.5 s stride now accumulate speaker evidence independently from ASR sentences; short segments receive consensus embeddings only when consistent. Existing word-aligned splitting remains. |  Real labeled two-speaker and overlap tests, DER, stable speaker tracking over 60 min and calibrated embedding thresholds. Mixed single-channel audio cannot guarantee all overlaps are separable. |
| 2. Stop without losing acknowledged samples | PCM frames now carry monotonic sequences and absolute sample offsets. Server detects missing, conflicting and replayed frames. WAV received and fsynced samples are tracked separately. Worklet tail is flushed before stop. |  Network reconnection with per-frame sequence, acknowledgement, durable spool/replay and abrupt browser crash recovery. The count covers emitted PCM, not physical input samples before sampling/resampling. |
| 3. Never discard short final audio | Final buffered audio is submitted to ASR regardless of the obsolete FINAL_DISCARD_S threshold. Regression includes a 100 ms trailing clip. | Whisper/FunASR VAD still cannot promise a nonempty transcript for 100 ms of speech; the original WAV is the only reliable evidence when enabled. |
| 4. Avoid duplicate rescans and hallucinated repetition | Unchanged audio endpoints are not rescanned, temporal duplicate guards preserve genuine repeated words, and confirmed overlap is tracked by absolute sample time. | Internal ASR hallucinations can still occur. Must evaluate repeated-speech clips and acoustic support with real recordings. Never blindly delete repeated phrases. |
| 5. Replace unvalidated small-model default | Quality-oriented Whisper large-v3 is the default candidate (still configurable). Optional startup prewarm is enabled by default, exposing missing runtime dependencies before the first meeting. Real-inference acceptance gates require labeled multilingual evidence. | Run real GPU benchmarks on the user's own meeting audio, compare Whisper small, medium, large-v3 and Qwen3-ASR, and verify real-time latency and memory. No model is certified yet. |
| 6. Decouple high-latency operations | Acoustic embeddings and offline translation run in bounded pools; ASR has highest shared scheduler priority. The Ollama worker defers claiming new jobs while recording is active. |  Sustained multi-session load, failure injection, GPU queue starvation and 60-minute memory measurements. GPU operations still share one serialized inference scheduler. |

## Defaults and behavior

- Interface language, default spoken language, translation output and AI output remain English.
- Users may explicitly select another speech language or Auto detect without changing output language.
- ASR model default: Whisper large-v3. Override with MEETING_ASR_MODEL.
- MEETING_PREWARM_ASR=1 loads the selected model at service startup; set to 0 for diagnostic workflows only.
- Long ASR utterances may be split into several speaker turns only if acoustic evidence is sufficiently distinct and word timings match the text. Otherwise the identity is marked unidentified; no transcript content is silently discarded.
- No account, cloud hosting, phone calls, or Zoom platform integration is introduced.

## Code validation (not an acoustic quality certification)

```powershell
python -m unittest discover -s tests -t . -v
pnpm type-check
pnpm build
node --test tests/frontend/contracts/*.test.mjs
```

## Real model validation

Collect a consented multilingual speech corpus with manually checked transcript
references and speaker-turn annotations. Put WAV PCM16 mono 16 kHz recordings
beside the JSONL manifest. Each record needs audio, reference, language,
language_mode and optionally terms fields.

Use a language set that includes English, Chinese, French, German, Spanish,
Japanese, Korean, Portuguese, Russian and Italian. Keep each language's
scores separate and include code-switching and overlapping-speaker clips.
Do not treat a single successful English recording as acceptance of other languages.

```powershell
.\.venv\Scripts\python.exe tools\phase07_benchmark.py --manifest .\benchmark\manifest.jsonl --output .\benchmark\large_v3.json --infer --engines whisper --whisper-model large-v3
.\.venv\Scripts\python.exe tools\phase09_acceptance.py --report .\benchmark\large_v3.json --languages en,zh,es,fr,de,ja,ko,pt,ru,it
```

The acceptance gate requires at least five verified samples per language,
reports per-language character error and mixed-token error, language-label
accuracy and p95 inference latency. Defaults are project targets, not Zoom
published performance numbers. Tune targets only before inspecting held-out
test results and never silently lower the bar after a failure.

## Manual end-to-end blocking cases

1. Stop while an utterance is less than one second long. Capture server audio_received and closed.saved, confirm no missing PCM, and compare the WAV to the source track.
2. Stop during a WebSocket backlog or disconnect. Recording must display an explicit failure, not a false green success.
3. Record two alternating speakers, then both simultaneously. Distinct speech must be attributed only with reliable evidence; ambiguous overlap must not be labeled as a confidently identified speaker.
4. Repeat an identical phrase at distinct timestamps. Both occurrences must remain. Replaying the same samples must not add a third occurrence.
5. Generate a translation slower than the live ASR cadence. Confirm subtitle ingestion continues and untranslated originals remain available.
6. Run a 60-minute noisy meeting, monitor P50/P95 latency, audio buffer pressure and GPU occupancy; crash and restart during the run; verify every persisted checkpoint.

## Phase 09 v2 engineering and acceptance

Speaker diarization runs on continuous acoustic windows independent of ASR sentences. A short transcript can inherit an acoustic embedding only with adequate temporal coverage and consistent speaker evidence. Unknown remains a valid outcome. This does not prove real-world DER.

The browser sends binary PCM frames with an audio sequence ID and absolute sample position. The server rejects gaps and conflicting frame replays. WAV recording includes explicit fsync checkpoints and reports durable sample counts. A hard browser crash may still lose audio not yet delivered to the server; there is no persistent client-side journal or cross-reconnect exactly-once mechanism.

During microphone recording, the separate Ollama worker defers claiming new jobs to protect live Whisper latency. It cannot preempt an already running Ollama request. The gate may be disabled using MEETING_AI_DEFER_DURING_RECORDING=0.

Quantitative speaker evaluation (manually labeled real recordings required):

```powershell
.\.venv\Scripts\python.exe tools\phase09_speaker_benchmark.py --reference .\benchmark\speakers_reference.json --hypothesis .\benchmark\speakers_prediction.json --max-der 0.20
```

The reference is a JSON array of {start_ms, end_ms, speaker}; the hypothesis is a JSON array of {start_ms, end_ms, speaker_id}. Unknown IDs count as missed speech and overlapping speakers count independently. Synthetic passing tests do not certify live diarization.

Remaining production acceptance blockers: browser-crash persistent audio spool/replay, measured DER on two-speaker and overlap clips, mixed-language WER/CER across labeled sources, recovery from backend crashes, sustained sixty-minute latency/VRAM measurements and concurrent-session fault injection.
