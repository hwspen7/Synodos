# Phase 09: Six-Issue Repair and Acceptance Register

Branch: refactor/phase09-six-issues

This is a draft repair branch. Never interpret green unit tests as proof
of real-world diarization or ASR quality. Only merge after the runtime and
recorded-audio acceptance conditions below have been met.

## Implementation status

| Requirement | Implemented and testable controls | Remaining acceptance blockers |
| --- | --- | --- |
| 1. Speaker-turn boundaries independent of ASR sentence boundaries | Acoustic embeddings are evaluated on independent voice windows in a bounded background worker. Verified ASR words are partitioned at detected turns. Inadequate evidence remains unidentified rather than being assigned to a plausible voice. Background completion updates the transcript on the processor thread. | Real labeled two-speaker and overlap tests, DER, stable speaker tracking over 60 min and calibrated embedding thresholds. Mixed single-channel audio cannot guarantee all overlaps are separable. |
| 2. Stop without losing acknowledged samples | Browser waits for AudioWorklet flush acknowledgement, sends a stop command with the exact PCM sample count, and requires backend receipt plus a successful persisted transcript before displaying success. Congestion is reported as failure. | Network reconnection with per-frame sequence, acknowledgement, durable spool/replay and abrupt browser crash recovery. The count covers emitted PCM, not physical input samples before sampling/resampling. |
| 3. Never discard short final audio | Final buffered audio is submitted to ASR regardless of the obsolete FINAL_DISCARD_S threshold. Regression includes a 100 ms trailing clip. | Whisper/FunASR VAD still cannot promise a nonempty transcript for 100 ms of speech; the original WAV is the only reliable evidence when enabled. |
| 4. Avoid duplicate rescans and hallucinated repetition | Unchanged audio endpoints are not rescanned, temporal duplicate guards preserve genuine repeated words, and confirmed overlap is tracked by absolute sample time. | Internal ASR hallucinations can still occur. Must evaluate repeated-speech clips and acoustic support with real recordings. Never blindly delete repeated phrases. |
| 5. Replace unvalidated small-model default | Quality-oriented Whisper large-v3 is the default candidate (still configurable). Optional startup prewarm is enabled by default, exposing missing runtime dependencies before the first meeting. Real-inference acceptance gates require labeled multilingual evidence. | Run real GPU benchmarks on the user's own meeting audio, compare Whisper small, medium, large-v3 and Qwen3-ASR, and verify real-time latency and memory. No model is certified yet. |
| 6. Decouple high-latency operations | Speaker analysis runs in a globally bounded worker pool, translation runs in a separate bounded pool, ASR retains higher GPU scheduler priority, and completed jobs are reconciled on the session processor thread. | Sustained multi-session load, failure injection, GPU queue starvation and 60-minute memory measurements. GPU operations still share one serialized inference scheduler. |

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
