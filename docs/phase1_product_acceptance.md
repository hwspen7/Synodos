# Phase 1 product acceptance contract

This project is evaluated on observable behavior expected of a reliable
meeting transcription product, not on the number of patches merged. Zoom's
publicly documented behavior (timecoded transcript sections, editable text,
correctable speaker names, recording-linked playback) is a UX reference, not
a claim to use or reproduce Zoom's proprietary implementations.

## Acceptance states

- Implemented: the function exists in the integrated product.
- Automated: deterministic unit, integration or end-to-end failure tests pass.
- Field verified: recorded, hand-annotated conversations and actual Windows
  operation meet the acceptance measurements below.
- Accepted: all required states are met. Passing CI alone is not accepted.

| Item | Required behavior | Deterministic regression | Field acceptance |
| --- | --- | --- | --- |
| 1. Speaker diarization | A-B-A returns the same stable IDs after pauses. Short answers are not assigned identities without acoustic evidence. Manual overrides persist. | Independent acoustic timeline, revision-safe speaker override, overlap/unknown DER scorer. | Labeled 2-4-speaker tests including short turns and overlaps. Report DER and confusion matrix. Project target DER at most 0.20 for non-overlap; report overlap separately. |
| 2. Audio integrity | Capture, transfer and WAV persistence distinguish sent, received and durable samples. Missing/out-of-order frames are explicit failures. | Sequence-gap, duplicate replay, WAV fsync checkpoint and stop count tests. | Normal stop: exact captured/sent/received/durable PCM equality. Abrupt process and network failures must be explicitly reported. Browser crash replay is still an open requirement. |
| 3. Final-word retention | Stop flushes every PCM sample before final ASR. Word timestamps refer to retained audio. | Worklet flush tests, short-tail replay, final cycle tests. | Hand-labeled tail phrases at 0.2, 0.5, 1 and 2 seconds; report recall rather than claiming perfect 200 ms recognition. |
| 4. No paragraph pile-up, duplication or invented words | Complete sentences have distinct readable time-coded sections where reliable word times exist. Legacy text uses non-destructive visual breaks. Neither implementation invents words or speaker IDs. | Real complaint transcript rendered as six sections, source word conservation, punctuation/pause cut, CJK/abbreviation regression, ASR temporal duplicate tests. | Compare displayed sections with 30+ hand-labeled recorded dialogue turns. Every displayed sentence must be traceable to source text and a real audio interval. Report hallucinations and boundary precision/recall separately. |
| 5. Recognition engine | Whisper large-v3 CUDA inference succeeds on target GPU; failures are explicit. Human-reviewed corrections remain distinct from raw transcription. | Startup CUDA inference; context correction edit audit and source checks. | Report WER for English and CER for Chinese on independent recordings with background noise, accents and code switching. Set thresholds against a frozen benchmark before release. |
| 6. Latency and isolation | Background speaker, translation and Saluki tasks cannot silently starve microphone capture. Queues are bounded and failures surfaced. | Priority scheduler, bounded workers, ASR/translation isolation, AI worker deferral tests. | 60-minute recording and 2-hour soak on Windows, plus network disruption and worker crashes. Report p50/p95 finalization latency, dropped samples, GPU memory peak and process recovery. |

## Explicitly open, not accepted

- A client-side crash-durable audio journal and resumable upload protocol.
- Real-world DER, WER, CER and speaker turn boundary recall from a frozen
  reference set. Synthetic unit fixtures are not field measurements.
- Repeated Windows 60-minute recordings with a simultaneous background
  Ollama workload and failure-injection proof.

## Required field evidence

For each release, archive the Git commit SHA, Python/Node versions,
Windows and GPU properties, test summary, paired audio/reference transcript,
scoring JSON and failure logs. An unresolved issue remains open even if all
automated checks are green.

## Release rule

Do not label all six issues "solved", "complete", or "Zoom-grade" until
the acceptance evidence exists and all blockers above are closed.
