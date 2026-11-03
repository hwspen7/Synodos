# Path to the 85-point engineering release

This design targets a measured production-like local meeting transcription
system. It does not claim a current 85-point score or parity with Zoom.

## Model policy

| Stage | Baseline | Candidate | Deployment |
|---|---|---|---|
| Capture | WebAudioWorklet, 16 kHz PCM, ordered frames, WAV checkpoints | Browser-crash durable journal and resumable uploads | Real-time, never blocked by AI |
| Speech / VAD | Silero VAD + Whisper large-v3 | Qwen3-ASR-1.7B or Parakeet for supported languages | Only one primary realtime ASR; benchmark other models offline |
| Text segmentation | Whisper words, verified acoustic boundaries, punctuation | Qwen3 ForcedAligner when real word times missing | Preserve source words and original timestamps |
| Speaker activity | CAM++ and continuous acoustic embedding timeline | pyannote Community-1 offline, Sortformer for streaming | Separate diarization process, no assumed identity from text |
| Natural-language review | Saluki context correction | Acoustic second-opinion review on disputed spans | Suggestions only, version-checked human acceptance |
| Summary / QA | Saluki and evidence citations | none required for accurate transcription | Out-of-band, defer on active capture |

See docs/offline_speaker_refinement.md for the optional second diarization
model. It uses a separate Python interpreter to avoid overwriting the live
Whisper / CUDA stack.

The selected second opinion is pyannote Community-1, a locally runnable
diarization model requiring acceptance of its Hugging Face terms. Its labels
are local to a WAV and must never be applied to an entire resumed meeting
without an explicit matching recording offset.

Sortformer is a strong *streaming* candidate for up to four speakers. Qwen3
ASR and its forced aligner are stronger *secondary ASR* candidates for
English/Chinese than blind grammatical rewriting of Whisper results.
None becomes the default merely because it is newer.

## Evidence required for an 85-point release

Use a frozen audio set collected on the user's actual microphone and GPU.
At minimum, include:

- 30+ single- and multi-speaker annotated meeting clips, including quiet
  short answers, overlapping speech, names, pauses and interruptions.
- English WER and Mandarin CER, both measured against human verbatim
  transcripts; report values for every tested ASR model.
- Diarization DER on the same clips, including speaker recurrence,
  separate overlap reporting and a speaker-count confusion matrix.
- Word/time alignment errors, false speech, missed last words,
  duplicate confirmed text and wrongly merged dialogue turns.
- Two uninterrupted recordings of at least 60 minutes, with live Saluki
  workload, monitored GPU VRAM, p50/p95 latency, sample loss and autosave.
- Network congestion, WebSocket loss, disk full, browser crash and
  process restart; separately identify failed and recovered cases.

Thresholds for this project must be frozen before comparing models.
A successful synthetic unit test is evidence of correct integration, not
proof of recognition accuracy. Do not mark a feature complete without its
real-audio measurement.

## Remaining blockers

1. Persistent browser-side PCM journal and cross-reconnect idempotent replay
   have not been implemented. Existing sequence checks detect gaps but do
   not recover crash-lost microphone samples.
2. The independent pyannote path has tests and UI but the second model
   is not installed or benchmarked on the user's Windows computer.
3. No frozen labeled corpus demonstrates a WER, CER, DER or streaming
   latency advantage over the present baseline.
4. Streaming Sortformer and secondary Qwen3-ASR are candidates, not
   installed or proven improvements.
5. In-flight Ollama work cannot currently be preempted by an active meeting.

The release gate is objective. Do not state "85/100 achieved" until these
issues have measured acceptance evidence.
