# Live caption P0 stabilization (2026-11-05)

## Specific failure pattern

The user reported English phrases appearing while nobody was speaking,
short tail fragments ("You." / Chinese last-name fragment), and unexpected
Speaker 3 labels in a nominal two-person interaction.

The frozen FLEURS 200 speech-containing-clip ASR benchmark (hybrid V2
macro primary error 0.07533, no empty or failed predictions) never tested
open-microphone silence, streaming caption commits or speaker-count errors.
That benchmark alone cannot certify this P0.

## Changes in this branch

1. Hybrid fallback no longer calls the other ASR backend on a genuine
   `speech_detected=False` first result merely because broad RMS is >=.001.
   Decoder errors and acoustically positive empty transcripts retain their
   bounded fallback route. The selected hybrid V2 language policy is unchanged.
2. Only the *live hybrid* caption pipeline uses `uninformative_audio`.
   Clear near-digital silence and very steady low-level stationary noise
   are not sent to decoding or speaker inference. Exact candidate-PCM
   validation also blocks promoting such content into confirmed subtitles.
   A short rolling tail is kept for the next onset. Recorded WAV input is
   retained separately. The legacy and Whisper-only quality/reclamation
   paths remain unchanged.
3. Newly discovered anonymous CAM++ clusters need evidence from two
   non-overlapping usable turns (or a long, precisely word-aligned voice
   segment) before receiving a positive Speaker ID. Previously approved
   speaker labels and manually identified participants remain stable.
   Weak singleton clusters stay Unidentified Speaker; text is never dropped.
4. The time-aligned duplicate guard remains conservative. Actual repeated
   sentences at different times are valid speech and are NOT suppressed
   merely because their text matches.

## Safety limits

A human voice played through speakers, music with sung words, high-energy
nonstationary noise or AEC failures may still trigger speech recognition.
Stationary-noise checks are conservative heuristics, not a human speech
classifier; quiet voices and unusual microphones must be tested locally.
This revision is not a guarantee of zero hallucinations, or of speaker DER
or complete transcription alignment.

## Acceptance to run on the Windows host

After pulling this branch, restart the backend; do not reuse an old Python
process. Verify three separate scenarios using one new meeting each:

- 60 seconds of unvoiced open microphone in the actual room, with optional
  ordinary HVAC noise: no confirmed transcript, no live preview words, no
  positive speaker labels.
- Genuine quiet Chinese and English short replies: no legitimate words lost
  because of silence detection; the final segment matches the audio.
- Two distinct speakers taking several alternate turns: no unsupported
  third ID; compare confirmed text, live revision behavior and the raw WAV
  to identify any remaining tails or cross-language artifacts.

User-side actual acoustic evidence is required; CI uses deterministic
synthetic waveforms and mocks only. Never claim release solely from CI.
