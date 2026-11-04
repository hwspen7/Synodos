# Live caption replay: reference scoring and segment-level audit

The Fixed195 multi-speaker clips have no verified verbatim reference
transcripts by default. Passing audio through an ASR model does not
create an independent reference. Delivery integrity, caption latency,
confirmed segments, and cluster diagnostics cannot certify completeness.

Run these commands from the repository root with the backend running
and the local Fixed195 WAV present. The script defaults to single-loop
real-time playback without automatic translation.

## 1. Export a private transcript and revision audit

PowerShell (use a fresh filename for each run):

    python -m tools.live_caption_ws_replay --audio benchmark/fixed195/audio/fixed195-en-m01.wav --loops 1 --speed 1.0 --output benchmark/fixed195/live_caption_en_m01_metrics.json --transcript-output benchmark/fixed195/local_hypothesis_en_m01.txt --segment-audit-output benchmark/fixed195/local_segment_audit_en_m01.json

These explicitly enabled exports contain spoken words and potentially
identifying names. They must stay local. The whole benchmark/fixed195/
directory is gitignored. Exporters refuse overwriting existing audit
or transcript files. Without opt-in switches, the aggregate report
remains free of speech text.

The audit contains:
- Final confirmed segments in audio order (not ID order), including text,
  timestamp, segment ID and speaker label.
- Observed speaker ID reassignments, speaker-name changes, confirmed text
  revisions and confirmed-segment disappearances, tied to segment ID and time.
- Review prompts for missing speaker embeddings, uncertain attribution,
  late relabeling, revised text, invalid timing, overlap, and long segments.

Review prompts do not mean that the event was an actual ASR or speaker
identity error. Verify by listening to the exact audio interval.

## 2. Create an independent human reference

Listen to the WAV and transcribe it manually into:

    benchmark/fixed195/human_reference_en_m01.txt

Do not treat an ASR hypothesis as ground truth. Record overlap and
speaker attribution separately when judging meeting quality.

## 3. Score the hypothesis against the human reference

    python -m tools.replay_transcript_score --reference-text benchmark/fixed195/human_reference_en_m01.txt --hypothesis-text benchmark/fixed195/local_hypothesis_en_m01.txt --output benchmark/fixed195/local_text_score_en_m01.json

This reports punctuation-insensitive CER, punctuation-sensitive character
error and WER when whitespace-separated words are meaningful. It does not
certify overlap, speaker labels, or sentence-boundary correctness.

## 4. Run regression tests

    python -m unittest tests.quality.test_live_caption_segment_audit -v
    python -m unittest discover -s tests -t . -v

Software tests do not substitute for a full real-audio replay.

## Interpretation

The previous en-m01 replay had 32 final segments, 5 segments without
embeddings, 8 labeled-to-different-labeled transitions, and 6 confirmed
text revisions. These are review targets, not proven accuracy failures.
The opt-in audit captures exact segments and revisions from a NEW replay.
Past aggregate-only JSON cannot reconstruct missing text history.
