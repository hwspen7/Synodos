# Real meeting audio quality screening

The three indexed AMI meetings contain 1,096 Whisper ASR segments.
This tool compares local, immutable ASR predictions against separately
downloaded official AMI word references. It is read-only and requires
neither a GPU nor PostgreSQL or Ollama.

To run in E:\Meeting-Fixed10-Test:
1. Pull the feature/meeting-knowledge-quality branch.
2. Run: python -m tools.evaluate_meeting_audio_quality
3. Inspect the timestamped JSON and CSV in
   E:\AI-Cache\MeetingReference\AMI\quality

CSV rows give 30-second audio ranges, Whisper text, official reference
text, source ASR segment IDs and an explicit UNREVIEWED review status.
Priorities include OVERLAP_REVIEW, MISSING_ASR, ASR_WITHOUT_REFERENCE,
HIGH_DISAGREEMENT and SPOT_CHECK.

Pooled screening WER is calculated only on windows with at least five
officially annotated words and at most 1% different-speaker overlap.
Because overlapping human reference words may be inaudible in the single
mono mix, and midpoint-binned segments create boundary mismatches, this
is a screening metric, not definitive WER or a release certification.

After inspecting the output, use the existing
tools.meeting_audio_review_export module with the three meeting IDs
ami_eval_en2002a, ami_eval_es2004a and ami_eval_is1009a, paired with
their verified original WAV paths, to generate a draft review CSV and
meeting_audio_review_gold.json. This review manifest starts with no
verified questions and cannot unlock the release gate. Independently
listen to original audio and verify at least 20 answerable and six
negative cases, including contradiction and cross-meeting isolation.
Keep all transcripts, review sheets and WAV data private on the E drive.
