# Real AMI recording listening queue

The earlier 10-second reference screening evaluated 1,715 of 12,303
officially annotated tokens (13.94%). That is a selected subset of
different utterances from the 30-second WER figure, not a comparable
full-meeting quality measurement. No model change is justified by these
numbers alone.

This tool uses the original official word timestamps to flag words
simultaneously annotated to distinct speakers. It counts zero-duration
annotations separately and measures nonoverlap reference-token fractions
without dropping an entire 10-second window for one brief interruption.
It also creates a balanced queue of actual short WAV clips, including
clean agreements, clear disagreements, possible ASR deletions,
possible insertions, and overlapping speech, when such windows exist.

The selected WAV files are byte-for-byte PCM crops of the original
locally verified AMI recordings. They are NOT synthesized or generated
speech. No audio model, database, Ollama service, or network is used.

Run from the updated project:

    cd E:\Meeting-Fixed10-Test
    git fetch origin feature/meeting-knowledge-quality
    git merge --ff-only FETCH_HEAD
    & .\.venv\Scripts\python.exe -m tools.prepare_audio_review_queue

Files are under E:\AI-Cache\MeetingReference\AMI\quality\listening_queue_<UTCSTAMP>:
- clips/*.wav: a short selection from each of three distinct meetings
- review_queue.csv: candidate category, source segment IDs, side-by-side
  official word reference and Whisper hypothesis, and UNREVIEWED status
- queue_inventory.json: per-meeting word-overlap and nonoverlap fractions

The default is twelve clips per meeting (36 total), each based on a
ten-second window with one second of audio padding at each edge.
Alternative option: --per-meeting 8.

Listen to the clips before making any factual statements about what
was intelligible. A reference word overlapping another speaker does
not prove that it is audible in the mono recording. The comparison
uses time-bucketed utterances and remains a screening mechanism.

This queue does not produce QA questions and cannot be used as the
human-reviewed real gold manifest. It never overwrites ASR predictions,
official annotations or meeting_test. When independently verified
audio facts are available, annotate >=20 answerable cases and >=6
negative cases, including at least one contradiction and one
cross-meeting isolation, in the existing private QA gold manifest.
