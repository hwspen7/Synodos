# Real-meeting reference corpus acquisition

The selected AMI recordings are **three new, distinct** recording sessions:
EN2002a, ES2004a, IS1009a (mixed-headset WAV). The same public mirror holds
AMI's manually annotated NXT XML word and segment files.

Source: https://huggingface.co/datasets/FluidInference/ami-corpus-mirror
Official corpus: https://groups.inf.ed.ac.uk/ami/corpus/
License: CC BY 4.0, AMI Consortium. Attribute the source if using or publishing results.

## Download and verify in E-drive storage

From the test worktree in PowerShell:

    cd E:\Meeting-Fixed10-Test
    & .\.venv\Scripts\python.exe -m tools.evaluation.corpora.prepare_meeting_reference_corpus --download

All four remote items are verified against the SHA256 hashes shown on the
mirror's individual file pages. Already verified files are reused without a
second download; files with unexpected contents are **never silently replaced**.
The script does not write to C, modify PostgreSQL, alter saved meetings or
create artificial human-reviewed gold. It stores an inventory at
E:\AI-Cache\MeetingReference\AMI\reference_inventory.json.

Verify existing files without network:

    & .\.venv\Scripts\python.exe -m tools.evaluation.corpora.prepare_meeting_reference_corpus

Expected downloads (approximately): 68.6 MB, 33.6 MB, 26.8 MB and 22.9 MB.
They are all English real meetings; multilingual ASR and live Mandarin speaker
separation require separate benchmarks. AMI's official manual annotations are
strong reference material, but are **not** a substitute for independent audio
review of the final meeting knowledge QA gold labels.

## Next independent gate

Use the downloaded audio to run the project's real ASR pipeline in isolated
meeting_test records, avoiding any collision with existing meeting IDs.
Preserve the exact processed WAV hash and audio-to-transcript timing mapping.
Before using tools.diagnostics.meeting_audio_review_export, verify there are three
saved meeting records with current-revision transcript segments. Independently
listen to the source recordings, correct transcription errors, and write at
least 20 answerable and six negative cases (including cross-meeting and
contradictory evidence) with segment IDs and time spans. Then run the real
knowledge acceptance. No downloaded reference annotation automatically sets
audio_reviewed or verified_from_audio to true.

The earlier synthetic lexical/hybrid 42-case PASS remains independent from
these new recorded-meeting gates.
