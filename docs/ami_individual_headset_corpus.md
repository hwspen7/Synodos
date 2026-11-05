# Official individual-headset AMI source inventory

To improve microphone/source quality, compare each original mixed-headset
recording against separate real headsets for EN2002a, ES2004a, IS1009a.
The official Edinburgh mirror serves recordings under each session's
audio directory. AMI Consortium recordings are CC BY 4.0.

This is source acquisition, not benchmark reference preparation.
Individual headset indices 0-3 are NOT automatically assumed to map
to speaker letters A-D in the separate NXT word transcripts.
The individual channel may still have cross-talk or faint speech.
A human must verify speaker-channel mapping, accurate timestamps and
audibility before using an isolated headset track as gold.

All WAV files are downloaded individually, integrity checked as 16 kHz
mono PCM, and checked for expected meeting duration. The resulting
E-drive inventory includes SHA256 for each file and exact source URL.
No audio, transcripts, speaker mapping or predictions are put on GitHub.

Dry run:
    python -m tools.prepare_ami_individual_headsets

Explicit download to E:
    python -m tools.prepare_ami_individual_headsets --apply

Output:
    E:\AI-Cache\MeetingReference\AMI\ihm\individual_headset_inventory.json

Twelve official WAV files (four per meeting) are considered separately.
Any pre-existing WAV is checked, never silently overwritten.
Incomplete downloads do not produce a verified manifest.
The manifest is immutable; rerunning --apply after full completion
refuses to overwrite it.

Until listening and speaker mapping are verified, the inventory has
eligible_for_model_scoring=false, may_auto_promote=false.
The independent headset recordings support acoustic assessment;
they do not replace the 10-language FLEURS reference benchmark or
the original mixed-track robustness cohort.
