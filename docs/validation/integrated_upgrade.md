# Phase 1 + Phase 2 integrated upgrade (2026-11-04)

## Scope and non-claims

This branch is based on `fix/ts3003b-integrity-forensics-20261104`,
**not** the stale initial-import `main` branch. It does not change
the original frozen test data, speaker RTTM, reference hashes, or ASR
predictions. No private raw recordings or human transcripts are committed.

The agreed Phase 1 product goal is **10 supported languages, CER < 10%,
standard independent speaker DER < 20%** on a frozen protocol, plus
a human-reviewed mixed-audio final transcript and delivery integrity.
A low diarization error rate alone is never a passing text test.
The prior 23.768% independent TS3003b DER still FAILS the new 20% gate.
A zero-duration XML word is *not* evidence of one second of speech.

The new model-independent reviewer is a Phase 2 workflow, not a
claim that ECAPA, CAM++ or Nemotron was retrained or has been
rebenchmark-tested. Real meeting reliability still requires GPU playback,
manual audio review, and hosted CI.

## Phase 1: fixes

- The official AMI parser retains exactly zero-duration lexical
  annotations without fabricating timing. Non-finite, missing, or
  reversed in-crop lexical timestamps fail with speaker, token and
  timestamp context. Malformed events completely outside the crop are
  excluded. The report counts zero-duration words.
- The independent zero-collar, overlap-inclusive AMI gate now has
  `--require-under-20`. The stricter historic `--require-under-10`
  remains available for scientific comparisons. Forced-alignment
  diagnostic metrics remain separately labelled and cannot replace the
  original `only_words` reference.
- The text gate requires actual human review of the original mixed
  microphone audio. The GitHub Actions gate remains fail-closed.

## Phase 2: already integrated

The project already has an enrolled speaker library, persisted embeddings,
recording-linked segments, frontend manual text and speaker review, and
`/api/speech/meetings/{meeting_id}/segments/{segment_id}`
correction endpoints. This upgrade extends those features rather than
creating an incompatible identity database.

- Voiceprints now record `embedding_model` metadata. Legacy enrollment
  rows default to `camplus`. Different models and dimensions must not
  be compared. Invalid/short vectors are ignored by the new candidate
  matcher.
- The read-only endpoint
  `GET /api/meetings/{id}/segments/{segment_id}/identity-candidate`
  returns an abstaining candidate and raw cosine/margin diagnostics.
  These are **not** calibrated probabilities. No candidate is silently
  committed as a real person.
- `PATCH /api/meetings/{id}/segments/{segment_id}/review`
  accepts `{"expected_revision":0,"text":"correct text",
  "speaker_profile_id":"existing-id","reason":"Reviewed audio"}`
  with fields optional except `expected_revision`, and rejects stale
  revisions, unregistered speakers, active recordings and invalid
  requests. Each non-noop change is an auditable revision.
- The original frontend's existing `/api/speech` corrections now also
  retain `original_asr_text`, cross-meeting `speaker_profile_id` and
  structured before/after provenance. Speaker names are not used as
  cross-meeting identity keys.
- `GET /api/meetings/{id}/segments/{segment_id}/revisions`
  returns the original ASR text and audit history. Segments expose
  `speaker_profile_id`, `review_count` and `manually_corrected`.
  Revision provenance survives the existing pickle/PostgreSQL JSONB
  saved state and processor restart path.
- Data read/write for a review is serialized by the existing
  process-local meeting lock. Run one backend writer per meeting. A
  distributed multi-worker compare-and-swap protocol is not supplied.

## Windows single-command acceptance

Run on the machine containing the frozen audio, models and private
references. Do not set the human-reviewed switch until the actual
microphone mix has been checked and documented.

```powershell
cd E:\Meeting-Fixed10-Test
git fetch origin upgrade/phase1-phase2-integrity-20261104
git switch -c phase12-local FETCH_HEAD
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_phase12_acceptance.ps1 -PythonExe $PY
```

The script first runs isolated Phase 2 tests, then the frozen Phase 1
text/audio replay and authenticated CI diagnostics. Failure means
NOT ACCEPTED. Do not change test set or relax thresholds to produce a
green status. If the checkout has local uncommitted changes, preserve
them in a separate worktree before switching branches.

Additional manually reviewed replay, speaker benchmark, staged
offline/online retention and recording soak tests remain required
before Phase 3 release approval. GitHub hosted jobs must actually
complete, and no billing/runner block can be cleared by source edits.
