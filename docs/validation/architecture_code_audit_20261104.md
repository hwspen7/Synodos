# Phase 1 and Phase 2: Code Audit and Acceptance Boundary (2026-11-04)

## Scope and interpretation

This report uses the six-stage **product roadmap** (not the separate
"Phase 1 ASR text gate" release rubric).

1. Phase 1 = independent speaker-verification / diarization model interfaces,
   including CAM++, ECAPA-TDNN and Nemotron 3 comparisons.
2. Phase 2 = cross-meeting speaker identities, evidence, manual review,
   version tracking, and conservative voiceprint enrollment.

Phase 3 RAG, Phase 4 evidence QA, Phase 5 secondary Qwen3-ASR and Phase 6 GPU
scheduling are **out of scope**. A passing backend unit test is NOT a
source-backed, real-GPU scientific result.

## File-by-file audit: confirmed fixes

| Severity | Component | Confirmed failure mode | Remediation |
| --- | --- | --- | --- |
| P0 | api/speakers.py | Caller could set `human_verified=true` when a saved segment merely had the target `speaker_profile_id`, without an actual manual review event | Require matching persisted manual review evidence, a current speaker override, plausible embedding, and reject uncertain/mixed voice |
| P0 | tests/backend/integration/test_verified_enrollment.py | Old positive fixture "verified" a profile just by populating `speaker_profile_id`; never exercised actual review history | Negative tests for a forged profile and mixed audio, positive test only after `apply_review` |
| P0 | storage/postgres_meetings.py | Read-modify-write API calls ran outside the DB row lock used by `save_state`, so two processes could overwrite each other's corrected state | Read current row revision into metadata and check expected version under `SELECT ... FOR UPDATE` before write, raising conflict on stale state |
| P0 | storage/meetings.py | Local meeting saves used a shared temporary name and had no revision check | Per-store lock, unique temporary file, versioned snapshots, and reject stale writes. Filesystem locking remains process-local |
| P0 | api/meetings.py | Manual review and meeting settings saved previously read state without store-wide revision comparison | Pass expected store revision to both save paths and report HTTP 409 on conflicts |
| P0 | storage/speakers.py | A corrupt speaker library was silently replaced by an empty in-memory store | Fail closed with explicit error instead of silently discarding enrolled identities |
| P0 | storage/speakers.py | Voiceprint vectors could be non-finite, zero-norm, wrong dimension, or a forged verified flag | Validate and normalize entire batch before mutation, require verified-source provenance, roll back on save failure |
| P1 | sessions/identity/identity_matcher.py | A match from an unverified voiceprint was indistinguishable to the caller from one backed by a manual review | Return explicit verified-provenance / evidence level and prohibit any automatic identity update; raw cosine is never a calibrated identity probability |
| P1 | tools/validation/cross_session_acceptance.py | Twenty positive/negative pairs could reuse too few distinct original voice clips, overstating independent evidence | Enforce distinct clip minimum and consistent clip-to-meeting and clip-to-identity mapping; flag pair dependence |
| P1 | src/components/Phase2IdentityReviewPanel.vue | UI had identity suggestion and history but no saved-text editing or manual speaker selection | Add manual text and identity correction controls, source-audio confirmation prompt, revision-based API submissions |
| P1 | src/components/SpeakerVectorAdder.vue and src/api/client.ts | UI could only import unverified meeting voiceprints, blocking the normal human-verified enrollment workflow needed for the cross-meeting trial builder | Add explicit reviewed-only opt-in and evidence labels; server still enforces correctness |
| P1 | src/types.ts and storage/meetings.py | Frontend lacked stored review metadata necessary for verified enrollment | Surface review count, profile ID, uncertainty and review state without exposing vectors |
| P1 | tests/backend/unit/test_speaker_store_integrity.py | No direct tests for corruption, invalid batch, rollback and false verified flags | Add deterministic fail-closed regressions |
| P1 | tests/quality/integrity/test_cross_session_acceptance.py | No test for repeated pair inflation and changed clip provenance | Add adversarial trial manifests |
| P1 | tests/infrastructure/integration/test_postgres_optional.py | Did not explicitly check a stale snapshot write against a later saved revision | Add disposable real-PostgreSQL read/write/CAS regression |

## Existing architecture retained

- `speech/diarization/speaker_backends.py` exposes separate embedding and diarization
  protocols. CAM++ and ECAPA are verification models; Nemotron 3 is an
  acoustic diarization model. A cosine-similarity EER is **not** diarization
  DER. These metrics and model weights must never be mixed.
- `tools/evaluation/speakers/speaker_stage1_compare.py`, `tools/evaluation/speakers/speaker_stage1_merge.py`,
  `speech/diarization/model_candidates.py` and
  `tools/evaluation/asr/diarization_benchmark.py` preserve same-input comparisons
  and offline model isolation. Device availability, missing model weights
  and licenses are not silently called success.
- `sessions/manual_review.py` saves pre/post-edit state and a monotonically
  increasing per-caption revision. Original ASR text remains available.
- `sessions/identity/identity_consensus.py` requires two independent model decisions
  to agree and never auto-binds a person based on cosine alone.
- `tools/validation/identity_trial_builder.py` uses only explicitly human-verified
  source enrollment. `tools/validation/cross_session_acceptance.py` freezes
  development/holdout identity and meeting partitions and refuses
  automatic production threshold promotion.

## What the present CI can actually certify

Run this fast, model-free code-integrity check on the existing E-drive
Windows development copy. It prints a small terminal result, not JSON.

```powershell
cd E:\Meeting-Fixed10-Test
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\check_phase12_integrity.ps1 `
  -PythonExe "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
```

This covers the above model interfaces, identity decisions, saved revisions,
enrollment rules, cross-session trial integrity and deterministic diarization
scoring. The full unit output is kept in a private E-drive logfile.

## Remaining mandatory evidence (NOT PASSED by this code audit)

**Phase 1**: Evaluate actual CAM++/ECAPA voiceprint pairs with human labels,
and actual Nemotron (plus optional pyannote/DiariZen) speaker-turn outputs on
the same untouched, reference-backed acoustic recordings. Score the original
AMI only_words protocol across ten independently reviewed meetings, including
speaker overlap, speaker count, latency and failure counts. Never substitute
forced-aligned reference scores or model-free CI.

**Phase 2**: Record genuinely human-reviewed enrollment across independent
sessions and speakers, then evaluate the development/holdout split with pinned
audio/identity provenance. Measure false-accept/false-reject rates and review
conflicts using live PostgreSQL before enabling any operational threshold.
Model score agreement is not proof of human identity, and a UI confirmation
prompt cannot independently prove that the operator listened.

**Operational limitations**: The local pickle compatibility backend is not a
cross-process transactional speaker database. Production PostgreSQL is
recommended for concurrent sessions. Legacy pickle files must be protected
from tampering. This code audit does not redesign the legacy serialization
format or certify third-party GPU dependency compatibility.

**Release state**: These are code-integrity improvements on a DRAFT PR.
The original ASR text quality and acoustic diarization acceptance gates
remain unchanged and not yet passed. No main-branch release is authorized.
