# Phase 1/2 hardening and Meeting knowledge development handoff

## Scope of this change

This is the **six-stage product roadmap**:
Phase 1 model separation/comparisons, Phase 2 verified cross-meeting speaker
identity and revisions, Phase 3 retrieval-augmented meeting knowledge,
Phase 4 attributable evidence question answering. The separate short-clip
ASR language quality rubric is not the same roadmap stage.

**Code improvement is not a substitute for actual acoustic or independent
human evidence.** The upgrades in this branch are candidate implementation
changes. The PR remains a draft pending user-side gates.

The primary product priority is ordinary alternating-speaker meetings,
subtitle content preservation, stable speaker labels and manually verified
identity. Overlap remains measured in the original full DER, but is lower
engineering priority. Gold reference speaker counts must never be fed to a
production or evaluation model to inflate a score.

## Phase 1/2 final code hardening

1. Streaming cluster labels: a *global*, deterministic optimum assignment now
   retains as many previous published speaker IDs as possible when raw cluster
   IDs are renumbered. A previous greedy decision could steal a stable ID and
   generate aliases. Participant/manual labels are not reassigned or recycled.
   This is a code-level stability fix, not proof that the acoustic model
   correctly identifies a human.
2. Human-verified enrollment: two-model CAM++/ECAPA identity agreement only
   counts matched candidates whose separate enrollment evidence was explicitly
   human-verified. A pair of unverified cosine matches cannot silently gain
   identity confidence from apparent agreement. Automatic profile mutations
   remain forbidden, and raw cosine is not a probability.
3. Previously established meeting-store compare-and-swap safeguards,
   preserved original transcription and versioned manual revisions remain
   enforced by their dedicated regression tests.
4. No forced overlap-model optimization and no production model replacement.

## Phase 3: revision-safe RAG knowledge

The existing PostgreSQL intelligence worker and retrieval endpoint remain
the primary implementation; this change repairs two important gaps:

- `intelligence.store.replace_chunks` obtains the meeting row lock and
  checks the current saved-state revision **before** deleting prior indexed
  chunks. If the meeting was edited since embeddings began, the stale job
  fails and leaves the current index intact. Requeue indexing on the new
  revision; the existing retriever already filters stale chunk revisions.
- `intelligence.contracts.build_chunks` splits long text segments without
  discarding content. Every piece retains its segment and timestamp evidence.
  Long mixed-language transcripts are not sent to the embedder as one
  unbounded input.

The existing worker is opt-in; configured PostgreSQL and a running AI worker
are needed for full end-to-end acceptance. The source revision check is tested
without a live database and separately by existing PostgreSQL CI.

## Phase 4: fail-closed evidence QA

Both `/api/intelligence/ask` and
`/api/intelligence/ask/evidence` now queue quote-checked evidence QA.
The existing frontend uses the explicit evidence route. Old `ask` queued
jobs also take the strict source-validation path.

The model can SELECT at most five short continuous verbatim quotes from
retrieved chunks. Unsupported citation numbers, fabricated quotes, boolean
IDs, excessively short excerpts and duplicates are excluded. Its generated
freeform answer is deliberately discarded. When no attributable quote
survives, the answer explicitly abstains. The result retains meeting ID,
revision and source segment provenance while excluding embeddings.

This is *verbatim source checking*, **not semantic entailment or independent
audio fact verification**. An incorrect ASR transcript can still contain a
verbatim but incorrect statement. Do not auto-approve decisions, speaker
identities or production actions from a quote.

## Reproducible code gates on E-drive

In the user's `phase12-local` checkout, preserve existing untracked
`.venv-pyannote` and `benchmark` folders:

```powershell
cd E:\Meeting-Fixed10-Test
git fetch origin feature/meeting-knowledge-quality
git merge --ff-only FETCH_HEAD
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"

& $PY -m unittest `
  tests.backend.unit.test_speaker_stability `
  tests.backend.unit.test_identity_matcher `
  tests.backend.unit.test_identity_consensus `
  tests.backend.integration.test_manual_review `
  tests.intelligence.test_retrieval_index_cas `
  tests.intelligence.test_evidence_qa `
  tests.intelligence.test_contracts -q

if ($LASTEXITCODE -ne 0) { throw "Phase 1-4 code regression failed" }
```

For the actual AI service, set `MEETING_AI_ENABLED=1`,
`MEETING_STORAGE_BACKEND=postgres`, configure `DATABASE_URL`,
start the dedicated intelligence worker, index the saved meeting
(`index_meeting`) and then submit `POST /api/intelligence/ask/evidence`.
Record successful index and answer jobs with checked source revisions.
Never run the AI worker on the low-latency ASR event loop.

## Actual release evidence still outstanding

- 10 real AMI Pyannote meetings completed: speaker-time weighted **DER
  21.77%**, overlap-inclusive, zero collar. **FAIL** against the frozen
  20% acoustic gate, and five meetings were above 20%. This gate must
  remain visible even though overlap modeling is lower priority.
- Five actual solo/overlap audits matched their frozen baseline.
  The non-overlap DERs varied from 3.57% to 20.00%. These five cases
  cannot certify ten-meeting normal-speech accuracy or streaming quality.
- Published subtitle CER for mixed real meetings, 12-language final product
  coverage and full latency/soak remain unapproved. The older ten-language
  short-clip CER pass is a separate result.
- Phase 2 independently human-verified disjoint dev/holdout FAR/FRR:
  **not yet evaluated** on real independent meeting data.
- Phase 3 end-to-end PostgreSQL/embedding/re-index validation and Phase 4
  local LLM quote-selection with current meetings: **not yet executed**
  by CI because the user-owned E-drive models and corpus are unavailable.
- No main-branch merge or production model promotion is authorized solely
  by passing deterministic code tests.

**Decision:** Phase 1/2 implementation hardening can be code-frozen and
Meeting knowledge *development* can proceed, but Phase 1/2 acoustic + human evidence
remains blocked. The overall Meeting knowledge feature-release gate is also open.
