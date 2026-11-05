# Meeting knowledge real-meeting release closure

This code adds an independent release gate to branch `feature/meeting-knowledge-quality` and the successful 42-question local Saluki 27B synthetic acceptance. It does **not** replace Phase 1/2 gates, change models, solve overlap diarization, or assert unverified real-meeting success.

## Required assets

The existing `E:\Meeting-Fixed10-Test` checkout, its Python environment, PostgreSQL 17 `meeting_test`, local Ollama 11434, the installed Saluki 27B and Qwen embedding model, and saved indexed *real* meeting transcripts are required. Run with the database credentials already set in the current PowerShell session. The scripts do not echo secrets and do not publish transcripts. All result files remain on `E:\AI-Cache\MeetingResults` by default.

The separately verified short-clip ASR policy remains **exactly ten languages**: `en, zh, fr, es, ar, id, ja, ru, de, hi`. Bengali and Portuguese are excluded from the *real-audio* release corpus here. The existing 12-language synthetic text-probe results are retained without being reinterpreted as ASR acceptance.

## Gold manifest

Create `E:\AI-Cache\MeetingResults\meeting_audio_review_gold.json` with schema `meeting-audio-review-gold` after listening to the recordings. Do not manufacture facts from an ASR hypothesis. At least three distinct actual recorded meetings, 20 independently audio-reviewed answerable questions and six negative questions are required. A contradiction case and a cross-meeting isolation case must be included. `audio_sha256` is the SHA256 of each original recording, not its transcript. The runner checks the corresponding private `audio_file` bytes against that hash before evaluating; the original file is never transmitted. Each answerable case names exact saved transcript segment IDs and an independently reviewed audio time interval. A missing word in the ASR must fail the answer, even if the model refuses correctly.

**Structure of a single meeting and two example case types** (illustrative, deliberately incomplete, will not pass the minimum cohort gate):

```json
{
  "schema": "meeting-audio-review-gold",
  "corpus_type": "real_recorded_meetings",
  "gold_source": "independent_human_audio_review",
  "reviewer_id": "reviewer-identifier",
  "meetings": [
    {
      "meeting_id": "real-meeting-001",
      "audio_file": "E:\\AI-Cache\\recordings\\REAL_01.wav",
      "audio_sha256": "REPLACE_WITH_64_HEX_DIGITS",
      "audio_reviewed": true,
      "synthetic_audio": false
    }
  ],
  "cases": [
    {
      "id": "qa001",
      "kind": "answerable",
      "meeting_id": "real-meeting-001",
      "language": "en",
      "question": "Which milestone was approved?",
      "expected_text": "HUMAN_CONFIRMED_PHRASE_FROM_AUDIO",
      "gold_segment_ids": [24],
      "gold_interval_ms": [165000, 179000],
      "verified_from_audio": true
    },
    {
      "id": "qa002",
      "kind": "cross_scope",
      "meeting_id": "real-meeting-001",
      "foreign_meeting_id": "OTHER_REAL_MEETING_ID",
      "foreign_expected_text": "FACT_ONLY_PRESENT_IN_OTHER_MEETING",
      "language": "en",
      "question": "Question about an answer present only in another meeting",
      "verified_from_audio": true
    }
  ]
}
```

Additional negative `kind` values are `unanswerable` and `contradiction`. For `contradiction`, include `"contradiction_segment_ids": [12, 15]` citing two different conflicting utterances. For `cross_scope`, specify `foreign_expected_text` that occurs in the foreign meeting and not the target meeting. These must be independently classified by listening and evidence review. The negative test passes **only** if the strict evidence QA abstains without citations.

Gold annotations and source media must remain private. Do not commit them to GitHub or copy them into CI. For transcripts saved only under a legacy backend, migrate into your separate `meeting_test` database with the existing migration tools *before* evaluating. No script copies live records automatically.

## One-command execution

Run from the project root after pulling the latest Meeting knowledge branch:

```powershell
cd E:\Meeting-Fixed10-Test
git fetch origin feature/meeting-knowledge-quality
git switch feature/meeting-knowledge-quality
git pull --ff-only origin feature/meeting-knowledge-quality
# The current shell must already have DATABASE_URL for the meeting_test database.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\Invoke-Meeting-Quality-Checks.ps1
```

The launcher freezes both successful synthetic reports, then evaluates real meeting cases in Lexical and Hybrid mode. By default it uses the **existing** saved index and does not reindex or alter meeting data. Use `-ReindexTestMeetings` only if the three or more reviewed meetings have not yet been indexed, and only in `meeting_test`. Supply `-AudioIdentityGate` and `-HostedCI` with independently produced, exact-commit status JSONs for the overall release decision.

The 42-case synthetic baseline is copied into a uniquely named immutable evidence folder, and SHA256 hashes are checked during the final decision. Each real-mode result includes Recall@8, citation attribution, exact answer phrase, negative abstention and p50/p95, with question IDs hashed and without raw transcript quotations. The real gate requires Recall@8 at least 90%, correct positive answers at least 85%, attribution at least 95%, valid refusals at least 95%, no cross-meeting leaks and p95 no greater than 15 seconds.

A real-audio RAG pass does not override the independently measured AMI inclusive DER 21.77% against the 20% threshold, missing subtitle-word verification, or disjoint human-verified speaker identity testing. The final report deliberately separates `local_synthetic_model_pass`, `knowledge_release_pass`, `code_and_database_ci_pass`, and `overall_product_release_pass`. No automatic model promotion, tag, merge, or production deployment is performed.

## CI

`tests.release.test_meeting_knowledge_quality` is deterministic, verifies fail-closed contracts, and uses no GPU, database, credentials, real media or user data. Run it in GitHub-hosted validation along with the existing complete suite. Local Saluki weights, annotated audio, human judgment and performance must be assessed on the user's machine, not simulated by hosted CI.

## Audio review export helper

To prepare the real gold review sheet from three *already saved* real meetings in `meeting_test`, run the following with your actual meeting IDs and audio file paths. This outputs the raw ASR text for private reference, but **does not claim independent audio truth**, and the generated manifest deliberately fails the release gate until a human reviews it.

```powershell
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
& $PY -m tools.meeting_audio_review_export `
  --meeting-id REAL_01 --meeting-id REAL_02 --meeting-id REAL_03 `
  --audio-path "E:\AI-Cache\recordings\REAL_01.wav" `
  --audio-path "E:\AI-Cache\recordings\REAL_02.wav" `
  --audio-path "E:\AI-Cache\recordings\REAL_03.wav" `
  --csv "E:\AI-Cache\MeetingResults\meeting_audio_review.csv" `
  --manifest "E:\AI-Cache\MeetingResults\meeting_audio_review_gold.json"
```

Listen to the actual corresponding audio, review the exported timestamps and speaker/content alignment, explicitly set `audio_reviewed: true` for verified recordings, fill in the real `reviewer_id`, and annotate at least 26 suitable questions in `cases`. Only then run the closure script.

The patch also installs `.github/workflows/meeting-knowledge-quality.yml`, a dedicated GitHub Actions check. GitHub CI asserts deterministic contracts only. It does not access recordings or Ollama and must not be treated as the final product acceptance gate. Commit and push the installed files to the feature branch when ready for hosted checks.


## Missing embedding model: local recovery

When the lexical report passes but the Ollama /api/tags listing does not contain
`qwen3-embedding:0.6b`, Hybrid has **not run**. A missing Hybrid JSON must not be
copied, renamed, fabricated or treated as PASS. The Knowledge acceptance
launcher now checks the installed embedder **before** rerunning Lexical and
supports `-HybridOnly` to preserve a successful lexical report.

Confirm which Ollama server the launcher is reaching:

```powershell
$endpoint = if ($env:MEETING_AI_ENDPOINT) { $env:MEETING_AI_ENDPOINT } else { "http://127.0.0.1:11434" }
(Invoke-RestMethod "$endpoint/api/tags").models | Select-Object name
```

If you operate a separate instance on port 11435, select it with
`-Endpoint http://127.0.0.1:11435` **only if both** Saluki and the embedding
model are installed on that instance. The two models must be reachable through
one configured endpoint for the current ModelGateway.

For an Ollama server whose **server-side** `OLLAMA_MODELS` is already configured
to an E-drive directory, pull the missing model explicitly:

```powershell
$env:OLLAMA_HOST = "127.0.0.1:11434"
ollama pull qwen3-embedding:0.6b
```

Setting `OLLAMA_MODELS` in a client-only PowerShell window does **not**
relocate an already-running Ollama server. Verify its actual storage location
before pulling to avoid using C-drive storage inadvertently.

With the correct endpoint and model available, resume:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\Invoke-Meeting-Knowledge-Acceptance.ps1 -HybridOnly
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\Invoke-Meeting-Quality-Checks.ps1
```

The second command still requires independently reviewed real meeting audio
before the final product gate can pass.
