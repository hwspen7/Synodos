# Meeting Phase 1 + Phase 2 integrated research upgrade

**Status: implementation committed, GPU and independent-human evaluations not yet accepted.**
This branch is deliberately a draft pull request until the frozen checks pass.
It never rewrites the published meeting captions using alternative-model guesses.

## Scope and accepted definitions

Phase 1A retains the 10-language / 110-clip frozen text gate and tests five
Whisper recovery and content-retention variants separately on AMI TS3003b.
The official AMI lexical archive is *not* a verified transcript of everything
intelligible in the mixed microphone. Human hearing review is mandatory.

Phase 1B includes three independent acoustic diarization paths: the original
Nemotron 3 completed **streaming** replay, an opt-in **offline** pyannote
Community-1 pipeline, and an opt-in **offline** DiariZen research model.
The common scorer uses 100ms frames, no collar, original `only_words`
speaker reference and overlap-inclusive DER. It does not silently replace
the reference with forced-aligned RTTM or choose a favorite on different
meetings. Offline and streaming latency are not directly comparable.
The model comparison defaults to **all ten matched AMI meetings**, not just
TS3003b. This is still a fixed benchmark, not universal real-world accuracy.

The Qwen3-ASR 1.7B shadow path uses VAD audio and already published caption
times to select bounded *candidate* gaps. It does not read AMI truth to select
gaps, nor promote unverified words into live text. Its outputs are confidential
review hypotheses, not an ASR accuracy claim. The existing production
Whisper and speaker models remain unchanged.

Phase 2 extends the enrolled speaker library with embedding-model provenance,
human-verified review status, auditable text/speaker revisions and a separate
cross-meeting identity suggestion. Automatic similarity remains an uncalibrated
cosine score, not a calibrated probability. Genuine/impostor trial construction
uses ONLY explicitly human-verified speaker enrollments from saved,
manually corrected meetings. Holdout meetings and people must both be
disjoint from threshold-development subjects. The selected threshold is
never automatically applied to live identity decisions.

**Target**: human-reviewed real-meeting CER <10%, original independent
full-overlap DER <20%, no substantial text loss, complete CI, and a
non-leaking Phase 2 holdout. The current historical TS3003b
15.86% reference-screening CER and 23.768% original-protocol DER
remain failures. The earlier 5.078% 10-language CER passes only its
separate fixed-clip gate.

## Source and output layout

Everything writable stays in `E:\Meeting-Fixed10-Test\benchmark\fixed195`.
Model caches, Hugging Face state and temporary paths are redirected to the
E-drive by the complete runner, if they were not already placed on E.

| Component | New entry point |
| --- | --- |
| Five Phase 1A ablations | `tools/run_phase1a_ablation.ps1` |
| pyannote / DiariZen adapters | `speech/phase1b_diarization_models.py` |
| One independent frozen meeting score | `tools/phase1b_diarization_benchmark.py` |
| Three-model 10-meeting paired score | `tools/run_phase1b_compare.ps1` and `tools/phase1b_compare.py` |
| Completed Nemotron replay manifest | `tools/phase1b_replay_manifest.py` |
| Qwen3-ASR audio-gap shadow | `tools/phase1b_qwen_shadow.py` |
| Review-backed genuine/impostor trials | `tools/phase2_build_trials.py` |
| Holdout speaker verification | `tools/phase2_cross_session_acceptance.py` |
| Combined local orchestration | `tools/run_phase12_acceptance.ps1` |
| Eight independent readiness gates | `tools/phase12_release_gate.py` |

## Optional model environments

Create dedicated environments **on E:**. They should not replace the
production `.venv`; DiariZen's repository pins older PyTorch components and
may require Python 3.10. Confirm its current README and license first.

- pyannote Community-1:
  `pip install pyannote.audio`, accept the upstream model conditions and
  arrange a local Hugging Face model directory or explicitly authorize a
  first download with `-AllowModelDownload`. Use `HF_TOKEN` locally if
  required. No token is written in benchmark JSON.
- DiariZen:
  use the official `BUTSpeechFIT/DiariZen` install procedure in a separate
  compatible environment. The **code is MIT**, but the pretrained
  `BUT-FIT/diarizen-wavlm-large-s80-md` weights are **CC BY-NC 4.0**.
  They are not embedded, redistributed or enabled by default.
- Qwen3-ASR:
  `pip install qwen-asr faster-whisper` in a dedicated Python 3.12
  environment, with a locally downloaded
  `Qwen/Qwen3-ASR-1.7B` or explicit permission for the first model
  download. The Qwen forced aligner is not used for shadow windows,
  avoiding false precision on unsupported languages.
- Nemotron 3:
  retain the existing pinned model and continuous streaming replay.
  An ordinary CAM++ replay is **not** accepted as Nemotron evidence.

Upstream references:
`https://huggingface.co/pyannote/speaker-diarization-community-1`,
`https://github.com/BUTSpeechFIT/DiariZen`,
`https://github.com/QwenLM/Qwen3-ASR`.

## Prepare and validate the ten paired AMI recordings

In a terminal with the existing Fixed10 environment:

```powershell
cd E:\Meeting-Fixed10-Test
$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
& $PY -m tools.ami_shadow10_prepare --root benchmark/fixed195/ami_shadow10
```

The 10 case JSONs must be complete and their WAV SHA-256 checksums must
match the pinned references. Do not edit or re-label frozen WAVs.

To create the ten independent Nemotron replays, start the normal backend
in a separate terminal with Nemotron streaming enabled on port 8000.
Use the existing tested Nemotron prerequisites and cache configuration.
Then use:

```powershell
& $PY -m tools.ami_shadow10_batch_replay `
  --root benchmark/fixed195/ami_shadow10 `
  --tag n3_phase12_01 `
  --require-nemotron `
  --ws-url "ws://127.0.0.1:8000/ws/audio?language=auto&record_audio=false&translate_to="
```

This must produce a **new** `phase1b_nemotron_manifest_n3_phase12_01.json`
with ten complete Nemotron tracks. A default or incomplete speaker
replay cannot satisfy this prerequisite. If ten completed independent
replays already exist, create a verified manifest instead:

```powershell
& $PY -m tools.phase1b_replay_manifest `
  --root benchmark/fixed195/ami_shadow10 `
  --output benchmark/fixed195/ami_shadow10/phase1b_existing_replays.json
```

## Create verified Phase 2 identity evidence

First, use the existing meeting UI to correct a saved segment's speaker
to an enrolled person. The meeting segment must have a
`speaker_profile_id` equal to that enrolled ID. An enrollment from
`POST /api/speakers/{id}/vectors/meeting` may then specify
`{"meeting_id":"...","segment_ids":[...],"human_verified":true}`.
An unverified or mismatched segment is refused.

Build **separate** groups with different speakers and different recording
sessions; the existing same-recording 68 pairs are diagnostics only and
are not a cross-meeting held-out validation.

```powershell
& $PY -m tools.phase2_build_trials `
  --dev-profiles "enrolledA,enrolledB" `
  --holdout-profiles "enrolledC,enrolledD" `
  --dev-meetings "dev1,dev2,dev3" `
  --holdout-meetings "test1,test2,test3" `
  --embedding-model camplus `
  --dev-output benchmark/fixed195/phase2_dev.json `
  --holdout-output benchmark/fixed195/phase2_holdout.json
```

These are illustrative identifiers, **not** actual speaker IDs. The
builder fails if the real verified enrollment samples are insufficient
or use incompatible embeddings. It outputs cosine scores and opaque
identifiers only; never raw vectors. Neither a local label assigned
by clustering nor an automatically generated transcript is sufficient
human truth.

## One-command full evaluation

The following is the complete runner interface. Use actual E-drive
interpreter and model paths, verified AMI replays, private segment-audit
WAV/JSON and identity manifests. Omit `-HumanReferenceVerified` unless
the mixed-microphone transcript has genuinely been listened to and
approved.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\run_phase12_acceptance.ps1 `
  -PythonExe $PY `
  -FullEvaluation `
  -PyannotePython "E:\Meeting-Models\pyannote\Scripts\python.exe" `
  -DiarizenPython "E:\Meeting-Models\diarizen\Scripts\python.exe" `
  -QwenPython "E:\Meeting-Models\qwen\Scripts\python.exe" `
  -NemotronReplayManifest "benchmark/fixed195/ami_shadow10/phase1b_existing_replays.json" `
  -PyannoteModelPath "E:\AI-Cache\Models\pyannote-community1" `
  -DiarizenModelPath "E:\AI-Cache\Models\diarizen-large" `
  -QwenModelPath "E:\AI-Cache\Models\Qwen3-ASR-1.7B" `
  -QwenAudio "benchmark/fixed195/ami_shadow10/audio/ami-TS3003b-1500-1740.wav" `
  -QwenSegmentAudit "benchmark/fixed195/ami_shadow10/transcript_TS3003b_segment_audit_YYYYMMDD-HHMMSS.json" `
  -Phase2Dev "benchmark/fixed195/phase2_dev.json" `
  -Phase2Holdout "benchmark/fixed195/phase2_holdout.json"
```

The paths above are **examples**, not promised existing files. A
missing model or missing human reference is a real blocked prerequisite,
not a reason to invent a green score. Models are run in dedicated
environments and only when `-FullEvaluation` is supplied.

Without `-FullEvaluation`, the runner performs deterministic local
regressions plus the existing Phase 1 live text/DER and CI closure. It
does **not** attest that Phase 1B model trials or Phase 2 held-out
verification have been run.

## Final Phase 3 release contract

Run `tools.phase12_release_gate` only with actual complete private
reports (language gate, human reviewed text gate, original Nemotron
acoustic score, ten-meeting paired model comparison, Qwen shadow,
Phase 2 disjoint holdout, hosted CI diagnostic and five-mode ablation).
All eight gates must be true for `next_stage=phase3`. The gate never
merges, tags, trains, removes data or rewrites production model settings.

Cross-meeting identity verification measures false accepts and rejects,
not DER; DER measures acoustic speaker time, not CER. Neither reference
leakage nor source separation guesses may be called recovered text.

The first-stage pass is a constrained engineering objective, not
a guaranteed real-world commercial quality claim.
