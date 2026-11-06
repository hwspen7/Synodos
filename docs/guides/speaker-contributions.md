# Individual speaker contributions

This feature separates three tasks: **Insights**, **Statements**, and
**Identity**, without treating voice matches as verified human identities.

## Use

1. Stop and save the meeting recording.
2. Open **Review → Speakers** and select a speaker.
3. **Insights**: choose **Analyze this speaker**. The AI worker reads only the
   selected positive speaker ID's saved turns. It returns a source-cited
   summary, viewpoints, stated decisions, actions/commitments, and questions.
4. Click any evidence chip to scroll the main transcript to its original turn.
5. **Statements**: read this speaker's attributed turns in chronological
   order. The full original text and timestamps are preserved, and **Go to
   transcript** navigates to the canonical source.
6. **Identity**: select a specific segment, inspect its candidate voiceprint,
   and correct the name or text manually.

Unidentified speaker turns may be read in Statements, but cannot be
independently summarized until their attribution has been resolved. A speaker
ID may refer to incorrectly merged or fragmented speakers; confirm identity
before relying on individual analysis. Review results are *suggestions*,
not statements of fact or proof of any human's identity.

## Backend flow

- `POST /api/intelligence/jobs` with `kind=speaker_insights`,
  `meeting_id`, `payload.speaker_id` and `payload.source_revision`
  queues an idempotent PostgreSQL AI job.
- `GET /api/intelligence/meetings/{id}/speakers/{speaker_id}/insights`
  returns the latest artifact, queued/running job, and source revision.
- Worker selects the exact speaker ID from the **saved** meeting transcript
  and refuses missing, unidentified or conflicting speaker labels.
- The prompt contains only the selected speaker's original turns and IDs;
  the model is instructed to treat source text as untrusted data.
- All extracted items require one or more valid IDs from those turns.
  Uncited items, foreign IDs, and uncited narrative are dropped.
- Long speeches are split into bounded consecutive groups without mixing
  other speakers' statements.
- The source revision and SHA-256 fingerprint of selected segments (including
  text, time and attribution) are checked before persisting and before
  displaying an artifact. Editing text or identity invalidates old results.
- Tasks run through the existing AI worker; no additional model is required.
  When live ASR is active, the worker normally defers GPU work.

## Local tests

```powershell
cd E:\Meeting-Fixed10-Test
git switch feature/meeting-knowledge-quality
git pull origin feature/meeting-knowledge-quality
python -m unittest tests.backend.test_speaker_insights -v
node --test tests/frontend/speaker_contributions.test.mjs
node --test tests/frontend/meeting_tool_workspace.test.mjs
node --test tests/frontend/review_simplification.test.mjs
pnpm build
```

## End-to-end acceptance

Use a saved recording with at least two speakers and multiple genuine turns
per speaker.

- Select Speaker 1. Confirm Statements contains no Speaker 2 segment IDs.
- Generate Insights and verify all cited IDs belong to Speaker 1.
- Select Speaker 2. Its content must be independent of Speaker 1 and the
  previous speaker's job/result must not appear.
- Edit a Speaker 1 sentence or correct its attribution, save the meeting,
  and verify that the prior analysis becomes stale.
- Follow a citation. The main transcript must scroll to the original turn.
- Check speaker identity and manually correct a segment. The new selection
  should be reflected in Statements and subsequent analysis.
- While recording, the analysis action must be disabled.
- If the local AI model is unavailable, Statements and Identity remain usable.
- Check narrow panels/mobile and long multilingual meetings.

**Limitations**: citation membership is checked mechanically; semantic
entailment and accurate diarization still require human evaluation. Do not
interpret inferred commitments or human identities as verified facts.
