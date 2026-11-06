# Dialogue-aware transcript correction

The Review with AI control appears on each meeting page beside the saved transcript.
The feature is opt-in, runs after recording stops, and uses the existing local Saluki
Ollama model through the separate intelligence worker.

## Two review modes

- Faithful transcript: Suggest only likely within-segment duplication, punctuation,
  and contextual ASR corruption. Preserve informal speech and original grammar.
- Improve readability: Additionally suggest minimal grammar repairs, while keeping
  facts, speaker identity, negation and numbers unchanged.

This is text-only review. Dialogue context cannot reveal what an unclear
sound actually was, so no model suggestion can be certified as faithful to
the audio without listening to the recording.

## Workflow

1. Stop recording and confirm the meeting transcript is saved.
2. Click Review with AI and choose a mode.
3. The backend queues a context_correction job. The worker reads saved segments,
   sends bounded windows of nearby turns to the configured local model, and returns
   suggestions with source segment IDs, original text, expected revision, category,
   model-estimated confidence and a brief reason.
4. Read suggestions, replay WAV when necessary, then click Accept or Dismiss.
5. Accept calls the existing revision-checked segment PATCH endpoint, which retains
   previous text and audit history. A stale revision returns HTTP 409.
6. The page reloads the updated saved transcript. No unapproved suggestion is
   stored as the definitive meeting transcript.

The reviewer processes up to 32 segments per request. To review a longer meeting,
accept or dismiss outstanding suggestions and select Review next 32 segments.
Previously manually corrected segments are omitted from new model suggestions.

## Safeguards

- Never assign or rename speaker identities based on dialogue text.
- Reject proposals that change numerals, negations, clearly written two-part names,
  or stray segment IDs, or have no exact source text.
- Reject very large edits, low-confidence proposals and invalid model responses.
- Treat transcript text as untrusted input, not as instructions to the model.
- No audio is deleted or rewritten.
- The worker can fail or return no suggestions, which does not affect the original text.
- Generated confidence is a model estimate, not a calibrated probability.

## Configuration

The one-click launcher already enables local intelligence and Ollama with Saluki.
This requires MEETING_AI_ENABLED=1, PostgreSQL meeting storage, an active intelligence
worker and the configured local model available in Ollama.

API: POST /api/intelligence/jobs with JSON body

{"kind":"context_correction","meeting_id":"YOUR_MEETING_ID","payload":{"mode":"conservative","offset":0,"limit":32}}

Poll GET /api/intelligence/jobs/JOB_ID for the review result.

Backend tests: tests/intelligence/test_context_correction.py
This is not a real-audio accuracy certification.
