# Registered Speaker Identity — Cross-Meeting Recognition

## Scope

A speaker enrolled in the local Speaker Library can be recognized in a later
meeting using compatible CAM++ embeddings. This is a *voiceprint match*, not
proof of a person's identity. The original transcript and human corrections
remain authoritative.

This module extends the existing meeting speaker system. It does not replace
Nemotron's anonymous acoustic diarization, ASR transcription, or the manual
speaker-correction workflow.

## Enrollment

1. Open **Speakers** and create a speaker profile with the intended name.
2. Add a clean voice recording or human-confirmed meeting excerpt.
3. Check that the profile has at least one CAM++ voiceprint longer than or equal
   to two seconds and that it is not flagged short.
4. Where possible, collect multiple samples from different sessions. Short,
   noisy, overlapping, or ECAPA-only profiles are not eligible for *automatic*
   global matching. They can still be kept in the library.

The saved record is kept in the local speaker storage. Its raw embedding is not
sent to the web frontend. Avoid registering voices without the speaker's consent.

## Subsequent meeting

Open a different meeting, then open **Settings → People & vocabulary**.

- **Recognize saved voices automatically** enabled, with no participants
  selected: consider eligible CAM++ profiles in the local library.
- Participants explicitly selected: compare only against those profiles.
- Automatic matching disabled with no participants selected: no registered
  profiles are searched.
- The deployment administrator can globally disable automatic matching by
  setting `MEETING_AUTO_SPEAKER_MATCH=0`.

Press **Apply speech settings** before starting a new recording. For a
running meeting, stop recording before changing these settings.

## Identity decisions

The backend extracts a CAM++ voice embedding for a usable speech turn. In
automatic mode, it normalizes vectors, evaluates the highest similarity for
each enrolled profile, and rejects a name assignment unless both conditions
are satisfied:

- Best similarity >= `MEETING_AUTO_SPEAKER_SIMILARITY`, default **0.78**.
- Best-minus-runner-up similarity >= `MEETING_AUTO_SPEAKER_MARGIN`, default
  **0.08**, when multiple profiles are eligible.

Both are raw cosine similarities, **not calibrated probabilities**. These
conservative defaults are initial guards, not validated operating points.
Tune only using a held-out set of same-person and different-person recordings
from the intended microphones and languages. Never infer a human identity
from a very short turn or from text content alone.

If a confident identity match exists, the registered speaker's name is
preserved in the canonical and published transcript. Nemotron's labels are
anonymous turn indexes and must not overwrite the recognized person's name.
When no match qualifies, the system retains an anonymous speaker label.

Manual speaker corrections always have priority over automatic predictions.
A saved speaker ID is local to each meeting. The persistent identity is the
profile ID in the speaker library.

## Persistence and controls

The meeting's `auto_match_saved` preference is persisted in meeting settings
and snapshots, passed on every WebSocket connection and restored on resume.
Selected profile IDs remain in `participant_ids`. Automatically discovered
profiles are never silently written into that selected-participant list.

On the transcript, a **Voice match** indicator denotes a machine match, not a
human-verified identification. Saved segment summaries record the matched
profile ID and raw similarity while omitting embedding vectors.

## Acceptance tests

```powershell
python -m unittest tests.backend.test_registered_voice_matching -v
python -m unittest tests.backend.test_registered_identity_pipeline -v
node --test tests/frontend/registered_voice_matching.test.mjs
pnpm build
```

Then run a real-audio evaluation:

1. Enroll Speaker A with two or more clean recordings.
2. Start Meeting B with no participant selection and auto mode enabled.
3. Speak as A and verify the registered name appears, with the Voice match
   indicator, and remains after stopping, saving, refresh, and resume.
4. Introduce an unknown speaker, a second enrolled speaker, short turns and
   overlap. Count false matches, misses, speaker switches, and unresolved turns.
5. Disable auto mode, save settings and start Meeting C. Confirm no global
   library lookup occurs.
6. Restrict Meeting D to one selected participant. Confirm unrelated registered
   profiles are not searched.
7. Correct a speaker manually and confirm later diarization cannot overwrite
   the correction.
8. Compare against recordings not used for enrollment. Report confusion
   matrix, false accept rate, false reject rate and retrieval latency.

A passing unit suite is not evidence that real-world identity recognition is
accurate. Do not mark the release as accepted until the real-audio evaluation
passes.
