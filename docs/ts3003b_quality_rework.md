# TS3003b audio-first quality investigation (2026-11-04)

This is an engineering investigation, not an accuracy certification. The
single necessary condition for a quality release is measured improvement on
immutable audio, with no unacceptable regressions elsewhere.

## Frozen observed result

AMI TS3003b is one 240-second, real-time 1x replay. The human-derived
reference has 217.1 speaker-seconds and four speakers. The 100-ms, zero-collar,
overlap-aware scorer produced:

| Metric | V6 | Repaired 2026-11-04 | Delta |
| --- | ---: | ---: | ---: |
| DER | 0.40534 | 0.40580 | +0.00046 |
| Missed speaker fraction | 0.33947 | 0.33625 | -0.00322 |
| False alarm fraction | 0.01750 | 0.01796 | +0.00046 |
| Confusion fraction | 0.04836 | 0.05159 | +0.00323 |
| Unidentified transcript speaker-time | 29.9s | 26.7s | -3.2s |
| No-transcript speaker-time | 30.8s | 32.7s | +1.9s |
| Reference speech without any caption (clock) | 29.9s | 31.3s | +1.4s |
| Positive speaker identities | 2 | 2 | unchanged |
| Full-overlap speaker recall | 0.0 | 0.0 | unchanged |

The repaired branch FAILED the existing non-regression gate. It must not be
promoted on code-only evidence. Earlier 2026-11-04 tests lacking the anonymous
timeline did not produce a DER at all; a blank candidate converted to zero in
PowerShell is explicitly NOT an improvement.

## Root-cause map: verified behavior versus hypotheses

1. **P0, source voice loses transcription.** The existing live processor
   advances its rolling audio buffer through the end of its latest committed
   ASR segment. With MEETING_ASR_COVERAGE_RETENTION=0, it does not preserve
   earlier VAD-positive gaps for a second opportunity. This behavior is
   verified in sessions/processor.py. Whether each 31.3s annotated gap is
   caused by Silero VAD, Whisper decode, word alignment, or the commitment
   boundary cannot be determined from DER alone.
2. **P0, short replies are systematically deprioritized.** The default
   recovery ignores VAD-backed gaps shorter than 900ms. The default retention
   ignores those shorter than 1000ms. The verifier requires 160ms of voiced
   support. Those are source-code facts, not a claim that every missing
   450-900ms span contains real words. Experimental thresholds of 450ms
   for retry/retention and 100ms voiced support are enabled only for A/B.
3. **P0, speaker evidence is not independent enough.** The authoritative
   diarization is formed from embeddings attached to recognized text.
   Continuous 2-second windows are used to backfill selected missing
   embeddings, not as an independent overlap-aware speaker diarization
   timeline. Missing ASR and overlapping voices therefore compromise the
   downstream speaker count before clustering begins. Two reference voices
   have only 13.8s and 13.6s of annotated speech. One has 16 of its 19
   annotated intervals shorter than 1.2s. These are difficult with the
   current 1.2s CAM++ extraction floor.
4. **P0, overlap remains untreated.** Optional final-buffer overlap
   analysis does not supply concurrent speaker channels to the live
   diarization. All 184 annotated 100ms overlap frames failed full speaker
   recall. Neither lowering CAM++ cosine thresholds nor relabeling 2 clusters
   can reconstruct an absent simultaneous source channel.
5. **P1, independent internal VAD may suppress boundaries.** Whisper uses
   Silero timestamps to construct its speech windows and can invoke its own
   VAD filtering again inside the model. This is a credible hypothesis for
   short/boundary loss but not established as the TS3003b cause. Separate
   the single-VAD A/B from text-retention changes before promotion.
6. **P1, clustering switches algorithm at 20 examples.** Bootstrap uses
   complete-link under 20 embeddings and legacy FunASR thereafter.
   Reclustering is global, so numeric label remapping and identity history
   can change mid-meeting. The 2026-11-04 live replay included 65 speaker
   relabels: 49 unknown-to-known and 16 known-to-different; only the latter
   is an unambiguous known-label switch. Candidate spherical clustering
   already exists but must be judged against real RTTM, not forced to K=4.
7. **P1, subtitle confirmation is slow.** The recorded live replay had
   roughly 4.87s median and 8.41s p95 confirmed lag. The 7s minimum
   commitment context is a deliberate latency-versus-accuracy tradeoff.
   Changing it requires separate latency and WER testing.
8. **P1, missing pipeline attribution.** Previous diagnostics reported
   final unknown captions but not how often a bounded acoustic second-pass
   succeeded. This branch adds anonymous per-cycle recovery counts at
   session close. These are cycle totals, NOT unique seconds or verified WER.
9. **P1, normal unit tests do not validate acoustic accuracy.** AMI RTTM
   supports DER and coverage, not trustworthy word-by-word WER/CER.
   Transcript correctness, hallucinations and duplication require
   separately verified human text references.
10. **P2, multilingual and production reliability remain unverified.**
    TS3003b is English. The user's 12-language goal requires held-out
    per-language WER/CER, code switching and microphone-specific
    sensitivity tests. A single meeting's DER is not a product-level score.

## What changed in the quality branch

- Speech recovery, retention and short-utterance verifier floors now have
  bounded experiment settings. The production V6 defaults stay unchanged
  until real audio demonstrates a gain.
- The engine feeds the explicit short-voice threshold into both the
  auto-language window path and the fixed-language path, as well as
  the actual secondary ASR decode; no text is fabricated.
- Session-closing anonymous diagnostics include counts of actual ASR
  recovery calls, accepted recovered utterances and failures.
- tools/ami_quality_experiments.py starts one isolated localhost backend
  at a time using sys.executable. It replays the same SHA-256 pinned WAV
  at 1x, scores it, and never overwrites existing artifacts.
- tools/ami_quality_gate.py requires strict DER, missing-time, caption
  coverage, false-alarm, confusion, speaker-count and overlap checks.
  A green unit test or permissive non-regression result cannot be called
  a quality improvement.

## Local Windows commands

First stop any already running meeting backend (so GPU contention is not
mistaken for an algorithm regression). In the repository-root PowerShell:

```powershell
cd E:\Meeting-Fixed10-Test
git fetch origin quality/ts3003b-audio-first-20261104
git switch --track -c quality/ts3003b-audio-first-20261104 origin/quality/ts3003b-audio-first-20261104

$PY = "E:\Meeting-Fixed10-Test\.venv\Scripts\python.exe"
& $PY -m unittest tests.quality.test_audio_first_quality_experiments -v

& $PY -m tools.ami_quality_experiments --profiles control,recovery,single_vad,speaker_candidate
```

If the local quality branch already exists, use git switch without -c;
never use reset --hard, clean -fd or cherry-pick over existing work.

The experiment script starts and stops its own backend on localhost port
18080. No uv installation is required. It checks the frozen
benchmark/fixed195/ami_shadow10/audit_TS3003b_v6_per_speaker.json and
produces distinct, private JSON, log and strict-gate summaries. Exit code
2 means none of the profiles achieved the frozen quality target; this is a
valid negative finding, not an invocation failure.

For explicit initial diagnosis the control uses the unchanged V6 acoustic flags.
The three candidate profiles are:
1. recovery: preserved acoustically supported ASR text, actual bounded
   second-pass decoding, 450ms VAD-based retry/retention and conservative
   100ms short voice support.
2. single_vad: same settings, additionally skipping the second internal
   Whisper VAD stage.
3. speaker_candidate: same settings, adding a spherical clustering
   candidate and 0.9s short CAM++ probe floor.

None forces a known reference speaker count or applies an oracle mapping.
A profile is never auto-activated even if its one-clip gate passes.

## Required further work before promotion

- Repeat the best profile at least twice and verify the same model, WAV
  checksum, machine load and RTTM.
- Evaluate at least several independent AMI meetings with different
  speaker sets. Prefer all ten prepared AMI cases when available.
- Use verified transcripts for WER/CER and check newly recovered words
  against audio before treating increased caption coverage as correctness.
- Benchmark a standalone, overlap-aware diarization model against live
  CAM++, preferably in an isolated environment. The existing Nemotron3
  and pyannote comparison tools are *candidates*, not automatically safe
  production replacements. Never copy anonymous offline speaker IDs into
  live sessions without a validated time and identity alignment.
- Retain the old branch for rollback. Only choose a release profile if
  DER, coverage, precision, overlap and latency meet held-out gates.
