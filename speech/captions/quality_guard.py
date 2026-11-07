"""Conservative, language-independent verification of Whisper time-aligned output.

Whisper's text is never classified by its wording. Verification uses Silero VAD
regions, timestamps and optional decoding diagnostics. This cannot prove text
correctness or distinguish real speech from speech-like music/noise.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from speech.contracts import Utterance, Word


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def speech_intervals(regions: Iterable[Mapping], duration_ms: int, sample_rate: int = 16000) -> tuple[tuple[int, int], ...]:
    """Convert VAD sample offsets to a nonoverlapping millisecond union."""
    intervals: list[tuple[int, int]] = []
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        start = _finite(region.get("start"))
        end = _finite(region.get("end"))
        if start is None or end is None:
            continue
        a = max(0, min(duration_ms, int(start * 1000 / sample_rate)))
        b = max(0, min(duration_ms, int(end * 1000 / sample_rate)))
        if b > a:
            intervals.append((a, b))
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((a, b) for a, b in merged)


def voice_ms(start: int, end: int, intervals: tuple[tuple[int, int], ...]) -> int:
    return sum(max(0, min(end, b) - max(start, a)) for a, b in intervals)


def _unreliable(piece: object) -> bool:
    """Apply metadata checks only when the actual model supplies diagnostics."""
    no_speech = _finite(getattr(piece, "no_speech_prob", None))
    logprob = _finite(getattr(piece, "avg_logprob", None))
    compression = _finite(getattr(piece, "compression_ratio", None))
    if no_speech is not None and logprob is not None and no_speech > 0.90 and logprob < -0.7:
        return True
    if compression is not None and compression > 3.0:
        return True
    return False


def _word_items(piece: object, duration_ms: int) -> list[Word]:
    result = []
    for word in getattr(piece, "words", None) or ():
        start = _finite(getattr(word, "start", None))
        end = _finite(getattr(word, "end", None))
        value = str(getattr(word, "word", "") or "")
        if not value.strip() or start is None or end is None:
            continue
        a, b = round(start * 1000), round(end * 1000)
        if a < 0 or b > duration_ms + 250:
            continue
        b = min(b, duration_ms)
        if b > a:
            result.append(Word(value, a, b))
    return result


def verify_utterances(
    pieces: Iterable[object],
    vad_regions: Iterable[Mapping],
    duration_ms: int,
    detected_language: str,
    *,
    sample_rate: int = 16000,
    preserve_disputed_text: bool = False,
    prefer_word_bounds: bool = False,
    allow_short_vad_supported: bool = False,
) -> tuple[Utterance, ...]:
    """Reject unsupported chunks and omit words positioned in nonspeech areas.

    A short chunk needs at least 160 ms of voiced evidence; a longer chunk
    needs at least 35% VAD overlap. Words lacking speech support are excluded;
    accepted words split at long gaps so silence is not represented as speech.

    A guarded opt-in preserves the original segment text when word alignment
    disagrees with already-verified segment acoustic support. It discards
    unreliable word timestamps instead of inventing word-level precision.
    Segment-level evidence checks and decoding diagnostic rejection remain
    mandatory. Under a separate quality-candidate profile, a 100-159 ms
    speech-supported reply can survive if most of its duration is VAD-positive.
    The original reference profile is unchanged pending real A/B testing.
    """
    if duration_ms <= 0:
        return ()
    intervals = speech_intervals(vad_regions, duration_ms, sample_rate)
    if not intervals:
        return ()
    accepted: list[Utterance] = []
    for piece in pieces:
        text = str(getattr(piece, "text", "") or "").strip()
        first = _finite(getattr(piece, "start", None))
        last = _finite(getattr(piece, "end", None))
        if not text or first is None or last is None or _unreliable(piece):
            continue
        start, end = round(first * 1000), round(last * 1000)
        if start < 0 or end > duration_ms + 250 or end <= start or end - start < 100:
            continue
        end = min(end, duration_ms)
        duration = end - start
        evidence = voice_ms(start, end, intervals)
        # A short reply can be shorter than the default 160 ms voice floor.
        # Only the opt-in candidate relaxes that floor, and it requires
        # stronger VAD coverage than an ordinary model segment. Do not
        # bypass _unreliable(), minimum model duration or silence checks.
        short_reply = (
            allow_short_vad_supported
            and 100 <= duration <= 450
            and evidence >= 100
            and evidence / duration >= 0.60
        )
        if not short_reply and (
            evidence < 160
            or evidence / duration < (0.35 if duration >= 2500 else 0.25)
        ):
            continue
        source_words = tuple(getattr(piece, "words", None) or ())
        words = _word_items(piece, duration_ms)
        # A partially invalid word alignment must not be advertised as a
        # complete alignment to the model's text. Preserve only segment timing.
        if preserve_disputed_text and len(words) != len(source_words):
            accepted.append(Utterance(text, start, end, detected_language, (), "segment"))
            continue
        if not words:
            if not source_words:
                accepted.append(Utterance(text, start, end, detected_language, (), "segment"))
            continue
        voiced: list[Word] = []
        for word in words:
            width = word.end_ms - word.start_ms
            support = voice_ms(word.start_ms, word.end_ms, intervals)
            if support >= max(30, min(120, round(width * 0.30))):
                voiced.append(word)
        if not voiced:
            if preserve_disputed_text:
                accepted.append(Utterance(text, start, end, detected_language, (), "segment"))
            continue
        # Preserve model punctuation; reconstruct only when words were removed.
        # Split long silent gaps to prevent a single segment from spanning silence.
        runs: list[list[Word]] = []
        for word in voiced:
            if not runs or word.start_ms - runs[-1][-1].end_ms >= 1200:
                runs.append([word])
            else:
                runs[-1].append(word)
        all_accepted = len(voiced) == len(source_words) and len(runs) == 1
        if preserve_disputed_text and not all_accepted:
            # Trimming words can turn imperfect alignment into missing meeting
            # content. Keep the model's wording while explicitly lowering
            # timestamp provenance; never retain a partial word alignment.
            accepted.append(Utterance(text, start, end, detected_language, (), "segment"))
            continue
        for run in runs:
            raw = text if all_accepted else "".join(word.text for word in run).strip()
            if not raw:
                continue
            run_start, run_end = run[0].start_ms, run[-1].end_ms
            # The legacy policy retains the original segment bounds, even
            # when all words were accepted. In the opt-in, word-aligned
            # caption policy, trim only to the first/last verified word:
            # no transcript text or intermediate word boundary is changed.
            a, b = (
                (run_start, run_end)
                if prefer_word_bounds or not all_accepted
                else (start, end)
            )
            if b > a:
                accepted.append(Utterance(raw, a, b, detected_language,
                                          tuple(run), "word"))
    return tuple(accepted)
