"""Speaker-turn diagnostics that never expose stored embeddings or names."""
from __future__ import annotations


def speaker_stats(segments: list[dict]) -> dict:
    items = sorted((s for s in segments if isinstance(s, dict)),
                   key=lambda s: (int(s.get("start_ms", 0)), int(s.get("end_ms", 0))))
    switches = 0
    overlap_ms = 0
    unknown = 0
    last_speaker = None
    running_end = 0
    for item in items:
        start = max(0, int(item.get("start_ms", 0)))
        end = max(start, int(item.get("end_ms", start)))
        raw_id = item.get("speaker_id")
        speaker = raw_id if isinstance(raw_id, int) and raw_id > 0 else None
        if speaker is None:
            unknown += 1
        if last_speaker is not None and speaker is not None and speaker != last_speaker:
            switches += 1
        if speaker is not None:
            last_speaker = speaker
        # This measures timestamp overlap only, not proven simultaneous speech.
        overlap_ms += max(0, min(running_end, end) - start)
        running_end = max(running_end, end)
    return {"segments": len(items), "speaker_switches": switches,
            "unidentified_segments": unknown, "timestamp_overlap_ms": overlap_ms}


def find_distinct_participant(participants: list[dict], embedding, minimum_margin: float) -> tuple[str | None, float]:
    """Select a participant only if its best similarity clearly beats the runner-up.

    Different voiceprints for the same participant are consolidated first, so
    adding multiple examples of one speaker does not create a false tie.
    """
    if not 0 <= minimum_margin <= 0.5:
        raise ValueError("Speaker match margin must be between 0 and 0.5")
    import numpy as np
    scores = []
    for person in participants:
        similarities = [float(np.dot(embedding, vec["emb"]))
                        for vec in person.get("vectors", [])]
        finite = [x for x in similarities if np.isfinite(x)]
        if finite:
            scores.append((max(finite), str(person["id"])))
    scores.sort(reverse=True)
    if not scores:
        return None, -1.0
    best_score, best_id = scores[0]
    if len(scores) > 1 and best_score - scores[1][0] < minimum_margin:
        return None, best_score
    return best_id, best_score
