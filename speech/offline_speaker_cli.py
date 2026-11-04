"""Dedicated pyannote process entrypoint; never run during live ASR."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from speech.offline_speaker import (
    _pipeline, normalize_turns,
)


def main(argv: list[str] | None = None) -> int:
    inputs = list(sys.argv[1:] if argv is None else argv)
    if len(inputs) != 1:
        print(json.dumps({"error": "Expected one WAV file"}))
        return 2
    file = Path(inputs[0]).resolve()
    if not file.is_file() or file.suffix.casefold() != ".wav":
        print(json.dumps({"error": "WAV file unavailable"}))
        return 2
    try:
        model = _pipeline()
        outcome = model(str(file))
        # Preserve full diarization, including 2+ simultaneous speakers.
        # The exclusive view remains separately available for word alignment.
        annotation = getattr(outcome, "speaker_diarization", None)
        if annotation is None:
            annotation = getattr(outcome, "exclusive_speaker_diarization", outcome)
        turns = normalize_turns(annotation)
        exclusive = getattr(outcome, "exclusive_speaker_diarization", None)
        print(json.dumps({
            "speaker_turns": turns,
            "exclusive_speaker_turns": (
                normalize_turns(exclusive) if exclusive is not None else None
            ),
        }, ensure_ascii=True))
        return 0
    except Exception as exc:
        # Print no traceback, file contents or access tokens in the API job.
        print(json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:200]}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
