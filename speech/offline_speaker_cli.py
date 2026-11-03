"""Dedicated pyannote process entrypoint; never run during live ASR."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from speech.offline_speaker import _pipeline, normalize_turns


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
        annotation = getattr(outcome, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(outcome, "speaker_diarization", outcome)
        turns = normalize_turns(annotation)
        print(json.dumps({"speaker_turns": turns}, ensure_ascii=True))
        return 0
    except Exception as exc:
        # Print no traceback, file contents or access tokens in the API job.
        print(json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:200]}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
