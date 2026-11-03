"""Non-secret environment diagnostics for Meeting Phase 06c."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

SAFE_VARIABLES = (
    "MEETING_ASR_ENGINE", "MEETING_ASR_MODEL", "MEETING_ASR_DEVICE",
    "MEETING_ASR_COMPUTE", "MEETING_ASR_LANGUAGE",
    "MEETING_WHISPER_DUAL_VAD", "MEETING_WHISPER_WINDOW_LID",
    "MEETING_SAFE_SILENCE_RECLAIM", "MEETING_QWEN_FORCED_ALIGNER",
    "MEETING_STORAGE_BACKEND", "MEETING_AI_ENABLED", "MEETING_AI_PROVIDER",
)


def describe(env: dict[str, str]) -> dict:
    result = {key: env.get(key, "<not set>") for key in SAFE_VARIABLES}
    result["DATABASE_URL"] = "configured" if env.get("DATABASE_URL") else "not configured"
    result["MEETING_AI_API_KEY"] = "configured" if env.get("MEETING_AI_API_KEY") else "not configured"
    return result


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    checks = {
        "python": sys.version.split()[0],
        "project": str(project),
        "configuration": describe(dict(os.environ)),
        "packages": {name: importlib.util.find_spec(name) is not None for name in
                     ("numpy", "faster_whisper", "webrtcvad", "qwen_asr", "psycopg")},
        "files": {name: (project / name).is_file() for name in
                  ("speech/multilingual_windows.py", "speech/silence_retention.py",
                   "speech/audio_windows.py", "tools/asr_benchmark_06c.py")},
    }
    print(json.dumps(checks, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
