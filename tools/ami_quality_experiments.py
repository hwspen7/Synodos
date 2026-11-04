"""Run isolated, real-time AMI A/B experiments without uv or a live UI server.

Each profile launches a NEW localhost backend using this exact Python
interpreter. Candidate settings never mutate the user's shell or source
audio. All reports are exclusive-create under gitignored Fixed195.
A tested profile is NOT applied to production automatically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

from tools.ami_shadow10_prepare import SELECTED, checksum
from tools.ami_shadow10_score import score_case
from tools.ami_quality_gate import assess_quality_gain

PROJECT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = (PROJECT / "benchmark" / "fixed195").resolve()

# Every baseline control is explicit. Inherited shell toggles cannot
# accidentally contaminate the A/B experiment.
CONTROL_ENV = {
    "MEETING_DIARIZATION_DIAGNOSTICS": "1",
    "MEETING_ASR_ENGINE": "whisper",
    "MEETING_ASR_MODEL": "large-v3",
    "MEETING_DIARIZATION_MODE": "cluster",
    "MEETING_SPEAKER_CLUSTER_STRATEGY": "bootstrap",
    "MEETING_SPEAKER_BOOTSTRAP_SIMILARITY": "0.76",
    "MEETING_SPEAKER_TURN_SIMILARITY": "0.58",
    "MEETING_SPEAKER_LABEL_REVISABLE_S": "120",
    "MEETING_MIN_COMMIT_CONTEXT_S": "7.0",
    "CLUSTER_MERGE_THR": "0.78",
    "MEETING_SPEAKER_PROBE_MIN_S": "1.2",
    "MEETING_WHISPER_WINDOW_LID": "1",
    "MEETING_WHISPER_SKIP_INTERNAL_VAD": "0",
    "MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT": "0",
    "MEETING_WHISPER_GAP_RECOVERY": "0",
    "MEETING_WHISPER_RECOVERY_MIN_GAP_MS": "900",
    "MEETING_WHISPER_SHORT_VOICE_SUPPORT_MS": "160",
    "MEETING_ASR_COVERAGE_RETENTION": "0",
    "MEETING_ASR_RETAIN_MIN_GAP_MS": "1000",
    "MEETING_ASR_SPEECH_BACKPRESSURE_GUARD": "0",
    "MEETING_WHISPER_DUAL_VAD": "0",
}

# Candidate #1 recovers words with a second acoustic ASR pass, protects
# untranscribed short VAD intervals, and preserves source text on disputed
# forced alignment. Candidate #2 also removes duplicated VAD filtering.
# Candidate #3 tests acoustic cluster selection and short voiceprints.
RECOVERY_ENV = {
    "MEETING_WHISPER_PRESERVE_UNALIGNED_TEXT": "1",
    "MEETING_WHISPER_GAP_RECOVERY": "1",
    "MEETING_WHISPER_RECOVERY_MIN_GAP_MS": "450",
    "MEETING_WHISPER_SHORT_VOICE_SUPPORT_MS": "100",
    "MEETING_ASR_COVERAGE_RETENTION": "1",
    "MEETING_ASR_RETAIN_MIN_GAP_MS": "450",
    "MEETING_ASR_SPEECH_BACKPRESSURE_GUARD": "1",
}

PROFILES = {
    "control": {},
    "recovery": RECOVERY_ENV,
    "single_vad": {
        **RECOVERY_ENV,
        "MEETING_WHISPER_SKIP_INTERNAL_VAD": "1",
    },
    "speaker_candidate": {
        **RECOVERY_ENV,
        "MEETING_WHISPER_SKIP_INTERNAL_VAD": "1",
        "MEETING_SPEAKER_CLUSTER_STRATEGY": "candidate",
        "MEETING_SPEAKER_PROBE_MIN_S": "0.9",
    },
}


def profile_environment(base: dict[str, str], profile: str) -> dict[str, str]:
    if profile not in PROFILES:
        raise ValueError("Unknown audio-first quality experiment profile")
    result = dict(base)
    result.update(CONTROL_ENV)
    result.update(PROFILES[profile])
    return result


def validate_profiles(value: str) -> list[str]:
    profiles = [part.strip() for part in value.split(",")]
    if not profiles or any(p not in PROFILES for p in profiles):
        raise ValueError("Select comma-separated profiles from " + ", ".join(PROFILES))
    if len(set(profiles)) != len(profiles):
        raise ValueError("Each profile may run only once")
    return profiles


def _port_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _await_ready(server: subprocess.Popen, port: int,
                 timeout_s: float = 300) -> None:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Backend exited early: {server.returncode}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, URLError, TimeoutError):
            pass
        time.sleep(1.0)
    raise RuntimeError("Backend did not become healthy before startup timeout")


def _stop_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=15)


def run_profile(
    *, root: Path, case: dict, profile: str, tag: str, port: int,
    baseline: dict, timeout_s: int,
) -> dict:
    meeting = case["source_meeting"]
    audio = (root / case["audio"]).resolve()
    if not audio.is_relative_to(root) or not audio.is_file():
        raise ValueError("Prepared reference WAV is missing or outside AMI root")
    if checksum(audio) != case.get("sha256"):
        raise ValueError("Source AMI WAV checksum differs from pinned reference")

    replay = root / f"replay_{meeting}_quality_{profile}_{tag}.json"
    audit = root / f"audit_{meeting}_quality_{profile}_{tag}.json"
    log = root / f"server_{meeting}_quality_{profile}_{tag}.log"
    if any(path.exists() for path in (replay, audit, log)):
        raise FileExistsError("Experiment name already exists; refuse overwrite")
    env = profile_environment(os.environ, profile)
    command = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    with log.open("x", encoding="utf-8") as output:
        server = subprocess.Popen(
            command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT,
        )
        try:
            _await_ready(server, port)
            url = (
                f"ws://127.0.0.1:{port}/ws/audio?"
                "language=auto&record_audio=false&translate_to="
            )
            completed = subprocess.run([
                sys.executable, "-m", "tools.live_caption_ws_replay",
                "--audio", str(audio), "--url", url,
                "--loops", "1", "--speed", "1.0",
                "--output", str(replay),
            ], cwd=PROJECT, env=env, timeout=timeout_s, check=False)
            if completed.returncode != 0:
                raise RuntimeError("Live replay failed or reported transport errors")
        finally:
            _stop_server(server)

    if not replay.is_file():
        raise RuntimeError("Real-time replay did not save a complete JSON report")
    replay_data = json.loads(replay.read_text(encoding="utf-8"))
    if (replay_data.get("server_closed") is not True
            or replay_data.get("delivery_matched") is not True
            or replay_data.get("errors") or replay_data.get("flags")
            or replay_data.get("playback_speed") != 1.0
            or replay_data.get("loops") != 1
            or not isinstance(
                (replay_data.get("final_cluster_diagnostics") or {}).get(
                    "anonymous_timeline"
                ), dict)):
        raise RuntimeError("Incomplete live replay; no quality claim permitted")
    scored = score_case(root, meeting, replay)
    with audit.open("x", encoding="utf-8") as dest:
        json.dump(scored, dest, indent=2)
        dest.write("\n")
    result = assess_quality_gain(baseline, scored)
    return {
        "profile": profile,
        "environment_changes": PROFILES[profile],
        "replay": str(replay),
        "audit": str(audit),
        "server_log": str(log),
        "metrics": scored["metrics"],
        "without_captions_s": (
            scored["reference_speech_without_captions"][
                "reference_speech_without_captions_seconds"
            ]
        ),
        "confirmation_lag_p95_ms": (
            replay_data.get("timing_by_elapsed_window") or [{}]
        )[0].get("confirmation_lag_p95_ms"),
        "asr_recovery_diagnostics": (
            replay_data.get("final_cluster_diagnostics") or {}
        ).get("asr_recovery_diagnostics"),
        "quality_gate": result,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path,
                   default=Path("benchmark/fixed195/ami_shadow10"))
    p.add_argument("--meeting", default="TS3003b",
                   choices=[x[0] for x in SELECTED])
    p.add_argument("--baseline", type=Path,
                   default=None)
    p.add_argument("--profiles", default="control,recovery,single_vad,speaker_candidate")
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--timeout-seconds", type=int, default=900)
    args = p.parse_args()

    if importlib.util.find_spec("uvicorn") is None:
        p.error("Run with the existing backend venv Python (uvicorn is missing)")
    if not 1024 <= args.port <= 65535 or args.port == 8000 or not _port_free(args.port):
        p.error("Experiment localhost port is unavailable")
    if not _port_free(8000):
        p.error("Stop the existing backend on port 8000 to avoid GPU contention")
    if not 300 <= args.timeout_seconds <= 3600:
        p.error("Timeout must be between 300 and 3600 seconds")
    profiles = validate_profiles(args.profiles)
    root = args.root.resolve()
    if not root.is_relative_to(PRIVATE_ROOT):
        p.error("AMI experiments must stay in gitignored benchmark/fixed195")
    baseline_path = (args.baseline or
                     (root / f"audit_{args.meeting}_v6_per_speaker.json")).resolve()
    if not baseline_path.is_file():
        p.error(f"Frozen V6 baseline not found: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    case_path = root / "cases" / f"{args.meeting}.json"
    if not case_path.is_file():
        p.error("Prepared AMI case is missing")
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if case.get("source_meeting") != args.meeting:
        p.error("Requested meeting and prepared case differ")
    if (baseline.get("source_audio_sha256") != case.get("sha256")
            or baseline.get("meeting") != args.meeting):
        p.error("Frozen baseline does not refer to the exact prepared WAV")
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = root / f"quality_experiments_{args.meeting}_{tag}.json"
    if summary_path.exists():
        p.error("A prior experiment summary already uses this timestamp")
    results = []
    for profile in profiles:
        print(f"Starting isolated real-time AMI profile: {profile}", flush=True)
        try:
            result = run_profile(
                root=root, case=case, profile=profile, tag=tag,
                port=args.port, baseline=baseline,
                timeout_s=args.timeout_seconds,
            )
            results.append(result)
            print(
                f"{profile}: DER={result['metrics']['der']:.5f}; "
                f"no_caption_s={result['without_captions_s']:.2f}; "
                f"strict_gain={result['quality_gate']['measured_gain_on_this_case']}",
                flush=True,
            )
        except (RuntimeError, ValueError, OSError,
                subprocess.TimeoutExpired) as error:
            results.append({
                "profile": profile, "error": str(error),
                "measured_gain_on_this_case": False,
            })
            print(f"{profile}: experiment FAILED: {error}", flush=True)
    gains = [
        row for row in results
        if row.get("quality_gate", {}).get("measured_gain_on_this_case") is True
    ]
    gains.sort(key=lambda row: (
        row["metrics"]["der"], row["without_captions_s"],
    ))
    report = {
        "schema": "meeting-ami-audio-first-quality-experiments-v1",
        "meeting": args.meeting,
        "source_audio_sha256": case["sha256"],
        "baseline": str(baseline_path),
        "python_executable": sys.executable,
        "profiles": results,
        "qualifying_profiles": [item["profile"] for item in gains],
        "best_pilot_candidate": gains[0]["profile"] if gains else None,
        "production_promoted": False,
        "limits": (
            "A single AMI clip cannot establish multilingual quality; "
            "repeat with independently annotated meetings and WER/CER. "
            "Never auto-activate a measured profile before validation."
        ),
    }
    with summary_path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps({
        "summary_path": str(summary_path),
        "best_pilot_candidate": report["best_pilot_candidate"],
        "qualifying_profiles": report["qualifying_profiles"],
        "production_promoted": False,
    }, indent=2), flush=True)
    return 0 if gains else 2


if __name__ == "__main__":
    raise SystemExit(main())
