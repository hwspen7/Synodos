"""Integrity-checked real-time WebSocket replay of the Fixed150 corpus.

Saves progress after EACH recording; resuming does not replay completed cases.
For a 100+50 set, --scope multi can be used to evaluate meeting speaker recovery
independently of offline 100-case ASR inference.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.evaluation.corpora.fixed10_replay import replay_one, _url
from tools.evaluation.corpora.fixed150_prepare import verify_lock, atomic_json


def replay_path(root: Path) -> Path:
    return root / "replay_results.json"


async def run(root: Path, url: str, scope: str, pace: float, limit: int) -> int:
    lock = verify_lock(root)
    selected = [
        c for c in lock["cases"]
        if scope == "all" or (scope == "multi" and c["kind"] == "multi")
        or (scope == "single" and c["kind"] == "single")
    ]
    path = replay_path(root)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "schema": "meeting-fixed150-replay-v1", "results": {},
    }
    if report.get("schema") != "meeting-fixed150-replay-v1":
        raise ValueError("Wrong replay report schema")
    if not isinstance(report.get("results"), dict):
        raise ValueError("Malformed replay report")
    actual_url = _url(url)
    ran = 0
    for case in selected:
        current = report["results"].get(case["id"])
        if current and current.get("status") == "completed" and current.get("sha256") == case["sha256"]:
            print(f"SKIP verified completed replay: {case['id']}", flush=True)
            continue
        if limit and ran >= limit:
            break
        print(f"REPLAY {case['id']} | {case['scenario']} | {case['duration_s']}s", flush=True)
        try:
            record = await replay_one(case, root, actual_url, pace=pace)
            record["sha256"] = case["sha256"]
        except Exception as exc:
            record = {
                "id": case["id"], "kind": case["kind"], "sha256": case["sha256"],
                "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            }
            print(record["error"], flush=True)
        report["results"][case["id"]] = record
        atomic_json(path, report)
        ran += 1
    good = sum(
        report["results"].get(c["id"], {}).get("status") == "completed"
        and report["results"][c["id"]].get("sha256") == c["sha256"]
        for c in selected
    )
    print(f"REPLAY {scope}: {good}/{len(selected)} with verified WAV SHA and complete server ACK")
    return 0 if good == len(selected) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/audio")
    p.add_argument("--scope", choices=("single", "multi", "all"), default="multi")
    p.add_argument("--pace", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0,
                   help="Optional first N pending recordings, for a controlled smoke run")
    args = p.parse_args()
    if not .25 <= args.pace <= 2 or args.limit < 0:
        p.error("Pace must be 0.25..2 and limit must be >= 0")
    return asyncio.run(run(args.root, args.url, args.scope, args.pace, args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
