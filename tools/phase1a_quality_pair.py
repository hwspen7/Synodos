"""Compare two actual TS3003b replays under identical reference protocols.

Pure evidence reporting. Never treats original AMI words as a verified
microphone transcript and never auto-enables candidate ASR in production.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SCHEMA = "meeting-asr-coverage-paired-evidence-v1"

def compare_pair(baseline: dict, candidate: dict) -> dict:
    for name, package in (("baseline", baseline), ("candidate", candidate)):
        acoustic = package.get("acoustic", {})
        text = package.get("text", {})
        trace = package.get("trace", {})
        if (acoustic.get("schema") != "meeting-ami-independent-nemotron-live-score-v1"
                or text.get("schema") != "meeting-phase1-human-text-acceptance-v1"
                or trace.get("schema") != "meeting-ami-asr-stage-gap-audit-v1"
                or acoustic.get("meeting") != "TS3003b"
                or trace.get("meeting") != "TS3003b"
                or trace.get("reference_protocol") != "only_words"):
            raise ValueError(f"{name} has an incompatible frozen evidence protocol")
        audio = acoustic.get("source_audio_sha256")
        source = acoustic.get("reference_source_sha256")
        if (not isinstance(audio, str) or len(audio) != 64
                or not isinstance(source, str) or len(source) != 64
                or text.get("source_audio_sha256") != audio
                or trace.get("source_audio_sha256") != audio
                or trace.get("reference_source_sha256") != source
                or not isinstance(text.get("human_reference_sha256"), str)):
            raise ValueError(f"{name} lacks consistent audio/reference fingerprints")
    for key in ("source_audio_sha256", "reference_source_sha256"):
        if baseline["acoustic"][key] != candidate["acoustic"][key]:
            raise ValueError(f"Different {key} between experimental replays")
    if baseline["text"]["human_reference_sha256"] != candidate["text"]["human_reference_sha256"]:
        raise ValueError("Different human-text reference hashes")

    def metrics(package: dict) -> dict:
        text = package["text"]
        acoustic = package["acoustic"]
        trace = package["trace"]
        measurements = text.get("measurement") or {}
        numbers = {
            "cer": measurements.get("cer"),
            "wer": measurements.get("wer"),
            "character_deletion_rate": measurements.get("reference_character_deletion_rate"),
            "independent_der": (acoustic.get("independent_acoustic_metrics") or {}).get("der"),
            "published_caption_der": (acoustic.get("published_caption_metrics") or {}).get("der"),
            "uncaptioned_reference_seconds": trace.get("reference_clock_no_caption_seconds"),
        }
        for key, value in numbers.items():
            if (type(value) not in (float, int)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Missing/invalid {key}")
        changes = measurements.get("confirmed_text_changes")
        if type(changes) is not int or changes < 0:
            raise ValueError("Missing confirmed text revision count")
        recovery = trace.get("recovery_counts") or {}
        return {
            **numbers,
            "confirmed_text_changes": changes,
            "complete_pcm_delivery": (text.get("gates") or {}).get("complete_pcm_delivery") is True,
            "independently_verified_text": (text.get("gates") or {}).get("human_reference_confirmed") is True,
            "inner_window_vad_bypass_cycles": trace.get("window_internal_vad_bypassed_cycles", 0),
            "recovery_attempted": recovery.get("recovery_attempted", 0),
            "recovery_confirmed": recovery.get("recovery_confirmed", 0),
        }

    a, b = metrics(baseline), metrics(candidate)
    keys = ("cer", "wer", "character_deletion_rate", "independent_der",
            "published_caption_der", "uncaptioned_reference_seconds")
    delta = {key: round(b[key] - a[key], 6) for key in keys}
    no_accuracy_regression = (
        b["cer"] <= a["cer"]
        and b["wer"] <= a["wer"]
        and b["character_deletion_rate"] <= a["character_deletion_rate"]
        and b["confirmed_text_changes"] == 0
        and b["complete_pcm_delivery"]
    )
    content_coverage_gain = (
        b["cer"] < a["cer"]
        and b["uncaptioned_reference_seconds"] < a["uncaptioned_reference_seconds"]
    )
    return {
        "schema": SCHEMA,
        "meeting": "TS3003b",
        "source_audio_sha256": baseline["acoustic"]["source_audio_sha256"],
        "reference_source_sha256": baseline["acoustic"]["reference_source_sha256"],
        "text_reference_sha256": baseline["text"]["human_reference_sha256"],
        "baseline": a, "quality_candidate": b, "candidate_minus_baseline": delta,
        "candidate_recovery_executed": b["recovery_attempted"] > 0,
        "candidate_inner_window_vad_bypassed": b["inner_window_vad_bypass_cycles"] > 0,
        "same_audio_nonregression_screen": no_accuracy_regression,
        "same_audio_coverage_improvement_screen": content_coverage_gain,
        "human_review_complete": a["independently_verified_text"] and b["independently_verified_text"],
        "production_promotion_authorized": False,
        "phase3_release_accepted": False,
        "note": (
            "These are paired screening observations on one fixed microphone "
            "recording; not a verified quality win, real-time latency result "
            "or permission to change defaults. The first-stage human review, "
            "multi-meeting CER/DER and holdout requirements remain separate."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline-tag", required=True)
    p.add_argument("--candidate-tag", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    root = args.root.resolve(strict=True)
    private = Path("benchmark/fixed195").resolve()
    out = args.output.resolve()
    if (not root.is_relative_to(private) or not out.is_relative_to(private)
            or out.exists() or out.suffix != ".json"
            or any(not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                                  for c in tag) for tag in (
                                      args.baseline_tag, args.candidate_tag))):
        p.error("Use a private frozen root, safe tags and a NEW private JSON output")
    def fetch(tag: str) -> dict:
        names = {
            "acoustic": f"audit_TS3003b_nemotron_{tag}.json",
            "text": f"audit_TS3003b_phase1_human_text_{tag}.json",
            "trace": f"audit_TS3003b_asr_stages_{tag}.json",
        }
        return {
            key: json.loads((root / filename).read_text(encoding="utf-8"))
            for key, filename in names.items()
        }
    result = compare_pair(fetch(args.baseline_tag), fetch(args.candidate_tag))
    if not out.parent.is_dir():
        out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
