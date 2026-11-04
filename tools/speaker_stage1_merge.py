"""Join speaker verification results from separate Python model environments.

A shared manifest checksum ensures paired inputs. This is a report merge,
not a method for comparing incompatible speaker embedding vectors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.speaker_stage1_compare import SCHEMA


def merge_verification(camplus: dict, ecapa: dict) -> dict:
    if any(item.get("schema") != SCHEMA for item in (camplus, ecapa)):
        raise ValueError("Incompatible stage-one comparison schema")
    digest = camplus.get("manifest_sha256")
    if not isinstance(digest, str) or not digest or digest != ecapa.get("manifest_sha256"):
        raise ValueError("Cannot compare different or unversioned manifests")
    if camplus.get("reference_status") != ecapa.get("reference_status"):
        raise ValueError("Reference statuses disagree")
    left = camplus.get("verification") or {}
    right = ecapa.get("verification") or {}
    if set(left.get("models", {})) != {"camplus"}:
        raise ValueError("The first report must contain CAM++ only")
    if set(right.get("models", {})) != {"ecapa"}:
        raise ValueError("The second report must contain ECAPA only")
    if left.get("pair_count") != right.get("pair_count"):
        raise ValueError("Verification pair counts differ")
    if (not isinstance(left.get("pair_count"), int)
            or left["pair_count"] < 2):
        raise ValueError("At least two labeled trials are required")
    model_results = {**left["models"], **right["models"]}
    complete = all(
        item.get("completed") == left["pair_count"]
        and item.get("failed") == 0
        and (item.get("metrics") or {}).get("eer") is not None
        for item in model_results.values()
    )
    return {
        "schema": "meeting-speaker-stage1-paired-verification-v1",
        "manifest_sha256": digest,
        "pair_count": left["pair_count"],
        "models": model_results,
        "paired_comparison_complete": complete,
        "reference_status": camplus["reference_status"],
        "candidate_models_not_promoted": True,
        "note": (
            "Scores use separate within-model embedding spaces and the same "
            "frozen input pairs. EER is post-hoc and does not certify any "
            "deployment verification threshold or user identity."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camplus-report", type=Path, required=True)
    parser.add_argument("--ecapa-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing comparison")
    if args.output.resolve() in {
        args.camplus_report.resolve(), args.ecapa_report.resolve()
    }:
        parser.error("Output must not replace either input")
    left = json.loads(args.camplus_report.read_text(encoding="utf-8"))
    right = json.loads(args.ecapa_report.read_text(encoding="utf-8"))
    report = merge_verification(left, right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as dest:
        json.dump(report, dest, indent=2)
        dest.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["paired_comparison_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
