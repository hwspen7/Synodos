"""Freeze synthetic reports and evaluate separate real-audio/release evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

from tools.meeting_knowledge_contract import (
    FREEZE_SCHEMA, load_json, release_decision, sha256_file,
    strict_synthetic, write_new,
)


def freeze(args):
    lexical = load_json(args.lexical)
    hybrid = load_json(args.hybrid)
    strict_synthetic(lexical, "lexical")
    strict_synthetic(hybrid, "hybrid")
    if lexical["text_model"] != hybrid["text_model"]:
        raise ValueError("Model changed between lexical and hybrid runs")
    if args.output.exists():
        raise ValueError("Baseline already exists. Never overwrite a frozen run")
    copies = {}
    base = args.output.parent / (args.output.stem + "_inputs")
    if base.exists():
        raise ValueError("Frozen input directory already exists")
    base.mkdir(parents=True, exist_ok=False)
    for mode, path in (("lexical", args.lexical), ("hybrid", args.hybrid)):
        dest = base / (mode + ".json")
        shutil.copyfile(path, dest)
        copies[mode] = {"filename": dest.name, "sha256": sha256_file(dest)}
    manifest = {
        "schema": FREEZE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        "scope": "synthetic_local_ollama_only",
        "text_model": lexical["text_model"],
        "embedding_model": hybrid["embedding_model"],
        "reports": copies,
        "real_audio_pass": False,
        "human_semantic_truth_review": False,
    }
    write_new(args.output, manifest)
    return 0


def decide(args):
    freeze_data = load_json(args.baseline)
    real_lexical = load_json(args.real_lexical) if args.real_lexical and args.real_lexical.exists() else None
    real_hybrid = load_json(args.real_hybrid) if args.real_hybrid and args.real_hybrid.exists() else None
    phase12 = load_json(args.phase12) if args.phase12 and args.phase12.exists() else None
    ci = load_json(args.ci) if args.ci and args.ci.exists() else None
    result = release_decision(freeze_data, real_lexical, real_hybrid,
                              phase12, ci, args.commit)
    # Freeze source consistency and reject replacing earlier input reports.
    checked = {}
    for mode, row in (freeze_data.get("reports") or {}).items():
        file = args.baseline.parent / (args.baseline.stem + "_inputs") / row.get("filename", "")
        checked[mode] = (file.is_file() and sha256_file(file) == row.get("sha256"))
        if checked[mode]:
            try:
                strict_synthetic(load_json(file), mode)
            except (ValueError, TypeError, KeyError):
                checked[mode] = False
    result["gates"]["frozen_input_sha256_intact"] = (
        set(checked) == {"lexical", "hybrid"} and all(checked.values())
    )
    result["local_synthetic_model_pass"] &= result["gates"]["frozen_input_sha256_intact"]
    result["knowledge_release_pass"] &= result["gates"]["frozen_input_sha256_intact"]
    result["overall_product_release_pass"] = all(result["gates"].values())
    write_new(args.output, result)
    print("MEETING KNOWLEDGE REAL RELEASE:", "PASS" if result["knowledge_release_pass"] else "NOT_ACCEPTED")
    print("OVERALL PRODUCT RELEASE:", "PASS" if result["overall_product_release_pass"] else "BLOCKED")
    print("REPORT:", args.output)
    return 0 if result["overall_product_release_pass"] else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("freeze")
    a.add_argument("--lexical", type=Path, required=True)
    a.add_argument("--hybrid", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    b = sub.add_parser("decide")
    for name in ("baseline", "real-lexical", "real-hybrid", "phase12", "ci"):
        b.add_argument("--" + name, type=Path, required=(name == "baseline"))
    b.add_argument("--commit", required=True)
    b.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        return freeze(args) if args.action == "freeze" else decide(args)
    except Exception as exc:
        print("MEETING KNOWLEDGE RELEASE NOT_ACCEPTED:", str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
