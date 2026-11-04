"""Compare actual Phase 1A ablations on identical immutable AMI audio.

Every acoustic run uses only_words standard reference DER, not the
forced-alignment result. CER/WER are *screening* metrics until mixed-audio
human review. A model candidate is never promoted automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "meeting-phase1a-five-candidate-ablation-v1"


def compare_ablation_inventory(inventory: dict, *, root: Path) -> dict:
    from tools.ami_shadow10_prepare import checksum
    if inventory.get("schema") != SCHEMA:
        raise ValueError("Unknown five-candidate ablation inventory")
    modes = {
        "content_first", "whisper_window_vad_bypass",
        "bounded_vad_gap_recovery", "preserve_verified_unaligned",
        "word_timestamp_bounds",
    }
    rows = inventory.get("candidate_trials")
    if not isinstance(rows, list) or {v.get("variant") for v in rows} != modes:
        raise ValueError("Exactly five distinct frozen Phase 1A modes required")
    full = []
    source_hash = reference_hash = None
    for row in rows:
        score_path = Path(row["independent_der_path"]).resolve()
        if not score_path.is_relative_to(root.resolve()) or not score_path.is_file():
            raise ValueError("Each mode requires a private independent DER report")
        score = json.loads(score_path.read_text(encoding="utf-8"))
        if score.get("schema") != "meeting-ami-independent-nemotron-live-score-v1":
            raise ValueError("Expected independent Nemotron acoustic audit")
        if score.get("meeting") != "TS3003b":
            raise ValueError("Wrong meeting scored")
        audio_hash = score.get("source_audio_sha256")
        ref_hash = score.get("reference_source_sha256")
        if (not isinstance(audio_hash, str) or len(audio_hash) != 64
                or not isinstance(ref_hash, str) or len(ref_hash) != 64):
            raise ValueError("Missing frozen evidence hashes")
        if source_hash is None:
            source_hash, reference_hash = audio_hash, ref_hash
        elif (source_hash, reference_hash) != (audio_hash, ref_hash):
            raise ValueError("Different source audio or original reference")
        acoustic = score.get("independent_acoustic_metrics") or {}
        der = acoustic.get("der")
        if type(der) not in (int, float) or not 0 <= der <= 1:
            raise ValueError("Missing independent full-protocol DER")
        # Time stamps and prefixes must refer to the same replay, but
        # a different model text hypothesis needs a separate reference.
        suffix = score_path.name.removeprefix("audit_TS3003b_nemotron_")
        replay = score_path.parent / ("replay_TS3003b_nemotron_" + suffix)
        if (replay == score_path or not replay.is_file()
                or checksum(replay) != score.get("replay_sha256")):
            raise ValueError("Ablation score is not paired with a pinned live replay")
        metadata = json.loads(replay.read_text(encoding="utf-8"))
        if metadata.get("source_audio_sha256") != audio_hash:
            raise ValueError("Replay audio checksum mismatch")
        text_score = metadata.get("reference_scoring") or {}
        cer = text_score.get("cer")
        wer = text_score.get("wer")
        if cer is not None and (
            type(cer) not in (int, float) or not 0 <= cer <= 1
        ):
            raise ValueError("Malformed reference-backed screening CER")
        full.append({
            "variant": row["variant"],
            "independent_der": der,
            "screening_cer": cer,
            "screening_wer": wer,
            "replay_sha256": score["replay_sha256"],
            "acoustic_report": str(score_path),
            "text_reference_human_verified": (
                inventory.get("reference_human_verified") is True
            ),
        })
    full.sort(key=lambda x: (x["independent_der"],
                              x["screening_cer"] if x["screening_cer"] is not None else 1))
    eligible = [
        row for row in full
        if row["independent_der"] < 0.20
        and row["screening_cer"] is not None
        and row["screening_cer"] < 0.10
        and row["text_reference_human_verified"]
    ]
    return {
        "schema": "meeting-phase1a-paired-ablation-comparison-v1",
        "source_commit": inventory.get("source_commit"),
        "source_audio_sha256": source_hash,
        "reference_source_sha256": reference_hash,
        "ranked_acoustic_variants": full,
        "candidate_meets_both_real_meeting_targets": (
            eligible[0]["variant"] if eligible else None
        ),
        "may_auto_promote": False,
        "note": (
            "Single AMI recording, 100ms original only_words independent "
            "acoustic DER. Text metrics are only valid after independent "
            "human review of the exact mixed audio. An apparent improvement "
            "does not itself validate downstream subtitle completeness."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args(argv)
    private = Path("benchmark/fixed195").resolve()
    inventory, output = args.inventory.resolve(strict=True), args.output.resolve()
    if (not inventory.is_relative_to(private)
            or not output.is_relative_to(private)
            or output.suffix != ".json" or output.exists()):
        p.error("Input must be private and output must be a NEW private JSON")
    report = compare_ablation_inventory(
        json.loads(inventory.read_text(encoding="utf-8")),
        root=inventory.parent,
    )
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["candidate_meets_both_real_meeting_targets"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
