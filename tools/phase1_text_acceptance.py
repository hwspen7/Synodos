"""Fail-closed Phase 1 text-integrity gate for a real, referenced replay.

The WebSocket replay must be scored with an independently checked human
transcript from the *same* source audio. No language model estimates truth.
All results are aggregate-only. The archived audio fingerprint and human
transcript fingerprint are frozen in each score before acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

SCHEMA = "meeting-phase1-human-text-acceptance-v1"
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def evaluate_phase1_text(
    replay: dict, *, expected_audio_sha256: str,
    expected_reference_sha256: str,
    human_reference_verified: bool = False,
    max_cer: float = 0.10,
    max_wer: float = 0.20,
    max_deletion_rate: float = 0.03,
) -> dict:
    """Verify provenance, no confirmed-text churn, WER/CER and deletion rate.

    A score without an independent human-approved reference cannot
    establish actual content fidelity. WER is optional for unsegmented CJK.
    A null error-operation breakdown fails closed instead of being zero.
    """
    if (not isinstance(replay, dict)
            or not SHA256.fullmatch(expected_audio_sha256)
            or not SHA256.fullmatch(expected_reference_sha256)
            or any(type(x) not in (float, int) or not 0 <= x <= 1
                   for x in (max_cer, max_wer, max_deletion_rate))):
        raise ValueError("Invalid reference pin or error rate threshold")
    scores = replay.get("reference_scoring")
    if not isinstance(scores, dict):
        scores = {}
    audio = replay.get("source_audio_sha256")
    reference = replay.get("reference_text_sha256")
    # Support the original replay's canonical checksum field only.
    if not isinstance(audio, str) or not isinstance(reference, str):
        raise ValueError("Pinned source audio and human transcript hashes required")
    if audio != expected_audio_sha256 or reference != expected_reference_sha256:
        raise ValueError("Audio or reference transcript checksum mismatch")
    cer = scores.get("cer")
    wer = scores.get("wer")
    char_deletion = scores.get("reference_character_deletion_rate")
    word_deletion = scores.get("reference_word_deletion_rate")
    if not isinstance(cer, (int, float)) or isinstance(cer, bool):
        raise ValueError("Missing reference-backed CER")
    if wer is not None and (
        not isinstance(wer, (int, float)) or isinstance(wer, bool)
    ):
        raise ValueError("Malformed WER")
    if (char_deletion is not None and
            (not isinstance(char_deletion, (int, float))
             or isinstance(char_deletion, bool))):
        raise ValueError("Malformed character deletion rate")
    if (word_deletion is not None and
            (not isinstance(word_deletion, (int, float))
             or isinstance(word_deletion, bool))):
        raise ValueError("Malformed word deletion rate")
    deletion = word_deletion if wer is not None else char_deletion
    # Unknown metrics never pass thresholds.
    gates = {
        "human_reference_confirmed": human_reference_verified is True,
        "complete_pcm_delivery": (
            replay.get("server_closed") is True
            and replay.get("delivery_matched") is True
            and type(replay.get("sent_samples")) is int
            and replay.get("audio_received_samples") == replay["sent_samples"]
            and not replay.get("errors")
            and not replay.get("flags")
        ),
        "confirmed_segments_never_disappeared": (
            replay.get("confirmed_segment_disappearances") == 0
        ),
        "confirmed_text_not_silently_revised": (
            replay.get("confirmed_text_changes") == 0
        ),
        "character_error_at_most_threshold": (
            0 <= cer <= max_cer
        ),
        "word_error_at_most_threshold_when_evaluable": (
            wer is None or 0 <= wer <= max_wer
        ),
        "reference_deletion_at_most_threshold": (
            deletion is not None and 0 <= deletion <= max_deletion_rate
        ),
        "reference_scoring_protocol_present": (
            replay.get("subtitle_completeness")
            == "reference_scored_manual_review_still_required"
            and (wer is None or scores.get("reference_words") is not None)
        ),
    }
    return {
        "schema": SCHEMA,
        "pass": all(gates.values()),
        "gates": gates,
        "measurement": {
            "cer": cer,
            "wer": wer,
            "reference_character_deletion_rate": char_deletion,
            "reference_word_deletion_rate": word_deletion,
            "confirmed_segment_disappearances": replay.get(
                "confirmed_segment_disappearances"
            ),
            "confirmed_text_changes": replay.get(
                "confirmed_text_changes"
            ),
        },
        "thresholds": {
            "max_cer": max_cer, "max_wer": max_wer,
            "max_deletion_rate": max_deletion_rate,
        },
        "source_audio_sha256": audio,
        "human_reference_sha256": reference,
        "note": (
            "This requires a genuinely checked reference of the same audio. "
            "CER/WER and edit-alignment deletion rates do not prove there "
            "are no missing meaningful sentences, especially in overlap. "
            "No source words, transcript or personally identifying speaker "
            "details are included in this report."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--audio-sha256", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--human-reference-verified", action="store_true")
    parser.add_argument("--max-cer", type=float, default=0.10)
    parser.add_argument("--max-wer", type=float, default=0.20)
    parser.add_argument("--max-deletion-rate", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = Path("benchmark/fixed195").resolve()
    replay, destination = args.replay.resolve(), args.output.resolve()
    if (not replay.is_relative_to(private) or not replay.is_file()
            or not destination.is_relative_to(private)
            or destination.suffix.lower() != ".json"
            or destination.exists()
            or replay == destination):
        parser.error("Use existing private replay and NEW private JSON report")
    result = evaluate_phase1_text(
        json.loads(replay.read_text(encoding="utf-8")),
        expected_audio_sha256=args.audio_sha256.lower(),
        expected_reference_sha256=args.reference_sha256.lower(),
        human_reference_verified=args.human_reference_verified,
        max_cer=args.max_cer, max_wer=args.max_wer,
        max_deletion_rate=args.max_deletion_rate,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
