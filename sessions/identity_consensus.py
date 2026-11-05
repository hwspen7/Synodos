"""Fail-closed optional CAM++ / ECAPA identity consensus.

Each backend independently performs its OWN embedding comparison. Cosine
scores from different embedding networks must never be averaged, and two
agreements are not proof of identity unless enrollment and human review
are independently validated.
"""
from __future__ import annotations

from collections.abc import Mapping


MODELS = ("camplus", "ecapa")


def reconcile_identity_decisions(
    decisions: Mapping[str, dict],
) -> dict:
    """Return a candidate only if two independently matched models agree.

    The result is read-only acoustic evidence; it cannot automatically
    write speaker labels, enroll voiceprints or alter production thresholds.
    """
    if not isinstance(decisions, Mapping) or set(decisions) - set(MODELS):
        raise ValueError("Only independent CAM++ and ECAPA matching are supported")
    matched = {}
    observed = {}
    for model in MODELS:
        result = decisions.get(model)
        if not isinstance(result, dict):
            observed[model] = "not_evaluated"
            continue
        status = result.get("status")
        claimed_model = result.get("embedding_model")
        if claimed_model != model:
            raise ValueError("Identity model provenance mismatch")
        if status not in (
            "matched", "ambiguous", "below_threshold",
            "invalid_probe", "no_compatible_enrollment",
        ):
            raise ValueError("Unknown matcher status")
        profile = result.get("profile_id")
        if (status == "matched" and
                (not isinstance(profile, str) or not profile)):
            raise ValueError("Matched voice lacks a valid enrolled profile")
        if status != "matched" and profile is not None:
            raise ValueError("Unmatched voice cannot have a verified profile")
        if status == "matched":
            if result.get("enrollment_human_verified") is not True:
                # A high unverified cosine match cannot become stronger
                # evidence merely because two networks happen to agree.
                observed[model] = "unverified_enrollment"
                continue
            if result.get("score_kind") != "raw_cosine_not_probability":
                raise ValueError("Identity candidate score provenance mismatch")
            matched[model] = profile
        observed[model] = status

    if len(matched) == 2:
        if len(set(matched.values())) == 1:
            status = "two_model_consensus"
            candidate = next(iter(matched.values()))
        else:
            status = "model_disagreement"
            candidate = None
    elif matched:
        status = "single_model_only"
        candidate = None
    else:
        status = "insufficient_evidence"
        candidate = None
    return {
        "status": status,
        "profile_id": candidate,
        "model_status": observed,
        "compared_models": list(MODELS),
        "raw_scores_combined": False,
        "automatic_identity_update": False,
        "human_review_required": True,
    }
