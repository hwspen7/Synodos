"""Evidence-only dashboard for Fixed150 multilingual ASR + speaker diarization.

Scores 100 individually referenced utterances and 50 separate two-/three-speaker
recordings. Never turns missing predictions, failed replays or unmeasured times
into a successful quality metric.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from html import escape
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.fixed150_prepare import verify_lock, atomic_json
from tools.phase09_speaker_benchmark import evaluate


def token_bag(text: str) -> Counter:
    return Counter(re.findall(r"[^\W_]{3,}", unicodedata.normalize("NFKC", text).casefold()))


def token_recall(reference: str, predicted: str) -> float | None:
    a, b = token_bag(reference), token_bag(predicted)
    if not a:
        return None
    return sum(min(v, b[k]) for k, v in a.items()) / sum(a.values())


def summarize_singles(lock: dict, report: dict | None) -> dict:
    cases = [c for c in lock["cases"] if c["kind"] == "single"]
    if report is None:
        return {"status": "NOT RUN", "valid": 0, "expected": 100}
    if not report.get("infer") or report.get("samples") != 100:
        return {"status": "BLOCKED", "reason": "Expected actual --infer results for 100 cases",
                "valid": 0, "expected": 100}
    outputs = {}
    for engine, part in report.get("engines", {}).items():
        rows = part.get("samples", [])
        if len(rows) != 100:
            outputs[engine] = {"status": "BLOCKED", "reason": "Incomplete result array",
                               "valid": 0, "expected": 100}
            continue
        valid = []
        failures = []
        for i, (case, row) in enumerate(zip(cases, rows)):
            if row.get("index") != i or row.get("error") or not row.get("reference_characters"):
                failures.append(case["id"])
            else:
                valid.append((case, row))
        if failures:
            outputs[engine] = {"status": "BLOCKED", "valid": len(valid), "expected": 100,
                               "failed_case_ids": failures}
            continue
        group = defaultdict(list)
        for case, row in valid:
            group[case["language"]].append(row)
        per_language = []
        for code, values in sorted(group.items()):
            chars = sum(x["reference_characters"] for x in values)
            tokens = sum(x["reference_tokens"] for x in values)
            per_language.append({
                "language": code, "samples": len(values),
                "cer": sum(x["character_edits"] for x in values) / chars,
                "mixed_token_er": sum(x["token_edits"] for x in values) / max(1, tokens),
                "lid": sum(x["expected_language"] == x["predicted_language"]
                           for x in values) / len(values),
            })
        time_ms = [x["latency_ms"] for _, x in valid]
        if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in time_ms):
            outputs[engine] = {"status": "BLOCKED", "valid": 100,
                               "reason": "A measured inference latency is missing"}
            continue
        ordered = sorted(time_ms)
        outputs[engine] = {
            "status": "MEASURED", "valid": 100, "expected": 100,
            "cer": sum(x["character_edits"] for _, x in valid) /
                   sum(x["reference_characters"] for _, x in valid),
            "mixed_token_er": sum(x["token_edits"] for _, x in valid) /
                              max(1, sum(x["reference_tokens"] for _, x in valid)),
            "language_accuracy": sum(x["expected_language"] == x["predicted_language"]
                                     for _, x in valid) / 100,
            "latency_p50_ms": statistics.median(time_ms),
            "latency_p95_ms": ordered[math.ceil(len(ordered) * .95) - 1],
            "latency_max_ms": ordered[-1],
            "per_language": per_language,
            "per_scenario": {
                key: {
                    "samples": len(items),
                    "cer": sum(x["character_edits"] for x in items) /
                           sum(x["reference_characters"] for x in items),
                }
                for key, items in {
                    scenario: [row for case, row in valid if case["scenario"] == scenario]
                    for scenario in sorted({c["scenario"] for c, _ in valid})
                }.items()
            },
        }
    if not outputs:
        return {"status": "BLOCKED", "reason": "No model predictions", "engines": {}}
    return {"status": ("MEASURED" if all(o["status"] == "MEASURED" for o in outputs.values())
                       else "BLOCKED"), "engines": outputs}


def _verified_prediction(case: dict, report: dict | None) -> tuple[list, str | None]:
    if report is None:
        return [], "not replayed"
    record = report.get("results", {}).get(case["id"])
    if record is None:
        return [], "not replayed"
    if (record.get("status") != "completed" or record.get("sha256") != case["sha256"] or
        record.get("acknowledged_samples") != record.get("sent_samples") or
        not isinstance(record.get("segments"), list)):
        return [], "replay failure or unverifiable source"
    return record["segments"], None


def _valid_segments(parts: list[dict]) -> bool:
    if not parts:
        return True
    return all(isinstance(p, dict) and
               isinstance(p.get("text"), str) and
               isinstance(p.get("start_ms"), (int, float)) and
               isinstance(p.get("end_ms"), (int, float)) and
               0 <= p["start_ms"] < p["end_ms"] and
               str(p.get("speaker_id", "")).strip()
               for p in parts)


def summarize_multis(lock: dict, report: dict | None) -> dict:
    cases = [c for c in lock["cases"] if c["kind"] == "multi"]
    results = []
    for case in cases:
        parts, err = _verified_prediction(case, report)
        if err or not _valid_segments(parts):
            results.append({"id": case["id"], "scenario": case["scenario"],
                            "status": "BLOCKED", "reason": err or "invalid prediction intervals"})
            continue
        annotated = [
            {"start_ms": int(s["start_ms"]), "end_ms": int(s["end_ms"]),
             "speaker": str(s["speaker"])}
            for s in case["reference_segments"]
            if int(s["end_ms"]) > int(s["start_ms"])
        ]
        predictions = [
            {"start_ms": int(p["start_ms"]), "end_ms": int(p["end_ms"]),
             "speaker_id": str(p["speaker_id"]) if str(p["speaker_id"]).lower() not in
                           ("unknown", "und", "none", "0") else "-1"}
            for p in parts
        ]
        try:
            diar = evaluate(annotated, predictions)
            mapping = diar["speaker_mapping"]
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            results.append({"id": case["id"], "scenario": case["scenario"],
                            "status": "BLOCKED", "reason": f"DER: {type(exc).__name__}"})
            continue
        total_hypothesis = " ".join(p["text"] for p in parts)
        recalls = []
        oracle_recalls = []
        for speaker in case["speakers"]:
            transcript = " ".join(
                p["text"] for p in parts
                if mapping.get(str(p["speaker_id"])) == str(speaker["speaker"])
            )
            recalls.append(token_recall(speaker["reference"], transcript))
            oracle_recalls.append(token_recall(speaker["reference"], total_hypothesis))
        results.append({
            "id": case["id"], "scenario": case["scenario"], "status": "MEASURED",
            "speaker_count_expected": len(case["speakers"]),
            "speaker_count_observed": len({
                str(x["speaker_id"]) for x in predictions
                if x["speaker_id"] not in ("-1", "0")
            }),
            "der": diar["der"], "attributed_content_recall": recalls,
            "unattributed_content_recall": oracle_recalls,
            "overlap_ratio": case.get("overlap_ratio"),
            "annotation_precision": case["annotation_precision"],
        })
    grouped = {}
    for scenario in sorted(set(c["scenario"] for c in cases)):
        subset = [r for r in results if r["scenario"] == scenario]
        complete = [r for r in subset if r["status"] == "MEASURED"]
        rates = [r["der"] for r in complete if r["der"] is not None]
        attrib = [v for r in complete for v in r["attributed_content_recall"]
                  if v is not None]
        oracle = [v for r in complete for v in r["unattributed_content_recall"]
                  if v is not None]
        grouped[scenario] = {
            "count": len(subset), "measured": len(complete),
            "median_der": statistics.median(rates) if rates else None,
            "mean_attributed_content_recall": statistics.mean(attrib) if attrib else None,
            "mean_unattributed_content_recall": statistics.mean(oracle) if oracle else None,
        }
    return {"status": "MEASURED" if all(r["status"] == "MEASURED" for r in results)
            else "BLOCKED", "expected": 50,
            "valid": sum(r["status"] == "MEASURED" for r in results),
            "groups": grouped, "cases": results}



def summarize_stress(root: Path, original: dict | None) -> dict:
    """Paired CER degradation, NOT 40 independent source recordings."""
    spec, output = root / "stress_lock.json", root / "stress_asr.json"
    if not spec.exists() and not output.exists():
        return {"status": "NOT RUN", "expected": 40}
    if not spec.exists() or not output.exists() or original is None:
        return {"status": "BLOCKED", "reason": "Missing stress references or real model predictions"}
    from tools.fixed150_prepare import verify_case
    cases = json.loads(spec.read_text(encoding="utf-8")).get("cases", [])
    report = json.loads(output.read_text(encoding="utf-8"))
    if len(cases) != 40 or report.get("samples") != 40 or not report.get("infer"):
        return {"status": "BLOCKED", "reason": "Need 40 actual synthetic-stress results"}
    for case in cases:
        verify_case(root, case)
    engines = {}
    for name, data in report.get("engines", {}).items():
        rows = data.get("samples", [])
        bases = original.get("engines", {}).get(name, {}).get("samples", [])
        if len(rows) != 40 or len(bases) != 100 or any(
            row.get("error") or row.get("index") != i for i, row in enumerate(rows)
        ) or any(row.get("error") for row in bases):
            engines[name] = {"status": "BLOCKED"}
            continue
        grouped = defaultdict(list)
        for case, row in zip(cases, rows):
            old = bases[int(case["base_id"].split("-")[-1]) - 1]
            grouped[case["condition"]].append(row["cer"] - old["cer"])
        engines[name] = {
            "status": "MEASURED",
            "paired_sources": 8,
            "conditions": {k: {"count": len(v), "mean_cer_change": statistics.mean(v)}
                           for k, v in sorted(grouped.items())},
        }
    return {"status": "MEASURED" if engines and all(
        x["status"] == "MEASURED" for x in engines.values()) else "BLOCKED",
            "expected": 40, "engines": engines}


def build_score(root: Path) -> dict:
    lock = verify_lock(root)
    single_path = root / "single_asr.json"
    replay_path = root / "replay_results.json"
    single = json.loads(single_path.read_text(encoding="utf-8")) if single_path.exists() else None
    replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else None
    return {
        "schema": "meeting-fixed150-scorecard-v1",
        "recordings": 150,
        "independent_source_recordings": 150,
        "original_language_count": len({c["language"] for c in lock["cases"] if c["kind"] == "single"}),
        "single": summarize_singles(lock, single),
        "multi": summarize_multis(lock, replay),
        "stress": summarize_stress(root, single),
        "release_certified": False,
    }


def _panel(title: str, figure=None, note: str = "") -> str:
    if figure is None:
        content = '<div class="missing">No verified inference yet</div>'
    else:
        content = figure.to_html(full_html=False, include_plotlyjs=False,
                                 config={"responsive": True, "displayModeBar": False})
    return (f'<section class="panel"><h2>{escape(title)}</h2>'
            f'<p>{escape(note)}</p>{content}</section>')


def render(score: dict) -> str:
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    single, multi = score["single"], score["multi"]
    colors = ["#e18e4b", "#238e90", "#9b82bd"]
    panels = []
    measured = [(name, d) for name, d in single.get("engines", {}).items()
                if d.get("status") == "MEASURED"]
    if measured:
        name, main = measured[0]
        languages = main["per_language"]
        fig = go.Figure(go.Bar(
            x=[r["language"] for r in languages],
            y=[r["cer"] * 100 for r in languages], marker_color=colors[0],
            text=[f'{r["cer"]*100:.1f}%' for r in languages], textposition="outside"))
        fig.update_layout(template="plotly_white", height=370,
                          yaxis_title="CER (%)", margin=dict(t=25, b=50))
        panels.append(_panel("Character error by language", fig,
                             "Original recordings only; no synthetic perturbations"))
        fig = go.Figure()
        for key, color in (("cer", colors[0]), ("mixed_token_er", colors[1])):
            fig.add_trace(go.Bar(
                name=key, x=[r["language"] for r in languages],
                y=[r[key] * 100 for r in languages], marker_color=color))
        fig.update_layout(template="plotly_white", barmode="group", height=350,
                          yaxis_title="Error (%)", margin=dict(t=25, b=50))
        panels.append(_panel("CER vs mixed-token error", fig,
                             "Scripts are normalized; different measures are not interchangeable"))
        fig = go.Figure(go.Bar(
            x=list(main["per_scenario"]),
            y=[v["cer"] * 100 for v in main["per_scenario"].values()],
            marker_color=colors[1]))
        fig.update_layout(template="plotly_white", height=330, yaxis_title="CER (%)")
        panels.append(_panel("Recorded-speech source", fig,
                             "Crowd-recorded intent versus read speech"))
        fig = go.Figure(go.Bar(x=["p50", "p95", "max"],
                               y=[main["latency_p50_ms"]/1000,
                                  main["latency_p95_ms"]/1000,
                                  main["latency_max_ms"]/1000],
                               marker_color=colors))
        fig.update_layout(template="plotly_white", height=300,
                          yaxis_title="Seconds")
        panels.append(_panel("Processing latency", fig,
                             "Includes cold start unless benchmark was run with explicit prewarming"))
    else:
        panels.extend(_panel(t) for t in (
            "Character error by language", "CER vs mixed-token error",
            "Recorded-speech source", "Processing latency"))
    stress = score["stress"]
    valid_stress = [d for d in stress.get("engines", {}).values() if d.get("status") == "MEASURED"]
    if valid_stress:
        condition_data = valid_stress[0]["conditions"]
        fig = go.Figure(go.Bar(
            x=list(condition_data),
            y=[100 * d["mean_cer_change"] for d in condition_data.values()],
            marker_color=colors[0]))
        fig.update_layout(template="plotly_white", height=330,
                          yaxis_title="Mean CER change (percentage points)")
        panels.append(_panel("Acoustic degradation", fig,
                             "Forty correlated artificial perturbations of eight originals"))
    else:
        panels.append(_panel("Acoustic degradation", None,
                             "Optional paired 8-source, 5-condition stress test"))
    groups = multi["groups"]
    if any(g["measured"] for g in groups.values()):
        names = list(groups)
        fig = go.Figure()
        for label, field, color in (
            ("Speaker-attributed", "mean_attributed_content_recall", colors[0]),
            ("Unattributed", "mean_unattributed_content_recall", colors[1])):
            fig.add_trace(go.Bar(name=label, x=names,
                y=[100 * (groups[n][field] or 0) for n in names],
                marker_color=color))
        fig.update_layout(template="plotly_white", barmode="group", height=340,
                          yaxis_title="Content-word recall (%)")
        panels.append(_panel("Two vs three speaker recovery", fig,
                             "Attribution is speaker-permutation invariant"))
        fig = go.Figure(go.Bar(
            x=names, y=[100 * (groups[n]["median_der"] or 0) for n in names],
            marker_color=colors[2]))
        fig.update_layout(template="plotly_white", height=320,
                          yaxis_title="Median diarization error (%)")
        panels.append(_panel("Speaker diarization error", fig,
                             "Source annotations differ in temporal granularity"))
    else:
        panels.extend(_panel(t) for t in (
            "Two vs three speaker recovery", "Speaker diarization error"))
    flags = []
    if single["status"] != "MEASURED":
        flags.append(f"100 single-language cases: {single['status']}")
    if multi["status"] != "MEASURED":
        flags.append(f"50 multi-speaker cases: only {multi['valid']}/50 measured")
    flags.append("Release score: NOT CERTIFIED; no real microphone / long-session test")
    badge = " / ".join(flags)
    cards = [
        ("Original recordings", "150"),
        ("Language codes", str(score["original_language_count"])),
        ("Single-speaker inference", "100/100" if single["status"] == "MEASURED" else "PENDING"),
        ("Multi-speaker full replay", str(multi["valid"]) + "/50"),
    ]
    card_html = "".join(f'<div class="stat"><small>{escape(k)}</small><strong>{escape(v)}</strong></div>'
                        for k, v in cards)
    grouped = "".join(
        f'<tr><td>{escape(name)}</td><td>{g["measured"]}/{g["count"]}</td>'
        f'<td>{"—" if g["median_der"] is None else f"{100*g["median_der"]:.1f}%"}</td></tr>'
        for name, g in groups.items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Meeting · Fixed150 Lab</title>
<script>{get_plotlyjs()}</script><style>
:root{{--ink:#2b2c2b;--muted:#767b7d;--orange:#e18e4b;--teal:#238e90}}
*{{box-sizing:border-box}}body{{margin:0;background:#f6f7f6;color:var(--ink);font-family:Segoe UI,Arial,sans-serif}}
main{{max-width:1440px;margin:auto;padding:36px 28px 80px}}
header{{border-bottom:1px solid #dce4e1;padding:0 0 24px}}
h1{{font-size:34px;letter-spacing:-.045em;margin:6px 0}}
header p,.panel p{{color:var(--muted);font-size:13px;line-height:1.6}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:25px 0}}
.stat,.panel,.audit{{background:white;border:1px solid #e5e8e5;border-radius:14px;padding:20px}}
.stat{{display:flex;flex-direction:column;gap:8px}}.stat small{{color:var(--muted);font-size:11px}}
.stat strong{{font-size:29px;font-variant-numeric:tabular-nums;letter-spacing:-.03em}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.panel h2{{font-size:16px;margin:0}}.panel p{{margin:6px 0 14px}}
.missing{{min-height:260px;display:grid;place-items:center;color:#9aa6a6;font-size:13px}}
.audit{{margin-top:18px}}.audit p{{font-size:13px;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{padding:12px;text-align:left;border-bottom:1px solid #e5e8e5}}
@media(max-width:850px){{.grid{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><main><header><small>MEETING / QUALITY ENGINEERING</small>
<h1>Fixed150 · multilingual and meeting audio</h1><p>100 original single-speaker cases
and 50 multi-speaker cases. 25 as-recorded two-person AMI meeting clips plus 25 synthetic
three-language overlapping LAION clips. All scores require real model output and SHA-verified audio.</p>
</header><div class="stats">{card_html}</div><div class="grid">{''.join(panels)}</div>
<section class="audit"><h2>Completion and evidence gates</h2>
<p>{escape(badge)}</p>
<table><thead><tr><th>Speaker source</th><th>Measured</th><th>Median DER</th></tr></thead>
<tbody>{grouped}</tbody></table><p>AMI speaker timing comes from timestamped utterances.
LAION uses original placement intervals. Neither score establishes commercial meeting parity.
Correlated noisy/echo/telephone variants are scored separately and never counted among 150
independent source clips. Speaker-attributed recall can be substantially lower than pooled transcript recall.</p>
</section></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed150"))
    args = parser.parse_args()
    score = build_score(args.root)
    atomic_json(args.root / "scorecard.json", score)
    (args.root / "dashboard.html").write_text(render(score), encoding="utf-8")
    print("Dashboard:", args.root / "dashboard.html")
    print("Single:", score["single"]["status"])
    print("Multi:", score["multi"]["valid"], "/ 50 verified replay results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
