"""Create an offline, evidence-only seven-panel benchmark dashboard.

Never substitutes fabricated metrics when the user's GPU has not run inference.
Inputs:
  benchmark/fixed10/lock.json
  benchmark/fixed10/manifest_single.jsonl
  benchmark/fixed10/single_asr.json (phase07_benchmark --infer)
  benchmark/fixed10/overlap_predictions.json (optional, integrated pipeline results)
Output:
  benchmark/fixed10/dashboard.html
  benchmark/fixed10/scorecard.json
Requires: pip install plotly
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
import unicodedata


def _tokens(value: str) -> Counter:
    words = re.findall(r"[^\W_]{3,}", unicodedata.normalize("NFKC", value).casefold(), re.UNICODE)
    return Counter(words)


def content_recall(reference: str, hypothesis: str) -> float | None:
    expected = _tokens(reference)
    if not expected:
        return None
    found = _tokens(hypothesis)
    return sum(min(n, found[w]) for w, n in expected.items()) / sum(expected.values())


def prepare(lock: dict, asr: dict | None, overlap: dict | None) -> dict:
    cases = lock["cases"]
    if len(cases) != 10 or sum(c["kind"] == "single" for c in cases) != 8:
        raise ValueError("Fixed10 lock contains the wrong number of cases")
    valid_report = asr is not None and asr.get("infer") is True and asr.get("samples") == 8
    if asr is not None and not valid_report:
        raise ValueError("A dashboard quality score requires eight real --infer results")
    per_engine: dict[str, dict] = {}
    if valid_report:
        for engine, body in asr["engines"].items():
            rows = body.get("samples", [])
            if len(rows) != 8 or any(r.get("index") != i for i, r in enumerate(rows)):
                raise ValueError("Engine result is not aligned to the frozen eight single-speaker cases")
            if any(r.get("error") for r in rows):
                per_engine[engine] = {"complete": False, "error": "Inference failed for one or more cases"}
                continue
            n_chars = sum(int(r["reference_characters"]) for r in rows)
            n_toks = sum(int(r["reference_tokens"]) for r in rows)
            if not n_chars or not n_toks:
                raise ValueError("Invalid reference denominator")
            latencies = [r.get("latency_ms") for r in rows]
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in latencies):
                per_engine[engine] = {"complete": False, "error": "Missing real inference timing"}
                continue
            ordered = sorted(latencies)
            per_engine[engine] = {
                "complete": True, "rows": rows,
                "cer": sum(r["character_edits"] for r in rows) / n_chars,
                "mixed_token_error": sum(r["token_edits"] for r in rows) / n_toks,
                "language_accuracy": sum(
                    r.get("predicted_language") == r.get("expected_language")
                    for r in rows
                ) / len(rows),
                "latency_p50_ms": statistics.median(latencies),
                "latency_p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
            }
    overlap_results = []
    if overlap is not None:
        if not isinstance(overlap, dict):
            raise ValueError("overlap_predictions.json must be an object keyed by clip id")
        for case in cases:
            if case["kind"] != "overlap":
                continue
            prediction = overlap.get(case["id"])
            if not isinstance(prediction, dict) or not isinstance(prediction.get("segments"), list):
                overlap_results.append({"id": case["id"], "complete": False})
                continue
            parts = prediction["segments"]
            if not all(isinstance(p, dict) and "text" in p and
                       "start_ms" in p and "end_ms" in p for p in parts):
                overlap_results.append({"id": case["id"], "complete": False})
                continue
            hypothesis = " ".join(str(p["text"]) for p in parts)
            recalls = [content_recall(s["reference"], hypothesis) for s in case["speakers"]]
            speakers = set(str(p.get("speaker_id", "-1")) for p in parts)
            identified = len([s for s in speakers if s not in ("-1", "0", "und", "unknown")])
            result = {
                "id": case["id"], "complete": True, "speaker_recalls": recalls,
                "languages": [s["language"] for s in case["speakers"]],
                "speaker_count": identified, "reference_speaker_count": 3,
                "turn_count": len(parts), "parts": parts,
                "expected": case["speakers"],
            }
            try:
                from tools.phase09_speaker_benchmark import evaluate
                reference_segments = [{
                    "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                    "speaker": s["speaker"]
                } for s in case["speakers"]]
                predicted_segments = [{
                    "start_ms": p["start_ms"], "end_ms": p["end_ms"],
                    "speaker_id": str(p.get("speaker_id", "-1"))
                } for p in parts]
                result["der"] = evaluate(reference_segments, predicted_segments)["der"]
            except (ImportError, ValueError, KeyError, TypeError):
                result["der"] = None
            overlap_results.append(result)
    gates = []
    for name, data in per_engine.items():
        if not data.get("complete"):
            gates.append({"engine": name, "status": "BLOCKED", "reason": data.get("error")})
            continue
        issues = []
        if data["cer"] > .22:
            issues.append("CER above exploratory 22% threshold. Compare language-specific ASR adapters.")
        if data["mixed_token_error"] > .32:
            issues.append("Token error above 32%. Inspect acoustic decoding and lexical deletion.")
        if data["language_accuracy"] < .75:
            issues.append("Wrong auto-language labels. Profile short-window multilingual LID.")
        if data["latency_p95_ms"] > 3500:
            issues.append("p95 inference over 3.5s. Profile GPU queue and model residency.")
        gates.append({"engine": name, "status": "REVIEW" if issues else "SMOKE PASS",
                      "reasons": issues})
    if not per_engine:
        gates.append({"engine": "none", "status": "NOT RUN",
                      "reason": "Actual model inference has not been measured"})
    if not overlap_results:
        gates.append({"engine": "overlap", "status": "NOT RUN",
                      "reason": "Speaker-tagged end-to-end overlap predictions are missing"})
    for item in overlap_results:
        if not item["complete"]:
            gates.append({"engine": item["id"], "status": "BLOCKED",
                          "reason": "No full-pipeline segments provided"})
            continue
        if item.get("der") is None or any(x is None for x in item["speaker_recalls"]):
            gates.append({"engine": item["id"], "status": "BLOCKED",
                          "reason": "Missing scorer inputs"})
        elif item["der"] > .35 or min(item["speaker_recalls"]) < .20:
            gates.append({"engine": item["id"], "status": "REVIEW",
                          "reason": "Poor overlapping speaker separation or transcription"})
        else:
            gates.append({"engine": item["id"], "status": "SMOKE PASS"})
    return {
        "schema": "meeting-fixed10-scorecard-v1", "cases": len(cases),
        "languages_in_smoke": sorted(set(
            [c["language"] for c in cases if c["kind"] == "single"] +
            [s["language"] for c in cases if c["kind"] == "overlap" for s in c["speakers"]]
        )),
        "real_inference": valid_report,
        "engines": per_engine, "overlap": overlap_results, "gates": gates,
        "release_certified": False,
    }


def _number(value, digits=1, suffix=""):
    return f"{value:.{digits}f}{suffix}" if value is not None else "—"


def _panel(title: str, subtitle: str, figure=None) -> str:
    if figure is None:
        content = '<div class="pending"><span class="pending-orbit">○</span><strong>Awaiting measured data</strong><p>Run real inference or provide speaker-tagged predictions.</p></div>'
    else:
        content = figure.to_html(full_html=False, include_plotlyjs=False, config={
            "displayModeBar": False, "responsive": True, "staticPlot": False,
        })
    return f'<article class="visual-panel"><div class="panel-copy"><h3>{escape(title)}</h3><p>{escape(subtitle)}</p></div>{content}</article>'


def html_report(lock: dict, data: dict) -> str:
    import plotly.graph_objects as go
    from plotly.offline.offline import get_plotlyjs
    base = dict(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, Arial", color="#40372f", size=12),
        margin=dict(l=54, r=25, t=18, b=48),
    )
    orange, teal, gray, red = "#d27729", "#178487", "#a0a6ad", "#c8514d"
    finished = {name: d for name, d in data["engines"].items() if d.get("complete")}
    charts = []
    # 1. Language-by-model heatmap, no values invented for missing rows.
    if finished:
        names = list(finished)
        labels = [c["language"] for c in lock["cases"] if c["kind"] == "single"]
        z = [[round(d["rows"][i]["cer"] * 100, 1) for i in range(8)] for d in finished.values()]
        fig = go.Figure(go.Heatmap(
            x=labels, y=names, z=z, zmin=0, zmax=100,
            colorscale=[[0, "#e5f1ea"], [.25, "#a9d3c9"], [.55, "#f2bd76"], [1, "#b74c36"]],
            colorbar=dict(title="CER %", thickness=12), text=z, texttemplate="%{text}%",
            hovertemplate="Engine %{y}<br>Language %{x}<br>CER %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(**base, height=290, yaxis=dict(autorange="reversed"))
        charts.append(_panel("Cross-language error matrix", "Measured character error by locale and ASR engine", fig))
        # 2. Error measures for primary engine.
        primary_name, primary = next(iter(finished.items()))
        row_cer = [r["cer"] * 100 for r in primary["rows"]]
        row_ter = [r["mixed_token_error_rate"] * 100 for r in primary["rows"]]
        fig = go.Figure()
        fig.add_bar(y=labels, x=row_cer, name="CER", orientation="h", marker_color=orange)
        fig.add_bar(y=labels, x=row_ter, name="Mixed-token ER", orientation="h", marker_color=teal)
        fig.update_layout(**base, height=360, barmode="group", xaxis_title="Error rate (%)",
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.15))
        charts.append(_panel("Error by writing system", "CER and mixed-token error are not interchangeable", fig))
        # 3. Inference latency.
        fig = go.Figure()
        for name, d in finished.items():
            fig.add_trace(go.Scatter(
                x=labels, y=[r["latency_ms"] for r in d["rows"]],
                mode="lines+markers", name=name, marker=dict(size=9),
                line=dict(width=2), hovertemplate="%{x}<br>%{y:.0f} ms<extra>%{fullData.name}</extra>"
            ))
        fig.add_hline(y=3500, line_dash="dot", line_color=red,
                      annotation_text="3.5s exploratory threshold")
        fig.update_layout(**base, height=330, xaxis_title="Language", yaxis_title="Inference latency (ms)")
        charts.append(_panel("Real-model latency profile", "Measured wall time on the actual inference environment", fig))
        # 4. language identification.
        fig = go.Figure()
        labels_per_model = list(finished)
        ratios = [d["language_accuracy"] * 100 for d in finished.values()]
        fig.add_trace(go.Bar(x=ratios, y=labels_per_model, orientation="h", marker_color=teal,
                             text=[f"{x:.0f}%" for x in ratios], textposition="auto"))
        fig.update_layout(**base, height=250, xaxis=dict(range=[0,100],title="Correct language label (%)"))
        charts.append(_panel("Automatic language identification", "Language detection, not hard-coded English or Chinese", fig))
    else:
        charts.extend([
            _panel("Cross-language error matrix", "CER per language and model"),
            _panel("Error by writing system", "Character and token error rates"),
            _panel("Real-model latency profile", "Inference time with p95 threshold"),
            _panel("Automatic language identification", "Auto language detection accuracy"),
        ])
    # 5. Three-speaker multilingual overlaps.
    overlaps = [x for x in data["overlap"] if x.get("complete")]
    if overlaps:
        fig = go.Figure()
        for x in overlaps:
            for i, value in enumerate(x["speaker_recalls"]):
                fig.add_trace(go.Bar(
                    x=[f"{x['id']} · {x['languages'][i]}"],
                    y=[(value or 0) * 100], marker_color=[orange, teal, "#8a80a9"][i],
                    name=f"Speaker {i+1}", showlegend=(x is overlaps[0]),
                    hovertemplate="%{x}<br>Content recall %{y:.1f}%<extra></extra>"
                ))
        fig.update_layout(**base, height=310, yaxis_title="Content-word recall (%)",
                          yaxis=dict(range=[0, 100]), legend=dict(orientation="h", y=1.17))
        charts.append(_panel("Overlap speaker recovery", "Per-person content recall; do not apply ordinary WER to simultaneous voices", fig))
    else:
        charts.append(_panel("Overlap speaker recovery", "Three concurrent speakers in two fixed recordings"))
    # 6. Speaker activity spans, grounded in actual reference annotations and optional hypotheses.
    sample = next((c for c in lock["cases"] if c["kind"] == "overlap"), None)
    if sample:
        fig = go.Figure()
        speakers = sample["speakers"]
        for i, sp in enumerate(speakers):
            fig.add_trace(go.Scatter(
                x=[sp["start_ms"]/1000, sp["end_ms"]/1000],
                y=[i, i], mode="lines", line=dict(width=14, color=[orange,teal,"#8a80a9"][i]),
                name=sp["speaker"]+" "+sp["language"], hovertemplate="Time %{x:.1f}s<extra>%{fullData.name}</extra>"
            ))
        fig.update_layout(**base, height=260, xaxis_title="Seconds", yaxis=dict(
            tickmode="array", tickvals=list(range(len(speakers))),
            ticktext=[f"{x['speaker']} · {x['language']}" for x in speakers],
            autorange="reversed"), showlegend=False)
        charts.append(_panel("Reference speaker occupancy", "Actual supplied source annotation, not a prediction", fig))
    else:
        charts.append(_panel("Reference speaker occupancy", "Ground-truth speaker activity timeline"))
    # 7. Live readiness and unresolved blockers.
    status_rows = []
    for g in data["gates"]:
        text = "; ".join(g.get("reasons", [])) or g.get("reason", "No blocking smoke-set flags")
        status_rows.append(f'<tr><td>{escape(g["engine"])}</td><td><span class="gate {escape(g["status"].lower().replace(" ","-"))}">{escape(g["status"])}</span></td><td>{escape(text)}</td></tr>')
    summary = next(iter(finished.values())) if finished else None
    metric = [
        ("Average CER", _number(100 * summary["cer"] if summary else None, 1, "%"), "lower is better"),
        ("Mixed-token ER", _number(100 * summary["mixed_token_error"] if summary else None, 1, "%"), "language-aware tokens"),
        ("Language ID", _number(100 * summary["language_accuracy"] if summary else None, 1, "%"), "higher is better"),
        ("Latency p95", _number(summary["latency_p95_ms"] if summary else None, 0, " ms"), "GPU processing time"),
    ]
    metrics = "".join(f'<div class="metric"><span>{escape(n)}</span><strong>{escape(v)}</strong><small>{escape(h)}</small></div>' for n,v,h in metric)
    status = "EVIDENCE INCOMPLETE" if not summary or len(overlaps) != 2 else "SMOKE METRICS AVAILABLE"
    style = """
:root{--ink:#25231f;--sub:#756b63;--orange:#d27729;--teal:#178487;--line:#e6e2dd;--paper:#f8f7f4}*{box-sizing:border-box}
body{margin:0;background:var(--paper);font-family:Inter,Segoe UI,Arial,sans-serif;color:var(--ink);line-height:1.55}
.shell{max-width:1510px;margin:0 auto;padding:42px 36px 95px}.topline{display:flex;align-items:center;gap:10px;font-size:12px;font-weight:750;color:var(--teal);letter-spacing:.15em;text-transform:uppercase}.topline:before{content:'';height:9px;width:9px;background:var(--teal);border-radius:100%}
.hero{display:flex;justify-content:space-between;align-items:end;gap:40px;padding:14px 0 36px;border-bottom:1px solid var(--line)}h1{font-size:38px;font-weight:770;letter-spacing:-.055em;line-height:1.08;margin:9px 0}.hero p{color:var(--sub);font-size:14px;max-width:740px;margin:12px 0 0}
.pill{padding:8px 12px;border:1px solid #d8d0c7;border-radius:60px;white-space:nowrap;color:#7c695a;font-size:11px;letter-spacing:.08em;font-weight:750}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:26px 0}.metric{background:#fff;border:1px solid var(--line);border-radius:14px;padding:19px 23px;display:flex;flex-direction:column;min-height:120px}
.metric span{color:var(--sub);font-size:11px;letter-spacing:.065em;text-transform:uppercase}.metric strong{font-size:33px;line-height:1.55;letter-spacing:-.045em;font-variant-numeric:tabular-nums}.metric small{font-size:11px;color:#8d8881}
.visual-grid{display:grid;grid-template-columns:1fr 1fr;gap:17px}.visual-panel{background:#fff;border:1px solid var(--line);border-radius:15px;overflow:hidden;min-width:0;box-shadow:0 8px 25px rgba(50,37,27,.023)}
.panel-copy{padding:23px 24px 0}.panel-copy h3{margin:0;color:var(--ink);font-size:15px;letter-spacing:-.015em;font-weight:750}.panel-copy p{margin:5px 0 0;color:var(--sub);font-size:11px}.pending{min-height:270px;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#a3a09b}.pending strong{font-size:13px;color:#786f69}.pending p{font-size:11px}.pending-orbit{font-size:45px;line-height:1.4;color:#d2ceca}
.section-head{display:flex;gap:20px;align-items:end;justify-content:space-between;margin:34px 0 16px}h2{margin:0;font-size:20px;letter-spacing:-.025em} .section-head span{color:var(--sub);font-size:12px}
.audit{background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden}table{border-collapse:collapse;width:100%;font-size:12px}th{text-align:left;background:#fbfaf8;color:#827971;font-size:11px;letter-spacing:.05em;text-transform:uppercase}td,th{padding:13px 18px;border-bottom:1px solid #eee9e4;vertical-align:top}tr:last-child td{border-bottom:0}
.gate{font-size:10px;border-radius:40px;padding:5px 10px;background:#eee9e2;color:#786b5f;font-weight:700;white-space:nowrap}.gate.smoke-pass{background:#e8f2ef;color:#198073}.gate.review{background:#faeddc;color:#ad6f23}.gate.blocked{background:#f9e9e8;color:#b04d44}
.method{margin-top:22px;color:var(--sub);font-size:12px;max-width:1050px}.method b{color:var(--ink)}.foot{margin-top:35px;padding-top:20px;border-top:1px solid var(--line);font-size:11px;color:#968c81}
@media(max-width:950px){.visual-grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.hero{flex-direction:column;align-items:flex-start}}
@media(max-width:600px){.shell{padding:25px 15px 50px}.hero h1{font-size:29px}.stats{gap:8px}.metric{padding:13px}.metric strong{font-size:25px}}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meeting · Fixed10 Quality Lab</title><style>{style}</style>
<script>{get_plotlyjs()}</script></head>
<body><main class="shell"><div class="topline">MEETING / QUALITY ENGINEERING</div>
<header class="hero"><div><h1>Fixed10 multilingual<br>quality lab.</h1>
<p>Frozen audio sources, immutable checksums and actual inference measurements.
Eight single-speaker languages plus two three-speaker overlap stress tests.
This is an exploratory regression suite, not a claim of commercial meeting parity.</p></div>
<div class="pill">{escape(status)}</div></header>
<section class="stats">{metrics}</section>
<div class="section-head"><h2>Acoustic performance</h2><span>All charts show evidence, not estimates</span></div>
<section class="visual-grid">{''.join(charts)}</section>
<div class="section-head"><h2>Release observations</h2><span>Thresholds are initial engineering hypotheses</span></div>
<section class="audit"><table><thead><tr><th>Scope</th><th>Result</th><th>Reason / next action</th></tr></thead>
<tbody>{''.join(status_rows)}</tbody></table></section>
<p class="method"><b>Interpretation.</b> The overlap source is a three-person mixture with reference times.
An ordinary ASR transcript cannot prove speaker identities. No low error rate on eight short scripted
clips can certify all mainstream languages. Expand to the full per-language FLEURS/real-meeting
hold-out set, compare models on the same frozen samples, and run long-duration recovery tests.
Speaker DER and word attribution are omitted until segment-level outputs exist.</p>
<footer class="foot">Sources PolyAI MInDS-14 and LAION overlapping speech benchmark · CC BY 4.0 ·
Source revisions and SHA-256 digests in lock.json · Generated without fabricated inference scores</footer>
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("benchmark/fixed10"))
    parser.add_argument("--asr", type=Path)
    parser.add_argument("--overlap", type=Path)
    args = parser.parse_args()
    root = args.root
    lock = json.loads((root / "lock.json").read_text(encoding="utf-8"))
    asr_file = args.asr or root / "single_asr.json"
    overlap_file = args.overlap or root / "overlap_predictions.json"
    asr = json.loads(asr_file.read_text(encoding="utf-8")) if asr_file.is_file() else None
    overlap = json.loads(overlap_file.read_text(encoding="utf-8")) if overlap_file.is_file() else None
    score = prepare(lock, asr, overlap)
    view = html_report(lock, score)
    root.mkdir(parents=True, exist_ok=True)
    (root / "dashboard.html").write_text(view, encoding="utf-8")
    # Exclude raw transcribed content and times from public scorecard.
    redacted = {k: v for k, v in score.items() if k not in ("engines", "overlap")}
    redacted["engines"] = {
        name: {key: value for key, value in body.items() if key != "rows"}
        for name, body in score["engines"].items()
    }
    redacted["overlap"] = [
        {key: value for key, value in body.items() if key not in ("parts", "expected")}
        for body in score["overlap"]
    ]
    (root / "scorecard.json").write_text(json.dumps(redacted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dashboard: {root / 'dashboard.html'}")
    for gate in score["gates"]:
        print(f"{gate['engine']}: {gate['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
