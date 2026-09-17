"""Offline dashboard artifacts rendered from finalized run facts."""

from __future__ import annotations

from html import escape
import json
import math
import os
from typing import Any

from .metrics import greater_is_better

SCHEMA_VERSION = 1

GLOSSARY = {
    "validation": (
        "Validation rows were reused to rank experiments. This score describes "
        "what the search steered toward, not an independent final check."
    ),
    "test": (
        "The internal test was scored once after every choice was fixed. It ranked "
        "nothing and is the run's independent check."
    ),
    "optimism": (
        "How much validation made a model look better than the internal test did. "
        "Positive values mean validation was flattering."
    ),
    "blend": (
        "The shipped prediction combines every terminal model using weights fixed "
        "before the internal test was read."
    ),
}


def build_dashboard_data(
    *,
    task_type: str,
    metric: str,
    selection_score: float,
    baseline_score: float,
    model_types: list[str],
    feature_names: list[str],
    split: dict | None = None,
    holdout: dict | None = None,
    cv_score: float | None = None,
    val_se: float | None = None,
    fold_metrics: list[float] | None = None,
    attribution: list[dict] | None = None,
    feature_catalog: list[dict] | None = None,
    blend_weights: list[float] | None = None,
    serving_contract: dict | None = None,
) -> dict[str, Any]:
    """Build the sanitized view model shared by JSON and HTML renderers."""
    scores = []
    for row in (holdout or {}).get("scores") or []:
        scores.append({
            key: row.get(key)
            for key in (
                "experiment_id", "model_type", "selected", "shipped",
                "val_score", "holdout_score", "optimism",
            )
        })
    shipped = next((row for row in scores if row.get("shipped")), None)
    if shipped is None:
        shipped = next((row for row in scores if row.get("selected")), None)
    test_score = None if shipped is None else shipped.get("holdout_score")
    gain = (
        selection_score - baseline_score
        if greater_is_better(metric)
        else baseline_score - selection_score
    )
    warnings = []
    if gain < 0:
        warnings.append({
            "level": "danger",
            "title": "Baseline not beaten",
            "message": "The shipped result did not beat the trivial baseline on validation.",
        })
    if split and split.get("val_is_thin"):
        warnings.append({
            "level": "warning",
            "title": "Thin validation set",
            "message": f"Only {split.get('n_val')} rows ranked the experiments.",
        })
    if split and split.get("holdout_is_thin"):
        warnings.append({
            "level": "warning",
            "title": "Thin internal test",
            "message": f"Only {split.get('n_holdout')} rows form the independent check.",
        })
    unmeasured = [step for step in (attribution or []) if not step.get("measured")]
    if unmeasured:
        warnings.append({
            "level": "warning",
            "title": "Feature attribution gap",
            "message": (
                f"{len(unmeasured)} feature step(s) were never measured separately; "
                "the model is valid but their individual credit is unknown."
            ),
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "task_type": task_type,
            "metric": metric,
            "direction": "higher" if greater_is_better(metric) else "lower",
            "validation_score": selection_score,
            "validation_se": val_se,
            "test_score": test_score,
            "baseline_score": baseline_score,
            "gain_vs_baseline": gain,
            "beats_baseline": gain >= 0,
            "cv_score": cv_score,
        },
        "prediction_output": (
            "positive-class probability; accuracy uses a 0.5 threshold"
            if task_type == "binary"
            else "numeric prediction"
        ),
        "split": split or {},
        "shipped": {
            "kind": "weighted blend" if len(model_types) > 1 else "single model",
            "model_types": list(model_types),
            "weights": list(blend_weights or ([1.0] if len(model_types) == 1 else [])),
        },
        "models": scores,
        "fold_metrics": list(fold_metrics or []),
        "features": {
            "names": list(feature_names),
            "catalog": list(feature_catalog or []),
            "steps": list(reversed(attribution or [])),
        },
        "serving_contract": serving_contract or {},
        "warnings": warnings,
        "glossary": GLOSSARY,
    }


def _number(value: Any, *, signed: bool = False) -> str:
    if value is None:
        return "not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return escape(str(value))
    if not math.isfinite(number):
        return "not available"
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def _tip(label: str, key: str, glossary: dict) -> str:
    definition = escape(glossary[key], quote=True)
    return (
        f'<span class="term" tabindex="0" aria-label="{escape(label)}: {definition}" '
        f'data-tip="{definition}">{escape(label)}</span>'
    )


def _model_rows(data: dict) -> str:
    models = data["models"]
    if not models:
        names = data["shipped"]["model_types"]
        weights = data["shipped"]["weights"]
        models = [
            {
                "model_type": name,
                "val_score": data["summary"]["validation_score"],
                "holdout_score": data["summary"]["test_score"],
                "optimism": None,
                "shipped": True,
                "weight": weights[i] if i < len(weights) else None,
            }
            for i, name in enumerate(names)
        ]
    weights = data["shipped"]["weights"]
    member_names = data["shipped"]["model_types"]
    score_values = [
        float(value)
        for row in models
        for value in (row.get("val_score"), row.get("holdout_score"))
        if value is not None and math.isfinite(float(value))
    ]
    low = min(score_values) if score_values else 0.0
    high = max(score_values) if score_values else 1.0
    span = high - low or 1.0
    optimism_values = [
        abs(float(row["optimism"]))
        for row in models
        if row.get("optimism") is not None and math.isfinite(float(row["optimism"]))
    ]
    max_optimism = max(optimism_values, default=1.0) or 1.0

    def score_bar(value):
        if value is None:
            return '<span class="subtle">not available</span>'
        quality = (float(value) - low) / span
        if data["summary"]["direction"] == "lower":
            quality = 1.0 - quality
        return f'<span class="score-bar"><i style="width:{15 + 85 * quality:.1f}%"></i></span>{_number(value)}'

    out = []
    for row in models:
        name = str(row.get("model_type") or "model")
        weight = row.get("weight")
        if weight is None and name in member_names:
            index = member_names.index(name)
            weight = weights[index] if index < len(weights) else None
        roles = []
        if row.get("selected"):
            roles.append("search pick")
        if row.get("shipped"):
            roles.append("shipped")
        optimism = row.get("optimism")
        optimism_bar = ""
        if optimism is not None:
            width = 100 * abs(float(optimism)) / max_optimism
            tone = "bad" if float(optimism) > 0 else "good"
            optimism_bar = (
                f'<span class="optimism-bar {tone}"><i style="width:{width:.1f}%"></i></span>'
            )
        out.append(
            "<tr>"
            f"<th scope=\"row\">{escape(name)}"
            f"<small>{escape(', '.join(roles))}</small></th>"
            f"<td>{score_bar(row.get('val_score'))}</td>"
            f"<td>{score_bar(row.get('holdout_score'))}</td>"
            f"<td class=\"{'warn' if (row.get('optimism') or 0) > 0 else ''}\">"
            f"{optimism_bar}{_number(optimism, signed=True)}</td>"
            f"<td>{_number(weight)}</td>"
            "</tr>"
        )
    return "".join(out)


def _feature_rows(data: dict) -> str:
    rows = data["features"]["catalog"]
    if not rows:
        rows = [
            {
                "feature": name,
                "kind": "feature",
                "source_columns": [],
                "pipeline_step": None,
                "description": "Shipped model feature; detailed lineage is unavailable.",
                "definition": "",
            }
            for name in data["features"]["names"]
        ]
    out = []
    for row in rows:
        sources = ", ".join(map(str, row.get("source_columns") or [])) or "—"
        definition = row.get("definition") or "No stored recipe."
        search = " ".join([
            str(row.get("feature", "")), str(row.get("kind", "")), sources,
            str(row.get("description", "")),
        ]).lower()
        out.append(
            f'<details class="feature" data-search="{escape(search, quote=True)}">'
            "<summary>"
            f"<code>{escape(str(row.get('feature', '')))}</code>"
            f"<span class=\"pill\">{escape(str(row.get('kind', 'feature')))}</span>"
            f"<span>{escape(str(row.get('description', '')))}</span>"
            "</summary>"
            "<dl>"
            f"<dt>Sources</dt><dd>{escape(sources)}</dd>"
            f"<dt>Pipeline step</dt><dd>{escape(str(row.get('pipeline_step') or 'input'))}</dd>"
            f"<dt>Recipe</dt><dd><pre>{escape(str(definition))}</pre></dd>"
            "</dl></details>"
        )
    return "".join(out)


def _feature_steps(data: dict) -> str:
    out = []
    for step in data["features"]["steps"]:
        added = ", ".join(map(str, step.get("added") or [])) or "no new columns"
        measured = step.get("measured")
        score = (
            f"best {_number(step.get('best_score'))} over "
            f"{step.get('n_experiments', 0)} experiment(s)"
            if measured else "not measured separately"
        )
        out.append(
            f'<li class="{"measured" if measured else "unmeasured"}">'
            f"<strong>{escape(str(step.get('method', 'feature step')))}</strong>"
            f"<span>{escape(added)}</span><small>{escape(score)}</small></li>"
        )
    return "".join(out) or "<li><span>No derived feature steps were recorded.</span></li>"


def _capability_review(data: dict) -> str:
    review = data.get("capability_review")
    if not review:
        return ""
    findings = []
    for row in review.get("findings") or []:
        category = str(row.get("category") or "finding").replace("_", " ")
        evidence = ", ".join(map(str, row.get("evidence_refs") or [])) or "none"
        findings.append(
            '<article class="capability-finding">'
            f'<p><span class="pill">{escape(category)}</span> '
            f'<small>{escape(str(row.get("confidence") or "unknown"))} confidence</small></p>'
            f'<h3>{escape(str(row.get("title") or "Run limitation"))}</h3>'
            f'<p>{escape(str(row.get("generic_capability") or ""))}</p>'
            '<details><summary>Evidence and falsification</summary><dl>'
            f'<dt>Evidence</dt><dd>{escape(evidence)}</dd>'
            f'<dt>Current tools</dt><dd>{escape(str(row.get("why_existing_tools_insufficient") or ""))}</dd>'
            f'<dt>Alternative</dt><dd>{escape(str(row.get("alternative_explanation") or ""))}</dd>'
            f'<dt>Risk</dt><dd>{escape(str(row.get("leakage_or_serving_risk") or ""))}</dd>'
            f'<dt>Test</dt><dd>{escape(str(row.get("testable_hypothesis") or ""))}</dd>'
            '</dl></details></article>'
        )
    return (
        '<section id="capability-review"><h2>What limited this run?</h2>'
        f'<p>{escape(str(review.get("overall_assessment") or ""))}</p>'
        + "".join(findings)
        + '<p class="subtle">These are owner-reviewable hypotheses, not product '
        'decisions. The internal test ranked nothing.</p></section>'
    )


def render_dashboard_html(data: dict[str, Any]) -> str:
    """Render one dependency-free, offline, keyboard-accessible HTML document."""
    summary = data["summary"]
    glossary = data["glossary"]
    split = data["split"]
    contract = data["serving_contract"]
    warning_html = "".join(
        f'<article class="notice {escape(item["level"])}"><strong>'
        f'{escape(item["title"])}</strong><p>{escape(item["message"])}</p></article>'
        for item in data["warnings"]
    ) or '<article class="notice ok"><strong>No report warnings</strong><p>The recorded checks passed.</p></article>'
    split_mode = (
        f"ordered by {split.get('split_on')}"
        if split.get("split_on")
        else "random, stratified" if split.get("stratified") else "random"
    )
    split_cards = "".join(
        f"<li><strong>{escape(label)}</strong><span>{escape(str(split.get(key, 0)))} rows</span></li>"
        for label, key in (("Work", "n_work"), ("Validation", "n_val"), ("Test", "n_holdout"))
        if split.get(key)
    )
    contract_rows = [
        ("Target", contract.get("target")),
        ("Serving columns", ", ".join(contract.get("serving_columns") or [])),
        ("Identifiers kept out of models", ", ".join(contract.get("identifier_columns") or [])),
        ("Static joins", ", ".join(contract.get("joins") or [])),
        ("Required streams", ", ".join(contract.get("required_streams") or [])),
    ]
    contract_html = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(str(value or 'none'))}</dd>"
        for label, value in contract_rows
    )
    capability_html = _capability_review(data)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tabular Agent run report</title>
<style>
:root{{--ink:#17202a;--muted:#667085;--paper:#f5f7fb;--card:#fff;--line:#dfe4ea;--blue:#2457d6;--green:#13795b;--amber:#a15c00;--red:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1120px;margin:auto;padding:32px 20px 64px}} h1{{margin:.2rem 0}} h2{{margin:0 0 16px;font-size:1.15rem}} .eyebrow{{color:var(--blue);font-weight:700;text-transform:uppercase;letter-spacing:.08em}}
.subtle,small{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:24px 0}} .card,section{{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:0 3px 14px #1822300a}}
.card{{padding:16px}} .card strong{{display:block;font-size:1.65rem;margin:.35rem 0}} section{{padding:22px;margin:16px 0}} .good{{color:var(--green)}} .bad,.danger{{color:var(--red)}} .warn,.warning{{color:var(--amber)}}
.split{{display:flex;gap:8px;padding:0;list-style:none}} .split li{{flex:1;border-left:5px solid var(--blue);padding:10px 14px;background:#f8faff;border-radius:6px}} .split span,.model-table small{{display:block;color:var(--muted)}}
.model-table{{width:100%;border-collapse:collapse}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:right}} th:first-child{{text-align:left}} .term{{border-bottom:1px dotted currentColor;cursor:help;position:relative}} .term:focus{{outline:2px solid var(--blue);outline-offset:3px}}
.score-bar,.optimism-bar{{display:inline-block;width:54px;height:6px;margin-right:7px;background:var(--line);border-radius:6px;overflow:hidden;vertical-align:middle}} .score-bar i,.optimism-bar i{{display:block;height:100%;background:var(--blue)}} .optimism-bar.bad i{{background:var(--red)}} .optimism-bar.good i{{background:var(--green)}}
.term:hover:after,.term:focus:after{{content:attr(data-tip);position:absolute;z-index:2;left:0;top:1.6em;width:min(320px,75vw);padding:10px;background:#111827;color:white;border-radius:8px;font-weight:400}}
.journey{{display:grid;gap:8px;padding-left:20px}} .journey li{{padding:10px;border-left:4px solid var(--green)}} .journey li.unmeasured{{border-color:var(--amber)}} .journey span,.journey small{{display:block}}
.toolbar{{display:flex;gap:10px;margin-bottom:12px}} input{{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;font:inherit}} details.feature{{border-top:1px solid var(--line);padding:10px 0}} details.feature[hidden]{{display:none}} summary{{display:grid;grid-template-columns:minmax(140px,1fr) auto 2fr;gap:10px;cursor:pointer;align-items:center}} code,pre{{font-family:ui-monospace,SFMono-Regular,monospace}} .pill{{background:#eef2ff;color:#3346a8;border-radius:999px;padding:2px 8px;font-size:.8rem}} dl{{display:grid;grid-template-columns:180px 1fr;gap:8px 16px}} dt{{font-weight:700}} dd{{margin:0}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f7;padding:10px;border-radius:8px}}
.notice{{border-left:5px solid var(--green);padding:12px 14px;margin:8px 0;background:var(--card)}} .notice.warning{{border-color:var(--amber)}} .notice.danger{{border-color:var(--red)}} .notice p{{margin:.2rem 0 0}} footer{{color:var(--muted);margin-top:28px}}
.capability-finding{{border-top:1px solid var(--line);padding:10px 0}} .capability-finding h3{{margin:.25rem 0}} .capability-finding details{{margin-top:8px}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr 1fr}}summary{{grid-template-columns:1fr}}.split{{display:grid}}dl{{grid-template-columns:1fr}}}}
@media print{{body{{background:white}}section,.card{{box-shadow:none;break-inside:avoid}}input{{display:none}}}}
@media(prefers-color-scheme:dark){{:root{{--ink:#f2f4f7;--muted:#aeb8c4;--paper:#101828;--card:#182230;--line:#344054}}.split li,pre{{background:#202b3c}}}}
</style>
</head>
<body><main>
<header><div class="eyebrow">Tabular Agent · finalized run</div><h1>Run report</h1>
<p class="subtle">{escape(summary["task_type"])} · {escape(summary["metric"])} · {escape(summary["direction"])} is better</p></header>
<div class="grid" aria-label="Run outcome">
<article class="card"><span>{_tip("Validation", "validation", glossary)}</span><strong>{_number(summary["validation_score"])}</strong><small>used during search</small></article>
<article class="card"><span>{_tip("Test", "test", glossary)}</span><strong>{_number(summary["test_score"])}</strong><small>internal, scored once</small></article>
<article class="card"><span>Baseline</span><strong>{_number(summary["baseline_score"])}</strong><small>trivial model</small></article>
<article class="card"><span>Gain vs baseline</span><strong class="{"good" if summary["beats_baseline"] else "bad"}">{_number(summary["gain_vs_baseline"], signed=True)}</strong><small>positive is better</small></article>
</div>
<section><h2>How the data was judged</h2><ol class="split">{split_cards}</ol><p class="subtle">{escape(split_mode)}. Validation ranked candidates; the internal test ranked nothing.</p></section>
<section><h2>What shipped</h2><p><strong>{escape(data["shipped"]["kind"])}</strong> · {escape(data["prediction_output"])}</p>
<table class="model-table"><thead><tr><th>Model</th><th>{_tip("Validation", "validation", glossary)}</th><th>{_tip("Test", "test", glossary)}</th><th>{_tip("Optimism", "optimism", glossary)}</th><th>Weight</th></tr></thead><tbody>{_model_rows(data)}</tbody></table></section>
<section><h2>How the feature set was built</h2><ol class="journey">{_feature_steps(data)}</ol></section>
<section><h2>Feature catalog <small>({len(data["features"]["names"])} shipped columns)</small></h2>
<div class="toolbar"><label for="feature-search" class="subtle">Search features</label><input id="feature-search" type="search" placeholder="Name, source, kind or meaning"></div>
<div id="features">{_feature_rows(data)}</div></section>
<section><h2>Serving contract</h2><dl>{contract_html}</dl>
<details><summary>Use these learned features in your own model</summary><pre>train = run.training_features()
test = run.bundle.feature_table(new_rows)
# train[run.bundle.feature_names] and test now share the same ordered columns</pre></details></section>
{capability_html}
<section><h2>Warnings and limits</h2>{warning_html}</section>
<footer>Generated from finalized run facts. This report never recomputes scores or selects a model. See run_report.json for the same sanitized data.</footer>
</main>
<script>
const search=document.getElementById("feature-search");
search.addEventListener("input",()=>{{const q=search.value.trim().toLowerCase();document.querySelectorAll(".feature").forEach(row=>row.hidden=!row.dataset.search.includes(q));}});
</script></body></html>"""


def _atomic_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write_dashboard_artifacts(run_ctx, markdown: str, data: dict[str, Any]) -> dict[str, str]:
    """Atomically write Markdown, sanitized JSON, and self-contained HTML."""
    paths = {
        "markdown": run_ctx.path("run_report.md"),
        "json": run_ctx.path("run_report.json"),
        "html": run_ctx.path("run_report.html"),
    }
    _atomic_text(paths["markdown"], markdown)
    _atomic_text(paths["json"], json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(paths["html"], render_dashboard_html(data))
    return paths


def update_dashboard_contract(run_dir: str, plan: dict) -> None:
    """Enrich a facade report after its confirmed plan is copied into the bundle."""
    json_path = os.path.join(run_dir, "run_report.json")
    html_path = os.path.join(run_dir, "run_report.html")
    if not os.path.exists(json_path):
        return
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)
    existing = data.get("serving_contract") or {}
    data["serving_contract"] = {
        "target": plan.get("target") or existing.get("target"),
        "serving_columns": list(plan.get("serving_columns") or []),
        "identifier_columns": list(plan.get("identifier_columns") or []),
        "joins": [
            str(item.get("right_table"))
            for item in (plan.get("joins") or [])
            if item.get("right_table")
        ],
        "required_streams": list(existing.get("required_streams") or []),
    }
    _atomic_text(json_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    _atomic_text(html_path, render_dashboard_html(data))
