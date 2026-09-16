"""Evidence-bound, read-only postmortems for finalized runs.

The feature agent may build and score tables.  This reviewer deliberately may
do neither: it receives a sanitized account of an already-finalized run, has no
tools, and can only write a diagnosis artifact.  That separation keeps a test
score useful as retrospective evidence without turning it into another model-
selection surface.
"""

from __future__ import annotations

from collections import defaultdict
import fcntl
import hashlib
from html import escape as html_escape
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal

from pydantic import BaseModel, Field

from .dashboard import render_dashboard_html


REVIEW_SCHEMA_VERSION = 1
BACKLOG_SCHEMA_VERSION = 1
REVIEW_FILENAME = "capability_review.json"
MARKDOWN_START = "<!-- capability-review:start -->"
MARKDOWN_END = "<!-- capability-review:end -->"

GapCategory = Literal[
    "capability_gap",
    "agent_gap",
    "tool_defect",
    "model_selection_gap",
    "data_contract_gap",
    "no_demonstrated_gap",
]
Confidence = Literal["low", "medium", "high"]


class CapabilityFindingDraft(BaseModel):
    """One diagnosis before deterministic validation and fingerprinting."""

    category: GapCategory
    title: str = Field(min_length=1, max_length=140)
    generic_capability_key: str = Field(min_length=1, max_length=80)
    generic_capability: str = Field(min_length=1, max_length=500)
    evidence_refs: list[str] = Field(min_length=1, max_length=12)
    existing_tools_considered: list[str] = Field(default_factory=list, max_length=12)
    why_existing_tools_insufficient: str = Field(min_length=1, max_length=1000)
    alternative_explanation: str = Field(min_length=1, max_length=1000)
    leakage_or_serving_risk: str = Field(min_length=1, max_length=1000)
    testable_hypothesis: str = Field(min_length=1, max_length=1000)
    confidence: Confidence


class CapabilityReviewDraft(BaseModel):
    """Typed output contract for the tool-less reviewer."""

    overall_assessment: str = Field(min_length=1, max_length=1200)
    findings: list[CapabilityFindingDraft] = Field(min_length=1, max_length=8)


SYSTEM_PROMPT = """You audit one already-finalized tabular-ML run.

Classify what limited the run, using only the supplied evidence IDs:
- capability_gap: a useful generic operation could not be expressed by the tools.
- agent_gap: an existing available tool could have expressed it, but was missed or misused.
- tool_defect: a tool or generic default produced misleading measurement or behavior.
- model_selection_gap: features were not the bottleneck; model families, objectives,
  compute budget, calibration, or blending/selection were.
- data_contract_gap: target, serving availability, grain, grouping, or split semantics
  were insufficient or wrong.
- no_demonstrated_gap: the evidence does not establish a generic shortcoming.

Requirements:
1. Cite exact supplied evidence IDs. Never invent an ID.
2. State capabilities generically, without dataset, competition, company, or column names.
3. A failed feature experiment does not prove a missing feature tool when the operation
   was already expressible. A score gap alone does not prove any category.
4. Explain why existing tools are insufficient, or explicitly say they were sufficient.
5. Include leakage/serving risk, a falsifiable hypothesis, and a plausible alternative.
6. The internal test score is retrospective evidence only. Never recommend choosing,
   refitting, or shipping a model because of it.
7. Do not propose code changes as facts. Findings are hypotheses for an owner-reviewed
   backlog, and no single run may establish a product change.
8. For no_demonstrated_gap use generic_capability_key='no_demonstrated_gap' and explain
   what additional evidence would be required.
"""


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _stable_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return key[:80] or "unspecified_gap"


def _fingerprint(category: str, capability_key: str) -> str:
    canonical = f"{category}:{_stable_key(capability_key)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _table_ref(value: Any, paths: dict[str, str]) -> str:
    if not value:
        return "raw"
    return paths.get(str(value), "raw")


def build_review_context(run_dir: str | os.PathLike) -> dict[str, Any]:
    """Build an allow-listed, row-free evidence package for the reviewer."""
    root = Path(run_dir)
    with open(root / "run_report.json", encoding="utf-8") as handle:
        report = json.load(handle)
    with open(root / "run_journal.json", encoding="utf-8") as handle:
        journal = json.load(handle)
    if not (journal.get("stop") or {}).get("finalized"):
        raise ValueError("capability review requires a finalized run")

    evidence: list[dict[str, Any]] = []

    def add(ref: str, kind: str, facts: dict[str, Any]) -> None:
        evidence.append({"id": ref, "kind": kind, "facts": facts})

    add("summary", "outcome", dict(report.get("summary") or {}))
    add("split", "data_contract", {
        key: (report.get("split") or {}).get(key)
        for key in (
            "profile", "n_work", "n_val", "n_holdout", "split_on",
            "stratified", "val_is_thin", "holdout_is_thin",
        )
    })
    add("shipped", "shipment", dict(report.get("shipped") or {}))
    add("serving_contract", "data_contract", dict(report.get("serving_contract") or {}))

    for index, row in enumerate(report.get("models") or [], start=1):
        identifier = row.get("experiment_id") or index
        add(f"model:{identifier}", "model_result", {
            key: row.get(key)
            for key in (
                "experiment_id", "model_type", "selected", "shipped",
                "val_score", "holdout_score", "optimism",
            )
        })
    for index, row in enumerate(report.get("warnings") or [], start=1):
        add(f"warning:{index}", "warning", {
            "level": row.get("level"),
            "title": row.get("title"),
            "message": row.get("message"),
        })
    for index, row in enumerate((report.get("features") or {}).get("steps") or [], start=1):
        add(f"feature_step:{index}", "feature_result", {
            key: row.get(key)
            for key in ("method", "added", "measured", "best_score", "n_experiments")
        })

    table_paths = {
        str(row.get("path")): f"table:{row.get('id') or index}"
        for index, row in enumerate(journal.get("tables") or [], start=1)
        if row.get("path")
    }
    for index, row in enumerate(journal.get("tables") or [], start=1):
        ref = f"table:{row.get('id') or index}"
        add(ref, "feature_attempt", {
            "method": row.get("method"),
            "source": _table_ref(row.get("source_table"), table_paths),
            "new_columns": list(row.get("new_columns") or []),
            "probe_score": row.get("probe_score"),
            "probe_se": row.get("probe_se"),
            "vs_source_table": row.get("vs_source_table"),
        })
    for index, row in enumerate(journal.get("leaderboard") or [], start=1):
        identifier = row.get("id") or index
        add(f"experiment:{identifier}", "experiment", {
            "model_type": row.get("model_type"),
            "source": _table_ref(row.get("source_table"), table_paths),
            "metric": row.get("metric"),
            "val_score": row.get("val_score"),
            "val_se": row.get("val_se"),
            "cv_score": row.get("cv_score"),
            "fold_spread": row.get("fold_spread"),
            "fit_seconds": row.get("fit_seconds"),
            "feature_count": len(row.get("features") or []),
            "target_transform": row.get("target_transform"),
        })
    add("stop", "run_state", {
        "refusals": (journal.get("stop") or {}).get("refusals", 0),
        "table_refusals": (journal.get("stop") or {}).get("table_refusals", {}),
        "finalized": True,
    })

    profile = (journal.get("split") or {}).get("profile") or "full"
    tools = [
        "inspect_data", "fit_encoding", "fit_group_stats", "filter_rows",
        "transform_features", "finalize_run",
    ]
    if profile == "full":
        tools.extend([
            "fit_history", "fit_stream_aggregate", "fit_reference_target",
            "explain_table", "explore_table", "explore_journal",
        ])
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "purpose": "diagnose generic limitations after finalization; never select a model",
        "profile": profile,
        "available_tools": tools,
        "evidence": evidence,
    }


def _validate_review(draft: CapabilityReviewDraft, context: dict[str, Any]) -> dict:
    evidence_ids = {item["id"] for item in context["evidence"]}
    findings = []
    for item in draft.findings:
        unknown = sorted(set(item.evidence_refs) - evidence_ids)
        if unknown:
            raise ValueError(f"capability reviewer cited unknown evidence IDs: {unknown}")
        row = item.model_dump()
        row["generic_capability_key"] = _stable_key(item.generic_capability_key)
        row["evidence_refs"] = list(dict.fromkeys(item.evidence_refs))
        row["fingerprint"] = _fingerprint(
            item.category, row["generic_capability_key"]
        )
        findings.append(row)
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "status": "proposed",
        "scope": "post-finalization diagnosis; no model or tool decision",
        "overall_assessment": draft.overall_assessment,
        "findings": findings,
    }


def _markdown_text(value: Any) -> str:
    """Keep reviewer prose from becoming active HTML in rendered Markdown."""
    return html_escape(str(value), quote=True).replace("\n", " ")


def _markdown(review: dict) -> str:
    lines = [
        MARKDOWN_START,
        "## What limited this run?",
        "",
        _markdown_text(review["overall_assessment"]),
        "",
    ]
    for finding in review["findings"]:
        category = finding["category"].replace("_", " ")
        lines.extend([
            f"### {_markdown_text(finding['title'])}", "",
            f"- Classification: **{category}** ({finding['confidence']} confidence)",
            f"- Generic capability: {_markdown_text(finding['generic_capability'])}",
            f"- Evidence: {', '.join(finding['evidence_refs'])}",
            f"- Why current tools are insufficient: {_markdown_text(finding['why_existing_tools_insufficient'])}",
            f"- Alternative explanation: {_markdown_text(finding['alternative_explanation'])}",
            f"- Leakage/serving risk: {_markdown_text(finding['leakage_or_serving_risk'])}",
            f"- Testable hypothesis: {_markdown_text(finding['testable_hypothesis'])}", "",
        ])
    lines.extend([
        "These are owner-reviewable hypotheses, not product decisions. The internal test",
        "was used only for retrospective diagnosis and ranked nothing.",
        MARKDOWN_END,
    ])
    return "\n".join(lines)


def _update_reports(run_dir: Path, review: dict) -> None:
    report_json = run_dir / "run_report.json"
    with open(report_json, encoding="utf-8") as handle:
        data = json.load(handle)
    data["capability_review"] = review
    _atomic_json(report_json, data)
    _atomic_text(run_dir / "run_report.html", render_dashboard_html(data))
    markdown_path = run_dir / "run_report.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    if MARKDOWN_START in markdown:
        markdown = markdown.split(MARKDOWN_START, 1)[0].rstrip()
    _atomic_text(markdown_path, f"{markdown}\n\n{_markdown(review)}\n")


def review_capabilities(
    run_dir: str | os.PathLike,
    *,
    model: Any,
) -> dict[str, Any]:
    """Review one finalized run once, with a tool-less structured-output agent."""
    root = Path(run_dir).resolve()
    artifact = root / REVIEW_FILENAME
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".capability-review.lock", "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if artifact.exists():
                with open(artifact, encoding="utf-8") as handle:
                    review = json.load(handle)
                _update_reports(root, review)
                return review
            context = build_review_context(root)
            from pydantic_ai import Agent

            reviewer = Agent(
                model,
                output_type=CapabilityReviewDraft,
                system_prompt=SYSTEM_PROMPT,
                defer_model_check=True,
            )
            result = reviewer.run_sync(json.dumps(context, ensure_ascii=False))
            review = _validate_review(result.output, context)
            _atomic_json(artifact, review)
            _update_reports(root, review)
            return review
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def capability_backlog(runs: list[Any]) -> dict[str, Any]:
    """Group matching run reviews; recurrence is evidence, never acceptance."""
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    seen_runs = set()
    for value in runs:
        run_dir = Path(getattr(value, "run_dir", value)).resolve()
        run_id = run_dir.name
        if run_id in seen_runs:
            continue
        seen_runs.add(run_id)
        artifact = run_dir / REVIEW_FILENAME
        if not artifact.exists():
            raise ValueError(f"run {run_id!r} has no capability review")
        with open(artifact, encoding="utf-8") as handle:
            review = json.load(handle)
        for finding in review.get("findings") or []:
            grouped[str(finding["fingerprint"])].append((run_id, finding))

    items = []
    for fingerprint, occurrences in grouped.items():
        sample = occurrences[0][1]
        run_ids = sorted({run_id for run_id, _ in occurrences})
        items.append({
            "fingerprint": fingerprint,
            "category": sample["category"],
            "generic_capability_key": sample["generic_capability_key"],
            "generic_capability": sample["generic_capability"],
            "status": "recurring" if len(run_ids) >= 2 else "proposed",
            "n_runs": len(run_ids),
            "run_ids": run_ids,
            "hypotheses": [row["testable_hypothesis"] for _, row in occurrences],
        })
    items.sort(key=lambda row: (-row["n_runs"], row["fingerprint"]))
    return {
        "schema_version": BACKLOG_SCHEMA_VERSION,
        "n_runs": len(seen_runs),
        "items": items,
        "note": (
            "Recurring findings are candidates for owner preregistration, not accepted "
            "product changes. Acceptance requires a test on an unseen run."
        ),
    }
