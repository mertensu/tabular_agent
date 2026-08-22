"""Reconstruct candidate-vs-incumbent block contrasts from stored probe predictions.

The primary rule and its falsifier were registered before this script was run in
``PREREGISTRATION.md``. This file deliberately keeps reconstruction separate from the
verdict so the controls can fail without being interpreted as a negative result.

Example after extracting ``artifacts.tar.gz``::

    uv run python research/archive/experiments/0080-0081-incumbent-threshold/analyze.py \
        --runs-root /tmp/tabular_incumbent_artifacts/_runs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_agent.compare import DEFAULT_BLOCKS, blockwise_difference
from tabular_agent.metrics import greater_is_better


PRIMARY_K = 0.25
SENSITIVITY_K = (0.0, PRIMARY_K, 0.5, 1.0)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def rule_metrics(rows: list[dict], k: float) -> dict:
    """Precision/recall for gain > k * block_sd against boolean ``truth``."""
    accepted = [row for row in rows if row["gain"] > k * row["block_sd"]]
    true_positive = sum(bool(row["truth"]) for row in accepted)
    truth_positive = sum(bool(row["truth"]) for row in rows)
    return {
        "accepted": len(accepted),
        "true_positive": true_positive,
        "truth_positive": truth_positive,
        "precision": None if not accepted else true_positive / len(accepted),
        "recall": None if not truth_positive else true_positive / truth_positive,
    }


def historical_low_incumbents(
    tables: list[dict], metric: str, work_probe: dict | None = None
) -> dict[str, dict]:
    """Candidate versus the best earlier scored table, preserving journal order."""
    higher = greater_is_better(metric)
    seen: list[dict] = []
    if work_probe and work_probe.get("probe_score") is not None:
        seen.append({"id": "work", "probe_score": float(work_probe["probe_score"])})

    out: dict[str, dict] = {}
    for table in tables:
        score = table.get("probe_score")
        if score is None:
            continue
        score = float(score)
        if seen:
            incumbent = max(seen, key=lambda row: row["probe_score"]) if higher else min(
                seen, key=lambda row: row["probe_score"]
            )
            gain = score - incumbent["probe_score"] if higher else incumbent["probe_score"] - score
            out[table["id"]] = {"incumbent": incumbent["id"], "gain": gain}
        seen.append({"id": table["id"], "probe_score": score})
    return out


def accumulate_high_utilities(edges: list[dict], anchor: str) -> dict[str, float]:
    """Turn lineage-edge gains into one relative utility per table."""
    utility = {anchor: 0.0}
    pending = list(edges)
    while pending:
        unresolved = []
        progressed = False
        for edge in pending:
            parent = edge["parent"]
            candidate = edge["candidate"]
            if parent not in utility:
                unresolved.append(edge)
                continue
            value = utility[parent] + float(edge["gain_high"])
            if candidate in utility and abs(utility[candidate] - value) > 1e-12:
                raise ValueError(f"conflicting high-capacity utility for {candidate}")
            utility[candidate] = value
            progressed = True
        if not progressed:
            missing = sorted({edge["parent"] for edge in unresolved})
            raise ValueError(f"cannot resolve high-capacity parent(s): {missing}")
        pending = unresolved
    return utility


def high_incumbent_contrasts(
    table_order: list[str], utility: dict[str, float]
) -> dict[str, dict]:
    """Candidate minus best earlier table, in positive-is-better units."""
    out: dict[str, dict] = {}
    seen: list[str] = []
    for table in table_order:
        if table not in utility:
            continue
        if seen:
            incumbent = max(seen, key=lambda name: utility[name])
            out[table] = {
                "incumbent": incumbent,
                "gain": utility[table] - utility[incumbent],
            }
        seen.append(table)
    return out


def _score_gain(candidate: float, reference: float, metric: str) -> float:
    delta = float(candidate) - float(reference)
    return delta if greater_is_better(metric) else -delta


def assert_score_gain_close(recomputed: float, from_rounded_scores: float, table_id: str) -> None:
    """Allow only the error induced by independently storing six-decimal scores."""
    if abs(float(recomputed) - float(from_rounded_scores)) > 2e-6:
        raise ValueError(
            f"{table_id} score gain mismatch: predictions={recomputed}, "
            f"rounded_scores={from_rounded_scores}"
        )


def count_tied_block_boundaries(order_values, n_blocks: int = DEFAULT_BLOCKS) -> int:
    """How many equal-row block cuts split one ordering value across two blocks."""
    values = np.sort(np.asarray(order_values), kind="stable")
    edges = np.linspace(0, len(values), n_blocks + 1).astype(int)[1:-1]
    return int(sum(values[edge - 1] == values[edge] for edge in edges))


def _prediction_path(run_dir: Path, table_id: str) -> Path:
    return run_dir / "probe_preds" / f"{table_id}.npy"


def _load_prediction(run_dir: Path, table_id: str, n_rows: int) -> np.ndarray:
    path = _prediction_path(run_dir, table_id)
    if not path.exists():
        raise FileNotFoundError(f"missing prediction sidecar: {path}")
    values = np.load(path)
    if len(values) != n_rows:
        raise ValueError(f"{path} has {len(values)} predictions for {n_rows} validation rows")
    return values


def _assert_parent_control(stored: dict, recomputed: dict, table_id: str) -> None:
    """The stored parent interval proves row alignment and historical reproduction."""
    if not stored:
        raise ValueError(f"{table_id} has no stored parent interval")
    for key in ("gain", "block_sd"):
        if float(stored[key]) != float(recomputed[key]):
            raise ValueError(
                f"{table_id} parent {key} mismatch: stored={stored[key]} "
                f"recomputed={recomputed[key]}"
            )
    if list(stored["block_gains"]) != list(recomputed["block_gains"]):
        raise ValueError(f"{table_id} parent block_gains do not reproduce")
    for key in ("sign_stable", "blocks_favouring_a", "n_blocks", "n_rows", "mode"):
        if stored.get(key) != recomputed.get(key):
            raise ValueError(f"{table_id} parent {key} does not reproduce")


def _contrast(
    *, metric: str, y, candidate, reference, order_values, labels
) -> dict:
    result = blockwise_difference(
        metric, y, candidate, reference, order_values, labels=labels
    )
    if result is None:
        raise ValueError("blockwise_difference returned no result")
    return result


def reconstruct_run(run_dir: Path) -> tuple[list[dict], dict]:
    """Rebuild parent, derived-incumbent and work-aware contrasts for one run."""
    journal = json.loads((run_dir / "run_journal.json").read_text())
    task = journal["task"]
    split = journal["split"]
    metric = task["metric"]
    val = pd.read_parquet(run_dir / "data" / "val.parquet")
    y = val[task["target"]].to_numpy()
    order_values = val[split["split_on"]].to_numpy()
    labels = sorted(pd.unique(y)) if task["task_type"] in ("binary", "multiclass") else None
    tables = [table for table in journal.get("tables") or [] if table.get("probe_score") is not None]
    table_by_id = {table["id"]: table for table in tables}
    if len(table_by_id) != len(tables):
        raise ValueError(f"duplicate table id in {run_dir.name}")

    derived_incumbents = historical_low_incumbents(tables, metric)
    work_incumbents = historical_low_incumbents(tables, metric, journal.get("work_probe"))
    work_score = (journal.get("work_probe") or {}).get("probe_score")
    score_by_id = {"work": work_score, **{table["id"]: table["probe_score"] for table in tables}}
    rows = []

    for table in tables:
        candidate_id = table["id"]
        parent_id = Path(table["source_table"]).stem
        candidate_preds = _load_prediction(run_dir, candidate_id, len(y))
        parent_preds = _load_prediction(run_dir, parent_id, len(y))
        parent = _contrast(
            metric=metric, y=y, candidate=candidate_preds, reference=parent_preds,
            order_values=order_values, labels=labels,
        )
        _assert_parent_control(table.get("probe_vs_source_interval"), parent, candidate_id)
        expected_parent_gain = _score_gain(
            table["probe_score"], score_by_id[parent_id], metric
        )
        if table.get("probe_vs_source_table") != round(expected_parent_gain, 6):
            raise ValueError(f"{candidate_id} stored parent scalar does not reproduce")
        assert_score_gain_close(parent["gain"], expected_parent_gain, candidate_id)

        row = {
            "run": run_dir.name,
            "metric": metric,
            "candidate": candidate_id,
            "method": table.get("method"),
            "parent": parent_id,
            "parent_gain": parent["gain"],
            "parent_block_sd": parent["block_sd"],
            "parent_block_gains": json.dumps(parent["block_gains"]),
            "parent_sign_stable": parent["sign_stable"],
        }

        derived = derived_incumbents.get(candidate_id)
        if derived:
            incumbent_id = derived["incumbent"]
            incumbent_preds = _load_prediction(run_dir, incumbent_id, len(y))
            incumbent = _contrast(
                metric=metric, y=y, candidate=candidate_preds, reference=incumbent_preds,
                order_values=order_values, labels=labels,
            )
            assert_score_gain_close(incumbent["gain"], derived["gain"], candidate_id)
            row.update({
                "incumbent": incumbent_id,
                "incumbent_gain": incumbent["gain"],
                "incumbent_block_sd": incumbent["block_sd"],
                "incumbent_block_gains": json.dumps(incumbent["block_gains"]),
                "incumbent_sign_stable": incumbent["sign_stable"],
            })

        work_aware = work_incumbents.get(candidate_id)
        if work_aware:
            work_incumbent_id = work_aware["incumbent"]
            work_incumbent_preds = _load_prediction(run_dir, work_incumbent_id, len(y))
            work_incumbent = _contrast(
                metric=metric, y=y, candidate=candidate_preds,
                reference=work_incumbent_preds, order_values=order_values, labels=labels,
            )
            assert_score_gain_close(work_incumbent["gain"], work_aware["gain"], candidate_id)
            row.update({
                "work_incumbent": work_incumbent_id,
                "work_incumbent_gain": work_incumbent["gain"],
                "work_incumbent_block_sd": work_incumbent["block_sd"],
                "work_incumbent_block_gains": json.dumps(work_incumbent["block_gains"]),
                "work_incumbent_sign_stable": work_incumbent["sign_stable"],
            })
        rows.append(row)

    control = {
        "run": run_dir.name,
        "tables": len(tables),
        "validation_rows": len(y),
        "distinct_order_values": int(pd.Series(order_values).nunique(dropna=False)),
        "tied_block_boundaries": count_tied_block_boundaries(order_values),
        "internal_block_boundaries": DEFAULT_BLOCKS - 1,
        "parent_intervals_reproduced": len(rows),
        "derived_incumbent_contrasts": sum("incumbent" in row for row in rows),
        "work_aware_contrasts": sum("work_incumbent" in row for row in rows),
    }
    return rows, control


def _capacity_edges(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["run"] not in {"restaurant_v1", "restaurant_v2"}:
                continue
            parent, candidate = row["step"].split("->")
            grouped.setdefault(row["run"], []).append({
                "parent": parent,
                "candidate": candidate,
                "gain_high": float(row["gain_high"]),
            })
    return grouped


def attach_high_capacity_outcomes(
    rows: list[dict], controls: list[dict], capacity_csv: Path, journals: dict[str, dict]
) -> list[dict]:
    """Attach incumbent outcomes reconstructed from six-decimal high-budget edges."""
    by_run = _capacity_edges(capacity_csv)
    row_by_key = {(row["run"], row["candidate"]): row for row in rows}
    labeled = []
    for run, edges in by_run.items():
        edge_candidates = [edge["candidate"] for edge in edges]
        if len(edge_candidates) != len(set(edge_candidates)):
            raise ValueError(f"{run} has duplicate high-capacity candidate edges")
        parents = {edge["parent"] for edge in edges}
        candidates = {edge["candidate"] for edge in edges}
        roots = sorted(parents - candidates)
        if len(roots) != 1:
            raise ValueError(f"{run} needs exactly one high-capacity anchor, got {roots}")
        utility = accumulate_high_utilities(edges, anchor=roots[0])
        order = [table["id"] for table in journals[run].get("tables") or []]
        high = high_incumbent_contrasts(order, utility)
        for candidate, contrast in high.items():
            key = (run, candidate)
            if key not in row_by_key:
                raise ValueError(f"missing reconstructed low contrast for {run} {candidate}")
            row = row_by_key[key]
            if row.get("incumbent") is None:
                raise ValueError(f"{run} {candidate} has no low incumbent")
            row.update({
                "capacity_incumbent": contrast["incumbent"],
                "capacity_incumbent_gain": contrast["gain"],
                "high_capacity_positive": contrast["gain"] > 0,
            })
            labeled.append(row)
        controls.append({
            "run": run,
            "capacity_edges": len(edges),
            "capacity_utilities_resolved": len(utility),
            "capacity_incumbent_labels": len(high),
            "incumbent_identity_differs_across_budgets": sum(
                row_by_key[(run, candidate)].get("incumbent") != contrast["incumbent"]
                for candidate, contrast in high.items()
            ),
        })
    return labeled


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(runs_root: Path, capacity_csv: Path) -> tuple[list[dict], dict]:
    rows = []
    controls = []
    journals = {}
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        journal_path = run_dir / "run_journal.json"
        if not journal_path.exists():
            continue
        journals[run_dir.name] = json.loads(journal_path.read_text())
        run_rows, control = reconstruct_run(run_dir)
        rows.extend(run_rows)
        controls.append(control)

    labeled = attach_high_capacity_outcomes(rows, controls, capacity_csv, journals)
    verdict_rows = [
        {
            "gain": float(row["incumbent_gain"]),
            "block_sd": float(row["incumbent_block_sd"]),
            "truth": bool(row["high_capacity_positive"]),
        }
        for row in labeled
    ]
    rules = {str(k): rule_metrics(verdict_rows, k) for k in SENSITIVITY_K}
    current = rules["0.0"]
    quarter = rules[str(PRIMARY_K)]
    primary_pass = bool(
        quarter["precision"] is not None
        and current["precision"] is not None
        and quarter["precision"] > current["precision"]
        and quarter["recall"] is not None
        and quarter["recall"] >= 0.5
    )
    summary = {
        "registered_primary_k": PRIMARY_K,
        "high_capacity_labeled_steps": len(labeled),
        "runs_with_high_capacity_outcomes": sorted({row["run"] for row in labeled}),
        "rules": rules,
        "registered_prediction_passed": primary_pass,
        "controls": controls,
        "limits": [
            "Two runs from one dataset family; steps are adaptive and dependent.",
            "IEEE contrasts are descriptive because no high-capacity reruling is available.",
            "Historical row-count blocks may split tied ordering values.",
        ],
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("_runs"))
    parser.add_argument(
        "--capacity-csv", type=Path,
        default=REPO / "results/ruler_deflation_across_runs.csv",
    )
    parser.add_argument(
        "--out-csv", type=Path,
        default=HERE / "contrasts.csv",
    )
    parser.add_argument(
        "--out-summary", type=Path,
        default=HERE / "summary.json",
    )
    args = parser.parse_args()

    rows, summary = run_analysis(args.runs_root, args.capacity_csv)
    _write_csv(args.out_csv, rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
