"""Three preregistered Rossmann repeats for Decision 0089."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

import tabular_agent as ta
from tabular_agent.harness.agent import gateway_model

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "rossmann_store_sales"
RUN_ROOT = REPO / "_runs" / "rossmann_forecasting_agent"
RESULT_ROOT = Path(__file__).resolve().parent / "results"
MODEL = "deepseek-v4-flash-0731@eu"

sys.path.insert(0, str(REPO / "scripts"))
from prep_rossmann import ABOUT, COLUMN_NOTES, METRIC  # noqa: E402

PROMPT = """Build the best safe, replayable model you can for this daily store-sales
dataset. The target is Sales. The unseen competition block spans 48 days and no
Sales values are revealed while it is forecast. Work until further feature ideas
stop producing real progress, then finalize the run."""


def configured_plan(agent: ta.ForecastingAgent) -> ta.DataPlan:
    plan = agent.inspect(DATA)
    test = pd.read_csv(DATA / "test.csv")
    static_columns = [
        column
        for join in plan.joins
        if join.right_table == "store.csv"
        for column in join.columns
    ]
    plan.target = "Sales"
    plan.metric = METRIC
    plan.notes = ABOUT
    plan.column_notes = COLUMN_NOTES
    plan.forecast_horizon = 48.0
    plan.split_on = "Date"
    plan.serving_columns = [
        column for column in test.columns if column != "Id"
    ] + static_columns
    plan.reject_identifier_proposals()
    for join in plan.joins:
        (join.accept() if join.right_table == "store.csv" else join.reject())
    plan.validate()
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repeat", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    repeat_root = RUN_ROOT / f"repeat_{args.repeat:02d}"
    if repeat_root.exists():
        raise SystemExit(f"refusing to overwrite {repeat_root}")

    model = gateway_model(MODEL)
    agent = ta.ForecastingAgent(
        model=model,
        run_root=repeat_root,
        val_fraction=0.05,
        holdout_fraction=0.05,
        fit_time_budget_s=20 * 60,
    )
    plan = configured_plan(agent)

    started = time.time()
    run = agent.run(plan, prompt=PROMPT)
    elapsed_min = (time.time() - started) / 60
    report = json.loads(run.report("json"))
    journal = run.journal
    histories = [
        {
            "id": row.get("id"),
            "source_table": row.get("source_table"),
            "new_columns": row.get("new_columns") or row.get("columns"),
            "probe_score": row.get("probe_score"),
            "probe_vs_source_table": row.get("probe_vs_source_table"),
        }
        for row in journal.get("tables", [])
        if row.get("method") == "history"
    ]
    predictions = run.bundle.predict(DATA)
    test = pd.read_csv(DATA / "test.csv")
    repeat_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Id": test["Id"], "Sales": predictions}).to_csv(
        repeat_root / "submission.csv", index=False
    )

    review = agent.review_capabilities(run)
    result = {
        "acceptance": "forecasting_agent_repeat",
        "repeat": args.repeat,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "model": MODEL,
        "run_dir": run.run_dir,
        "elapsed_minutes": round(elapsed_min, 2),
        "workflow": (journal.get("split") or {}).get("workflow"),
        "forecast_horizon": (journal.get("split") or {}).get("forecast_horizon"),
        "history_attempted": bool(histories),
        "history_tables": histories,
        "n_tables": len(journal.get("tables") or []),
        "shipped_feature_count": len(run.bundle.feature_names),
        "summary": report.get("summary"),
        "models": report.get("models"),
        "predictions": {
            "rows": int(len(predictions)),
            "finite": bool(np.isfinite(predictions).all()),
            "minimum": float(np.min(predictions)),
            "maximum": float(np.max(predictions)),
        },
        "capability_review": review,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = RESULT_ROOT / f"repeat_{args.repeat:02d}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "result": str(destination),
        "run_dir": run.run_dir,
        "elapsed_minutes": result["elapsed_minutes"],
        "history_attempted": result["history_attempted"],
        "history_tables": len(histories),
        "validation_score": (report.get("summary") or {}).get("validation_score"),
        "test_score": (report.get("summary") or {}).get("test_score"),
        "features": result["shipped_feature_count"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
