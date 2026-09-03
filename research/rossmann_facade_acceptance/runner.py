"""One live Rossmann acceptance run through the public P0 facade."""

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

import tabular_agent as ta

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "rossmann_store_sales"
RUN_ROOT = REPO / "_runs" / "rossmann_facade_acceptance"
MODEL = "fireworks/deepseek-v4-flash-0731"

sys.path.insert(0, str(REPO / "scripts"))
from prep_rossmann import ABOUT, COLUMN_NOTES, METRIC  # noqa: E402

PROMPT = """Build the best safe, replayable model you can for this daily store-sales
dataset. The target is Sales. The unseen competition block spans 48 days and no
Sales values are revealed while it is forecast. Work until further feature ideas
stop producing real progress, then finalize the run."""


def main() -> int:
    agent = ta.Agent(
        model=MODEL,
        profile="full",
        run_root=RUN_ROOT,
        val_fraction=0.05,
        holdout_fraction=0.05,
        fit_time_budget_s=20 * 60,
    )
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
        if join.right_table == "store.csv":
            join.accept()
        else:
            join.reject()
    plan.validate()

    contract = {
        "training_table": plan.training_table,
        "serving_table": plan.serving_table,
        "target": plan.target,
        "metric": plan.metric,
        "split_on": plan.split_on,
        "forecast_horizon": plan.forecast_horizon,
        "serving_columns": plan.serving_columns,
        "identifier_columns": plan.identifier_columns,
        "joins": [vars(join) for join in plan.joins],
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RUN_ROOT / "confirmed_facade_plan.json", "w") as handle:
        json.dump(contract, handle, indent=2)
    print(json.dumps(contract, indent=2), flush=True)

    started = time.time()
    run = agent.run(plan, prompt=PROMPT)
    predictions = run.bundle.predict(DATA)
    result = {
        "run_dir": run.run_dir,
        "elapsed_min": round((time.time() - started) / 60, 2),
        "n_predictions": int(len(predictions)),
        "predictions_finite": bool(np.isfinite(predictions).all()),
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "report_path": str(Path(run.run_dir) / "run_report.md"),
        "journal_path": str(Path(run.run_dir) / "run_journal.json"),
        "bundle_path": run.bundle.path,
    }
    with open(RUN_ROOT / "result.json", "w") as handle:
        json.dump(result, handle, indent=2)
    pd.DataFrame({"Id": test["Id"], "Sales": predictions}).to_csv(
        RUN_ROOT / "submission.csv", index=False
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
