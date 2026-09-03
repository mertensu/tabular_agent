"""One live House Prices run through the public P0 facade (a second practice
dataset after Titanic, requested 2026-09-03). Regression + `rmsle`, so it also
exercises the log1p target path and the identifier role (`Id`)."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import tabular_agent as ta

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "_runs" / "houseprices_facade_acceptance" / "data" / "files"
RUN_ROOT = REPO / "_runs" / "houseprices_facade_acceptance" / "runs"
MODEL = "fireworks/deepseek-v4-flash-0731"

PROMPT = """Build the best safe, replayable regressor you can for house sale prices.
The target is SalePrice and the competition metric is rmsle. Use only train.csv;
sample_submission.csv is a submission template, never training evidence. Derive
generalizable features (quality, size, age, neighborhood, condition) where they
measure real validation progress, and use fold-safe encodings for the many
categoricals. Work until further ideas stop producing real progress, then finalize."""


def main() -> int:
    agent = ta.Agent(
        model=MODEL,
        profile="full",
        run_root=RUN_ROOT,
        val_fraction=0.2,
        holdout_fraction=0.2,
        fit_time_budget_s=15 * 60,
    )
    plan = agent.inspect(DATA)
    plan.target = "SalePrice"
    plan.metric = "rmsle"
    plan.notes = (
        "Regression of a house's sale price. `Id` identifies rows only; the "
        "official test.csv defines the serving schema. Metric rmsle scores the "
        "log of the price, so relative error matters."
    )
    plan.serving_columns = list(plan.proposed_serving_columns or [])
    plan.accept_identifier_proposals()
    plan.split_on = False
    plan.reject_joins()
    plan.validate()

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    contract = {
        "training_table": plan.training_table,
        "serving_table": plan.serving_table,
        "target": plan.target,
        "metric": plan.metric,
        "split_on": plan.split_on,
        "identifier_columns": plan.identifier_columns,
        "n_serving_columns": len(plan.serving_columns),
    }
    (RUN_ROOT.parent / "confirmed_facade_plan.json").write_text(json.dumps(contract, indent=2))
    print(json.dumps(contract, indent=2), flush=True)

    started = time.time()
    run = agent.run(plan, prompt=PROMPT)
    test = pd.read_csv(DATA / "test.csv", low_memory=False)
    preds = np.asarray(run.bundle.predict(test), dtype=float)
    if len(preds) != len(test):
        raise ValueError(f"expected {len(test)} predictions, got {len(preds)}")
    if not np.isfinite(preds).all():
        raise ValueError("bundle produced non-finite SalePrice predictions")

    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": preds})
    submission_path = RUN_ROOT.parent / "submission.csv"
    submission.to_csv(submission_path, index=False)
    result = {
        "run_dir": run.run_dir,
        "elapsed_min": round((time.time() - started) / 60, 2),
        "n_predictions": int(len(preds)),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "pred_below_zero": int((preds < 0).sum()),
        "report_path": str(Path(run.run_dir) / "run_report.md"),
        "bundle_path": run.bundle.path,
        "submission_path": str(submission_path),
    }
    (RUN_ROOT.parent / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
