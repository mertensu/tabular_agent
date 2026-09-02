"""One live Titanic run through the public P0 facade."""

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

import tabular_agent as ta


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "_runs" / "titanic_facade_acceptance" / "data" / "files"
RUN_ROOT = REPO / "_runs" / "titanic_facade_acceptance" / "runs"
MODEL = "fireworks/deepseek-v4-flash-0731"

PROMPT = """Build the best safe, replayable classifier you can for Titanic survival.
The target is Survived and the competition metric is accuracy. Use only train.csv;
gender_submission.csv is a submission template, never training evidence. Derive
generalizable passenger, family, title, ticket, fare, and cabin features where they
measure real validation progress. Work until further ideas stop producing real
progress, then finalize the run."""


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
    plan.target = "Survived"
    plan.metric = "accuracy"
    plan.notes = (
        "Binary passenger-survival classification. PassengerId identifies rows; "
        "the official test.csv defines the serving schema."
    )
    plan.serving_columns = list(plan.proposed_serving_columns or [])
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
        "serving_columns": plan.serving_columns,
        "joins": [vars(join) for join in plan.joins],
    }
    (RUN_ROOT.parent / "confirmed_facade_plan.json").write_text(
        json.dumps(contract, indent=2)
    )
    print(json.dumps(contract, indent=2), flush=True)

    started = time.time()
    run = agent.run(plan, prompt=PROMPT)
    test = pd.read_csv(DATA / "test.csv", low_memory=False)
    predictions = np.asarray(run.bundle.predict(test))
    if len(predictions) != len(test):
        raise ValueError(f"expected {len(test)} predictions, got {len(predictions)}")
    if not np.isfinite(predictions.astype(float)).all():
        raise ValueError("bundle produced non-finite Titanic predictions")
    labels = (predictions >= 0.5).astype(int)
    if set(np.unique(labels)) - {0, 1}:  # pragma: no cover - explicit contract guard
        raise ValueError(f"thresholding produced non-binary labels: {np.unique(labels)!r}")

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": labels})
    submission_path = RUN_ROOT.parent / "submission.csv"
    submission.to_csv(submission_path, index=False)
    result = {
        "run_dir": run.run_dir,
        "elapsed_min": round((time.time() - started) / 60, 2),
        "n_predictions": int(len(labels)),
        "class_counts": submission["Survived"].value_counts().sort_index().to_dict(),
        "probability_min": float(predictions.min()),
        "probability_max": float(predictions.max()),
        "report_path": str(Path(run.run_dir) / "run_report.md"),
        "journal_path": str(Path(run.run_dir) / "run_journal.json"),
        "bundle_path": run.bundle.path,
        "submission_path": str(submission_path),
    }
    (RUN_ROOT.parent / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
