"""One run of ForecastingAgent on IEEE-CIS, with a later test block it never sees.

    uv run python research/ieee_cross_row/runner.py [tag]

The competition's geometry, rebuilt inside its labelled table: the last 30 days of
`train_joined.parquet` are the test block, the 30 days before them are dropped (the
competition's gap), and the agent gets everything earlier. Writes
`result_<tag>.json` with the shipped bundle's AUC on the test block. See README.md.

A tag starting with `k` is the competition's own split instead: the agent gets every
labelled row, and the shipped bundle predicts Kaggle's test file into
`_runs/ieee_cross_row/<tag>/submission_<tag>.csv`, to be submitted once.
"""

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = REPO / "_runs" / "ieee_cross_row"
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research" / "capacity_ship_minimal"))

import prep_ieee as prep  # noqa: E402

TARGET = "isFraud"
DAY = 24 * 3600
TEST_DAYS = GAP_DAYS = 30
PROMPT = (
    "Build the best model you can for this dataset of payment transactions. The "
    "target is isFraud. Work until you stop making progress, then finalize."
)
#: The agent model of the credit default, freMTPL2 and Otto runs.
MODEL, EFFORT = "gpt-6.1-terra@eu", "medium"


def split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(agent rows, test rows): the last 30 days are the test, the 30 before them a gap."""
    df = pd.read_parquet(prep.OUT)
    end = df[prep.SPLIT_ON].max()
    test = df[df[prep.SPLIT_ON] >= end - TEST_DAYS * DAY]
    train = df[df[prep.SPLIT_ON] < end - (TEST_DAYS + GAP_DAYS) * DAY]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main(tag: str = "r1") -> int:
    import tabular_agent as ta
    from runner import agent_model
    from tabular_agent.metrics import compute_metric

    kaggle = tag.startswith("k")
    if kaggle:
        train, test = pd.read_parquet(prep.OUT), pd.read_parquet(prep.TEST_OUT)
    else:
        train, test = split()
    files = OUT / tag / "files"
    files.mkdir(parents=True, exist_ok=True)
    train.to_parquet(files / "train.parquet", index=False)
    # The serving schema only: five training rows without the target.
    train.drop(columns=[TARGET]).head(5).to_parquet(files / "test.parquet", index=False)

    agent = ta.ForecastingAgent(model=agent_model(MODEL, EFFORT), run_root=OUT / tag / "runs",
                                fit_time_budget_s=20 * 60)
    plan = agent.inspect(files)
    plan.target = TARGET
    plan.metric = prep.METRIC
    days = int(round((train[prep.SPLIT_ON].max() - train[prep.SPLIT_ON].min()) / DAY))
    plan.notes = prep.ABOUT.replace("spanning 182 days", f"spanning {days} days")
    plan.column_notes = prep.COLUMN_NOTES
    train_cols = set(plan.tables[plan.training_table]["columns"])
    plan.serving_columns = [c for c in (plan.proposed_serving_columns or []) if c in train_cols]
    plan.accept_identifier_proposals()
    plan.split_on = prep.SPLIT_ON
    plan.forecast_horizon = prep.FORECAST_HORIZON
    plan.forecast_length = TEST_DAYS * DAY
    plan.future_known = []
    plan.reject_joins()
    plan.validate()

    started = time.time()
    run = agent.run(plan, prompt=PROMPT)
    elapsed = time.time() - started
    p = np.asarray(run.bundle.predict(test.drop(columns=[TARGET], errors="ignore")),
                   dtype=float)
    if p.ndim == 2:
        p = p[:, -1]
    if kaggle:
        path = OUT / tag / f"submission_{tag}.csv"
        pd.DataFrame({"TransactionID": test["TransactionID"], TARGET: p}).to_csv(path,
                                                                                  index=False)
    result = {
        "tag": tag,
        "model": MODEL,
        "reasoning_effort": EFFORT,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "run_dir": str(run.run_dir),
        "elapsed_min": round(elapsed / 60, 2),
        "identifier_columns": plan.identifier_columns,
        "n_train": len(train),
        "n_test": len(test),
        "test_auc": None if kaggle else round(
            float(compute_metric("auc", test[TARGET].to_numpy(), p)), 6),
    }
    (HERE / f"result_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
