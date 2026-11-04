"""Finalize a stopped Kaggle-split run as the default ships it (0100), and predict.

    uv run python research/ieee_cross_row/finish.py k1

k1 was stopped during its blend fits, with its table already chosen. The run
directory is the whole state, so this resumes it with `blend=False`: nothing is
searched or chosen again. Writes `_runs/ieee_cross_row/<tag>/submission_<tag>_lgbm.csv`
and `result_<tag>.json`. No LLM calls.
"""

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runner import OUT, TARGET, TEST_DAYS, DAY  # noqa: E402
import prep_ieee as prep  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import finalize_run  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402


def main(tag: str = "k1") -> int:
    run_dir = sorted((OUT / tag / "runs").glob("*"))[-1]
    manifest = json.loads((run_dir / "prepared" / "api_plan.json").read_text())
    spec = TaskSpec(
        target=TARGET, metric=prep.METRIC, notes=prep.ABOUT, column_notes=prep.COLUMN_NOTES,
        forecast_horizon=prep.FORECAST_HORIZON, forecast_length=TEST_DAYS * DAY,
    )
    session = RunSession(
        str(run_dir), str(run_dir / "prepared" / "train.parquet"), spec,
        val_fraction=0.2, split_on=prep.SPLIT_ON,
        serving_columns=manifest["serving_columns"], future_known=[],
        fit_time_budget_s=20 * 60, profile="full", workflow="forecast", blend=False,
    )
    out = finalize_run(session)
    shutil.copy2(run_dir / "prepared" / "api_plan.json", run_dir / "bundle" / "api_plan.json")
    print(json.dumps({k: out.get(k) for k in ("model_type", "bundle_dir")}), flush=True)

    test = pd.read_parquet(prep.TEST_OUT)
    p = np.asarray(Run(run_dir).bundle.predict(test), dtype=float)
    if p.ndim == 2:
        p = p[:, -1]
    path = OUT / tag / f"submission_{tag}_lgbm.csv"
    pd.DataFrame({"TransactionID": test["TransactionID"], TARGET: p}).to_csv(path, index=False)
    chosen = json.loads((run_dir / "run_journal.json").read_text())["terminal_selection"]
    result = {"tag": tag, "run_dir": str(run_dir), "shipped": "one early-stopped lgbm (0100)",
              "table": chosen["chosen"], "rounds": chosen["rounds"][chosen["chosen"]],
              "val_auc": chosen["scores"][chosen["chosen"]], "submission": str(path),
              "mean_prediction": round(float(p.mean()), 4)}
    (HERE / f"result_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
