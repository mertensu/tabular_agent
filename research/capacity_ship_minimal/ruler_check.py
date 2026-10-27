"""Could the run's own ruler have seen what the closure windows are worth? (no LLM)

    uv run python research/capacity_ship_minimal/ruler_check.py r4

Offline, short windows over `Open` (closed yesterday / tomorrow, open days in the
last / next 7) added +0.0064 rmspe on the 48 test days (rossmann_open_windows.py).
In r4 the agent never built those windows. This asks whether it would have helped
if it had: judge the same windows the way the run judges any table.

For a source table of the r4 run (t16, its best; t5, what shipped):
  1. replay the table's lineage on the run's validation rows;
  2. add the `Open` windows, fitted on the training rows, served to the validation
     rows with the stored tail (0093);
  3. fit the run's ruler (`fit_probe`, 200 trees, fixed seed) on the training rows
     with and without the windows, predict the validation rows;
  4. compare the two with `compare_predictions` and apply the run's selection rule
     (0090, `stop._has_demonstrated_gain`): gain larger than its block spread.

Also scores the same windows plus `SchoolHoliday` and `Promo`. Writes
`ruler_check_rossmann_<tag>.json`.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rossmann_open_windows import WINDOWS  # noqa: E402
from runner import HERE  # noqa: E402

from tabular_agent.compare import compare_predictions  # noqa: E402
from tabular_agent.encoders import FutureWindowEncoder  # noqa: E402
from tabular_agent.harness.stop import _has_demonstrated_gain  # noqa: E402
from tabular_agent.harness.tools import _feature_frame, _load_lineage  # noqa: E402
from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.probe import fit_probe, probe_predict  # noqa: E402

TARGET, METRIC, GROUP, ORDER, BLOCK = "Sales", "rmspe", "Store", "Date", 48.0


def ruler(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], task: dict) -> np.ndarray:
    fit = fit_probe(train[cols], train[TARGET], "regression", METRIC,
                    objective=task.get("objective"))
    return np.asarray(probe_predict(fit, val[cols], "regression"), dtype=float)


def main(tag: str = "r4") -> int:
    result = json.loads((HERE / f"result_rossmann_{tag}.json").read_text())
    root = Path(result["run_dir"])
    task = json.loads((root / "run_journal.json").read_text())["task"]
    work = pd.read_parquet(root / "data" / "work.parquet")
    val_raw = pd.read_parquet(root / "data" / "val.parquet")
    y_val = val_raw[TARGET].to_numpy()
    order = val_raw[ORDER].to_numpy()

    # Windows fitted on the training rows, served to the validation rows alone.
    added, train_win, val_win = {}, [], []
    for value in ("Open", "SchoolHoliday", "Promo"):
        enc = FutureWindowEncoder([GROUP], value, ORDER, BLOCK, WINDOWS).fit(work)
        train_win.append(enc.transform(work)[enc.output_columns])
        val_win.append(enc.transform(val_raw.drop(columns=[TARGET]))[enc.output_columns])
        added[value] = enc.output_columns
    train_win, val_win = pd.concat(train_win, axis=1), pd.concat(val_win, axis=1)

    out = {"tag": tag, "run_dir": str(root), "n_val_rows": int(len(val_raw)), "tables": {}}
    for table_id in ("t16", "t5"):
        path = root / "tables" / f"{table_id}.parquet"
        table = pd.read_parquet(path)
        if len(table) != len(work):
            raise SystemExit(f"{table_id} was filtered; this check needs unfiltered rows")
        X, _ = _feature_frame(table, TARGET)
        base_cols = list(X.columns)
        val = _load_lineage(str(path), "regression").transform(val_raw.drop(columns=[TARGET]))
        # A window the agent already built under the same name stays the table's own.
        new = [c for c in train_win.columns if c not in table.columns]
        train = pd.concat([table.reset_index(drop=True), train_win[new]], axis=1)
        val = pd.concat([val.reset_index(drop=True), val_win[new]], axis=1)

        def plus(*values):
            return base_cols + [c for v in values for c in added[v] if c in new]

        base = ruler(train, val, base_cols, task)
        arms = {"open": plus("Open"), "open_calendar": plus("Open", "SchoolHoliday", "Promo")}
        row = {"base_score": round(compute_metric(METRIC, y_val, base), 6)}
        for arm, cols in arms.items():
            p = ruler(train, val, cols, task)
            interval = compare_predictions(METRIC, y_val, p, base, order_values=order)
            row[arm] = {
                "score": round(compute_metric(METRIC, y_val, p), 6),
                "interval": interval,
                "would_be_selectable": _has_demonstrated_gain(
                    {"probe_vs_source_interval": interval}),
            }
        out["tables"][table_id] = row
        print(table_id, json.dumps(row, indent=1, default=str), flush=True)

    (HERE / f"ruler_check_rossmann_{tag}.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
