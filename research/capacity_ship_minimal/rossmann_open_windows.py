"""Do windows over the known-in-advance `Open` calendar help Rossmann? (offline, no LLM)

    uv run python research/capacity_ship_minimal/rossmann_open_windows.py r3

Takes the table Rossmann r3 shipped (refitted on training + validation) and adds
windows over `Open` per store, built with the product's own encoder
(`FutureWindowEncoder`, 0093): closed days in the last and next 2 and 7 days.
The idea is the winner's closure features: sales jump right before and after a
store is shut, and the calendar of closures is known in advance.

The windows are computed on every row, closed days included, and then joined to
the training table: r3's lineage drops closed days first, and a window over
`Open` computed after that would only ever see 1s. At prediction the request
holds the closed days too; the encoder's stored tail carries the days before it.

Arms, all LightGBM with the early-stopping protocol of evaluate.py, scored on the
48 held-out test days with a store-level paired bootstrap:
  base            the shipped table's features
  open            + the `Open` windows
  open_calendar   + the same windows over `SchoolHoliday` and `Promo`

Writes `open_windows_rossmann_<tag>.json`.
"""

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import _lgbm, capacity_arm, fitted_rows, paired, score  # noqa: E402
from runner import HERE, OUT  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.encoders import FutureWindowEncoder  # noqa: E402

TARGET, METRIC, GROUP, ORDER = "Sales", "rmspe", "Store", "Date"
BLOCK = 48.0
WINDOWS = [
    {"family": "trailing", "window": 2, "stat": "min"},   # 0 if closed yesterday
    {"family": "trailing", "window": 7, "stat": "sum"},   # open days in the last week
    {"family": "forward", "window": 2, "stat": "min"},    # 0 if closed tomorrow
    {"family": "forward", "window": 7, "stat": "sum"},    # open days in the next week
]


def windows(value: str, history: pd.DataFrame, audit: pd.DataFrame):
    """The value's windows on training rows and on the test days served alone."""
    enc = FutureWindowEncoder([GROUP], value, ORDER, BLOCK, WINDOWS).fit(history)
    keys = [GROUP, ORDER]
    train = enc.transform(history)[keys + enc.output_columns]
    served = enc.transform(audit)[keys + enc.output_columns]
    return train, served, enc.output_columns


def main(tag: str = "r3") -> int:
    result = json.loads((HERE / f"result_rossmann_{tag}.json").read_text())
    run = Run(result["run_dir"])
    audit = pd.read_parquet(OUT / "rossmann" / "audit.parquet")
    audit[ORDER] = pd.to_datetime(audit[ORDER])
    y_audit = audit[TARGET].to_numpy()

    selected = run.training_features()
    table = pd.read_parquet(Path(run.run_dir) / run.bundle._bundle["training_table"])
    base_cols = [c for c in selected.columns if c != TARGET]
    keep = list(dict.fromkeys([GROUP, ORDER, *base_cols]))   # Store may be a feature too
    train = table[keep + [TARGET]].copy()
    test = run.bundle.feature_table(audit.drop(columns=[TARGET]), include_input=True)[keep]

    history = fitted_rows(run)            # every training + validation row, closed days included
    history[ORDER] = pd.to_datetime(history[ORDER])
    added = {}
    for value in ("Open", "SchoolHoliday", "Promo"):
        tr, te, added[value] = windows(value, history, audit)
        train = train.merge(tr, on=[GROUP, ORDER], how="left")
        test = test.merge(te, on=[GROUP, ORDER], how="left")
    arms = {
        "base": [],
        "open": added["Open"],
        "open_calendar": added["Open"] + added["SchoolHoliday"] + added["Promo"],
    }
    assert len(test) == len(audit)

    preds, fits = {}, {}
    for arm, extra in arms.items():
        cols = base_cols + extra
        preds[arm], rounds, secs = capacity_arm(
            _lgbm(False), train[cols], train[TARGET], test[cols], False, "log1p", train[ORDER])
        fits[arm] = {"n_features": len(cols), "rounds": rounds, "seconds": secs}
        print(arm, round(score(METRIC, y_audit, preds[arm]), 6), fits[arm], flush=True)

    shipped = run.bundle.predict(audit.drop(columns=[TARGET]))
    out = {
        "tag": tag,
        "run_dir": result["run_dir"],
        "windows": WINDOWS,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(audit)),
        "test_scores": {"shipped_blend": round(score(METRIC, y_audit, shipped), 6),
                        **{k: round(score(METRIC, y_audit, p), 6) for k, p in preds.items()}},
        "vs_base": {k: paired(METRIC, y_audit, preds["base"], p, audit[GROUP])
                    for k, p in preds.items() if k != "base"},
        "fits": fits,
    }
    (HERE / f"open_windows_rossmann_{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
