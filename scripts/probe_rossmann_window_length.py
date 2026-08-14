"""Why did the trailing window lose to the all-time mean — recency, or sample size?

The control (`probe_rossmann_windows_control.py`) found that a 364-day trailing
mean of `Store`+`DayOfWeek` scores **0.01590 worse** than 0046's all-time target
mean of the same key. Two explanations fit that, and they say opposite things
about whether the capability is worth having:

1. **Sample size.** Bounding the aggregate throws data away. The window held a
   median of 39 observations where the all-time mean has ~134, and a mean over 39
   is a noisier estimate of the same quantity. If this is the mechanism, recency
   was never the issue and the window is simply a worse estimator.

2. **Recency is genuinely worthless here.** Rossmann's per-store weekday level is
   stable over 2.5 years, so restricting to recent history describes the same
   thing with less evidence *and* no compensating gain.

They are separable: **sweep the length.** Under (1) the score improves
monotonically as the window grows toward the all-time mean, and the curve has no
interior optimum. Under (2) there is no reason for it to be monotone — a store
whose trade level moved would be described best by some middle length.

This is a measurement of how window length trades against coverage, which is a
general property of the tool. It is **not** a search for the right length for
Rossmann: a length that wins here is a fact about this dataset and goes nowhere
near `src/` (CLAUDE.md, 0022).

## Registered prediction

Monotone improvement with length, no interior optimum, converging on but not
beating the all-time number — i.e. explanation (1). If a middle length beats both
ends, that is explanation (2) and the slice has a real argument after all.
"""
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import (  # noqa: E402
    fit_encoding,
    fit_history,
    transform_features,
)
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RAW = os.path.join(REPO, "rossmann_store_sales")
OUT = os.path.join(RAW, "train_joined.parquet")
TEST_OUT = os.path.join(RAW, "test_joined.parquet")
VAL_FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
RUN_DIR = os.path.join(REPO, "_runs", f"rossmann_window_length_{VAL_FRACTION}")

BASE_CODE = (
    "def transform_features(df):\n"
    "    df = df.drop(columns=['Customers'], errors='ignore')\n"
    "    df['Store_DOW'] = df['Store'].astype(str) + '_' + df['DayOfWeek'].astype(str)\n"
    "    return df\n"
)

#: Spread across the range the tool can express, not chosen to flatter anything.
#: 91 is already known to be too short (3 observations); 728 is two years, near
#: the whole work half, so it should behave almost like the all-time mean.
LENGTHS = [91, 182, 364, 546, 728]


def main() -> None:
    if not os.path.exists(OUT):
        sys.exit(f"missing {OUT} — run scripts/prep_rossmann.py first")
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)

    session = RunSession(
        RUN_DIR, OUT,
        TaskSpec(target="Sales", metric="rmspe", n_splits=5, forecast_horizon=1.0),
        val_fraction=VAL_FRACTION, holdout_fraction=0.05, split_on="Date",
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"],
    )

    base = transform_features(session, BASE_CODE)
    b = base["probe_score"]
    print(f"  {'baseline + Store_DOW key':<34}: {b:.5f}")

    for w in LENGTHS:
        out = fit_history(session, by=["Store", "DayOfWeek"], value="Sales",
                          windows=[[w, "mean"]], keep=800, table=base["table"])
        n = out["window_observations"].get(f"{w:g}")
        print(f"  {'trailing ' + str(w) + ' mean':<34}: {out['probe_score']:.5f}  "
              f"({b - out['probe_score']:+.5f})   [{n:g} obs]")

    te = fit_encoding(session, columns=["Store_DOW"], method="target",
                      table=base["table"])
    print(f"  {'all-time mean (unbounded, 0046)':<34}: {te['probe_score']:.5f}  "
          f"({b - te['probe_score']:+.5f})   [all obs]")


if __name__ == "__main__":
    main()
