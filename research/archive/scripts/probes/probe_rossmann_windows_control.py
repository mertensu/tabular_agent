"""The control the first window probe was missing: trailing versus all-time.

`probe_rossmann_windows.py` found that a 364-day window keyed on
`Store`+`DayOfWeek` gains **+0.045** over the baseline, the largest single-feature
gain measured in this project. That number on its own does not say what produced
it, and there are two candidates:

1. **The averaging is trailing** — "this store's Wednesdays over the last year"
   describes a store whose trade level moved recently, where an all-time mean
   does not. This is slice 3's whole thesis.
2. **The key is periodic** — "this store on a Wednesday" is simply a good group,
   and *any* summary of it would score. **This half is already built and already
   paid for**: `fit_encoding(method="target")` on a composite `Store_DOW` key was
   v7's t8 at +0.0243, the largest gain of that run (0046).

If (2) accounts for the whole effect, slice 3 bought nothing that 0046 had not
already bought, and `docs/slice3_windows.md` is explicit that this is a result to
publish rather than argue away.

So this script puts the two on the same table, the same split and the same probe,
differing only in whether the aggregate is bounded in time.

## Registered predictions

1. **The all-time target encoding of `Store_DOW` scores well** — it did on
   house-prices' cousin and on v7. Expect a large positive gain, in the same
   region as the window's +0.045.
2. **The trailing window beats it, but by much less than +0.045.** The increment
   over the all-time version is what slice 3 actually bought; the rest belongs to
   0046. Predicted: positive but under a third of the total.
3. **Both together beat either alone**, because a level and a trailing level are
   a trend, and a GBDT can difference them.
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
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RAW = os.path.join(REPO, "rossmann_store_sales")
OUT = os.path.join(RAW, "train_joined.parquet")
TEST_OUT = os.path.join(RAW, "test_joined.parquet")

METRIC = "rmspe"
VAL_FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
RUN_DIR = os.path.join(REPO, "_runs", f"rossmann_windows_control_{VAL_FRACTION}")

#: Drop `Customers` (absent from test.csv, so 0030 refuses to score a table
#: holding it) and build the composite key both halves of the comparison use.
BASE_CODE = (
    "def transform_features(df):\n"
    "    df = df.drop(columns=['Customers'], errors='ignore')\n"
    "    df['Store_DOW'] = df['Store'].astype(str) + '_' + df['DayOfWeek'].astype(str)\n"
    "    return df\n"
)


def main() -> None:
    if not os.path.exists(OUT):
        sys.exit(f"missing {OUT} — run scripts/prep_rossmann.py first")
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)

    session = RunSession(
        RUN_DIR, OUT,
        TaskSpec(target="Sales", metric=METRIC, n_splits=5, forecast_horizon=1.0),
        val_fraction=VAL_FRACTION, holdout_fraction=0.05, split_on="Date",
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"],
    )
    print(f"work {len(load_table(session.data_path)):>8,} rows | metric {METRIC} "
          f"| val_fraction {VAL_FRACTION}\n")

    base = transform_features(session, BASE_CODE)
    b = base["probe_score"]
    print(f"  {'baseline + Store_DOW key':<44}: {b:.5f}")

    # All-time: every row of the group, unbounded in time (0046).
    te = fit_encoding(session, columns=["Store_DOW"], method="target",
                      table=base["table"])
    print(f"  {'all-time target mean of Store_DOW (0046)':<44}: "
          f"{te['probe_score']:.5f}  ({b - te['probe_score']:+.5f})")

    # Trailing: the same group, bounded to the last 364 periods (slice 3).
    win = fit_history(session, by=["Store", "DayOfWeek"], value="Sales",
                      windows=[[364, "mean"]], keep=400, table=base["table"])
    print(f"  {'trailing 364 mean of Store+DOW (slice 3)':<44}: "
          f"{win['probe_score']:.5f}  ({b - win['probe_score']:+.5f})"
          f"   [{win['window_observations']}]")

    # Both, to see whether the difference between them is itself the signal.
    both = fit_history(session, by=["Store", "DayOfWeek"], value="Sales",
                       windows=[[364, "mean"]], keep=400, table=te["table"])
    print(f"  {'both':<44}: {both['probe_score']:.5f}  "
          f"({b - both['probe_score']:+.5f})")

    print(f"\n  what slice 3 bought over 0046: "
          f"{te['probe_score'] - win['probe_score']:+.5f}")


if __name__ == "__main__":
    main()
