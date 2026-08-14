"""Does averaging add anything once recency and periodicity are already there?

Slice 3's one question, and `docs/slice3_windows.md` is explicit that the answer
may be "no": *"If a window adds nothing over [364, 728], that is a publishable
result and the slice should stop, not be argued into existence."* This script is
built to be able to say that.

Not an agent run. `fit_history` is driven by hand against v7's split, so the
number moves for one reason. Datasets are probes, not targets (`CLAUDE.md`): the
question is whether the capability is worth having, not whether Rossmann's score
goes up.

Run it as `probe_rossmann_windows.py [val_fraction]`. The default 0.05 is v7's
split, whose work half ends on a public holiday with 98% of stores shut — the
boundary that destroyed `_latest` in 0049/0050. That is deliberate: a window is
row-anchored and is *supposed* to survive it, and this is the split that tests it.
Pass 0.06 for the ordinary-Wednesday control.

---

## Registered predictions

Written before the first run and not revised afterwards. 0050 records that v8's
prediction 3 was wrong precisely because it treated `_latest` and offsets as one
capability, so these are split by **mechanism**.

1. **Mechanical.** `fit_history` accepts `windows`, emits `<base>_<w>_<stat>` and
   one `<base>_<w>_count` per length, and the counts are large — several hundred
   for a `Store` key over 364 days, ~50 for `Store`+`DayOfWeek`. A count near 1
   would mean the window degenerated to `_latest` and the whole reading is void.

2. **Windows beat the baseline, unlike `_latest`.** `_latest` cost 0.132 on this
   split (0049) because it is anchored to the freeze; a window is anchored to the
   row. Predicted: positive gain, and **no repeat of the −0.13 catastrophe even on
   the holiday boundary**. This is the prediction that would falsify the design
   argument if it failed.

3. **A `Store` window does NOT beat `[364, 728]` offsets, and may lose to them.**
   The offsets work because 364 days is 52 weeks and keeps the weekday aligned
   (0050). A 364-day mean over a `Store` key averages every weekday together and
   throws that alignment away. Averaging buys variance reduction and costs the
   periodicity, and on this dataset the periodicity looks like the larger term.

4. **A `Store`+`DayOfWeek` window is the better window**, because it averages
   *within* the weekday instead of across it — that is the competition winner's
   feature, and the one shape where averaging and periodicity are not traded off
   against each other. Predicted: the best window of the three, and the only one
   with a real chance of beating the offsets.

5. **Offsets + window together beat either alone, but the increment over offsets
   alone is small** — under half of what the offsets bought over the baseline.
   This is the prediction that decides whether the slice ships: a large increment
   means averaging is a capability, a negligible one means offsets already had it.

6. **`median` beats `mean` on this dataset**, because the failure 0050 documented
   is a spike (the last open day before a holiday, +53%), and this dataset is full
   of them. If it does not, the median's per-row cost is not worth carrying.
"""
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import fit_history, transform_features  # noqa: E402
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RAW = os.path.join(REPO, "rossmann_store_sales")
OUT = os.path.join(RAW, "train_joined.parquet")
TEST_OUT = os.path.join(RAW, "test_joined.parquet")

METRIC = "rmspe"
SPLIT_ON = "Date"
VAL_FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
HOLDOUT_FRACTION = 0.05
RUN_DIR = os.path.join(REPO, "_runs", f"rossmann_windows_probe_{VAL_FRACTION}")

#: Kaggle's test block starts one day past the end of training, so the floor
#: across the block is 1. The rows deeper in are 2-48 days out (0047).
FORECAST_HORIZON = 1.0

#: `Customers` is not in `test.csv`, so 0030 refuses to score a table holding it.
#: Dropping it is the agent's first move on this dataset and has to happen here
#: too, or there is no servable baseline to compare against.
DROP_CUSTOMERS = (
    "def transform_features(df):\n"
    "    return df.drop(columns=['Customers'], errors='ignore')\n"
)


def make_session() -> RunSession:
    return RunSession(
        RUN_DIR,
        OUT,
        TaskSpec(
            target="Sales",
            metric=METRIC,
            n_splits=5,
            forecast_horizon=FORECAST_HORIZON,
        ),
        val_fraction=VAL_FRACTION,
        holdout_fraction=HOLDOUT_FRACTION,
        split_on=SPLIT_ON,
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"],
    )


def gain(before: float, after: float) -> float:
    """Positive means better. `rmspe` is an error, so improvement is a decrease."""
    return before - after


#: The comparison the slice exists to make. Every variant is one `fit_history`
#: call on the same baseline table, so the only thing that varies is the
#: parameter — the shape 0050 established as the way to read this tool.
VARIANTS = [
    ("offsets [364,728]  (v8's shape)",
     dict(by=["Store"], value="Sales", offsets=[364, 728], keep=740)),
    ("window 364 mean, Store",
     dict(by=["Store"], value="Sales", windows=[[364, "mean"]], keep=400)),
    ("window 364 median, Store",
     dict(by=["Store"], value="Sales", windows=[[364, "median"]], keep=400)),
    ("window 364 mean, Store+DOW",
     dict(by=["Store", "DayOfWeek"], value="Sales", windows=[[364, "mean"]], keep=400)),
    ("window 364 median, Store+DOW",
     dict(by=["Store", "DayOfWeek"], value="Sales", windows=[[364, "median"]], keep=400)),
    ("window 91 mean, Store+DOW",
     dict(by=["Store", "DayOfWeek"], value="Sales", windows=[[91, "mean"]], keep=400)),
    ("offsets [364,728] + window 364 mean, Store",
     dict(by=["Store"], value="Sales", offsets=[364, 728],
          windows=[[364, "mean"]], keep=740)),
]


def main() -> None:
    if not os.path.exists(OUT):
        sys.exit(f"missing {OUT} — run scripts/prep_rossmann.py first")
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)

    session = make_session()
    work = load_table(session.data_path)
    print(f"work {len(work):>8,} rows | val {len(load_table(session.val_path)):>7,} "
          f"rows | metric {METRIC} | val_fraction {VAL_FRACTION}")
    end = work["Date"].max()
    that_day = work[work["Date"] == end]
    print(f"freeze date {pd.Timestamp(end).date()}, "
          f"{float((that_day['Sales'] == 0).mean()):.1%} of stores shut "
          "— the boundary a row-anchored feature is supposed to survive\n")

    base = transform_features(session, DROP_CUSTOMERS)
    print(f"  {'baseline (Customers dropped)':<40}: {base['probe_score']:.5f}")
    b = base["probe_score"]

    for label, kwargs in VARIANTS:
        out = fit_history(session, table=base["table"], **kwargs)
        s = out.get("probe_score")
        if s is None:
            print(f"  {label:<40}: unscored — {out.get('probe_unavailable')}")
            continue
        counts = out.get("window_observations") or {}
        held = ", ".join(f"{k}:{v:g} obs" for k, v in counts.items())
        print(f"  {label:<40}: {s:.5f}  ({gain(b, s):+.5f})"
              + (f"   [{held}]" if held else ""))
        if "WARNING" in (out.get("note") or ""):
            print(f"      WARNING fired — {out['note'].split('WARNING:')[1].strip()[:110]}")

    print(f"\n  block_span was {out.get('block_span')} days — every window above has "
          "to clear that before it holds anything.")


if __name__ == "__main__":
    main()
