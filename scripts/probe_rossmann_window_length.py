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

## Registered prediction (first run, `n_splits=5`)

Monotone improvement with length, no interior optimum, converging on but not
beating the all-time number — i.e. explanation (1). If a middle length beats both
ends, that is explanation (2) and the slice has a real argument after all.

**Result: wrong.** The curve has an interior optimum at 364, and 546/728 hold the
same number of observations while scoring 0.093 apart. 0052 proposed a third
mechanism neither option named — a window longer than the shortest fold's training
prefix is unfillable in training and full at serving — **and the second run below
refuted that too.** The explanation is still open; see 0053.

---

## Second run: is that ceiling ours? (`n_splits` as argv[2])

Ulf's question, and it is the right one. Forward chaining gives fold 0 a prefix of
roughly `span / (n_splits + 1)`, so 136 days here is an artifact of `n_splits=5`,
not a property of the data. If the mechanism in 0052 is real, **the ceiling should
move when that knob moves.**

**The confound, named before running:** fewer splits lengthens the prefix *and*
widens the validation blocks by the same factor, since both are `span/(n+1)`. A
longer prefix helps (more depth to fill a window); a wider block hurts (rows sit
further past their freeze, so a window loses more of its recent end). The two pull
opposite ways and scores alone cannot separate them.

**So the score is the weaker reading here and the coverage ratio is the stronger
one.** `train_obs / served_obs` for a given window is a direct measurement of the
mechanism, independent of how the model happens to score.

### Registered predictions

1. **Mechanical, and the one that matters.** At `n_splits=3` the train/serve
   coverage ratio for `w=728` improves markedly on `n_splits=5`'s **1.71x** —
   toward but not reaching 1.0, since fold 0 is always the shallowest. If this
   does not move, 0052's explanation is wrong.
2. **`w=364`'s ratio (1.26x at n_splits=5) improves too, by less**, because it was
   closer to the prefix to begin with.
3. **`w=91` is unaffected**, its failure being a coverage floor (3 observations)
   that has nothing to do with prefixes.
4. **The score curve's optimum shifts upward**, but weakly and possibly not at
   all, because of the block-width confound above. **A null result here does not
   refute 1.**

### Results, and prediction 1 failed

| feature | n=5 | n=3 | swing |
|---|---|---|---|
| all-time mean (0046) | +0.06048 | +0.06004 | 0.0004 |
| window 364 | +0.04458 | +0.05338 | 0.0088 |
| window 546 | -0.00030 | +0.04078 | 0.0411 |
| window 728 | **-0.09287** | **+0.04654** | **0.1394** |

`w=728`'s ratio is **identical** in both (59 against 101, 1.71x) while the score
swings 0.139, and across pairs the ratio is *anti*-correlated with the score. So
0052's mechanism is refuted and 0053 records it. What survives is that a window's
worth is an artifact of the fold count while the unbounded aggregate is not, and
that the all-time mean beats every window under both settings.
"""
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from tabular_agent.encoders import HistoryEncoder  # noqa: E402
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
VAL_FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
N_SPLITS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
RUN_DIR = os.path.join(
    REPO, "_runs", f"rossmann_window_length_{VAL_FRACTION}_{N_SPLITS}"
)

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
        TaskSpec(target="Sales", metric="rmspe", n_splits=N_SPLITS,
                 forecast_horizon=1.0),
        val_fraction=VAL_FRACTION, holdout_fraction=0.05, split_on="Date",
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"],
    )

    # The served side of the ratio, which the tool never sees: the final encoder
    # is fitted on the whole work half, so a val-half row's window fills as deeply
    # as history allows. That is the column the model actually meets at predict
    # time, and comparing it with the trained column is the measurement.
    work = load_table(session.data_path)
    val = load_table(session.val_path)

    base = transform_features(session, BASE_CODE)
    b = base["probe_score"]
    print(f"  n_splits={N_SPLITS}, baseline + Store_DOW key: {b:.5f}\n")
    print(f"  {'window':<10}{'probe':>10}{'vs base':>10}"
          f"{'train obs':>11}{'served obs':>12}{'ratio':>8}")

    for w in LENGTHS:
        out = fit_history(session, by=["Store", "DayOfWeek"], value="Sales",
                          windows=[[w, "mean"]], keep=800, table=base["table"])
        n_train = out["window_observations"].get(f"{w:g}")
        enc = HistoryEncoder(["Store", "DayOfWeek"], "Sales", "Date",
                             windows=[(w, "mean")], keep=800).fit(work)
        served = enc.transform(val)[f"Store_DayOfWeek_Sales_{w:g}_count"].dropna()
        n_serve = float(served.median())
        print(f"  {w:<10}{out['probe_score']:>10.5f}{b - out['probe_score']:>+10.5f}"
              f"{n_train:>11g}{n_serve:>12g}"
              + (f"{n_serve / n_train:>8.2f}x" if n_train else f"{'--':>9}"))

    print(f"\n  shortest training prefix: {out['shortest_training_prefix']:g} days")

    te = fit_encoding(session, columns=["Store_DOW"], method="target",
                      table=base["table"])
    print(f"  {'all-time mean (unbounded, 0046)':<34}: {te['probe_score']:.5f}  "
          f"({b - te['probe_score']:+.5f})   [all obs]")


if __name__ == "__main__":
    main()
