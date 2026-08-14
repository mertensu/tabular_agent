"""Does frozen history earn its place on Rossmann, and what did the freeze cost?

Not an agent run. This drives `fit_history` by hand against the same split v7
used, so the number moves for one reason — the feature — instead of moving for
whatever an LLM happened to try. Datasets are probes, not targets (see
`CLAUDE.md`): the question is whether the capability built for 0047/0048 is worth
having, not whether Rossmann's score goes up.

Two readings, and the second is the one worth the script:

**1. Through the harness.** Baseline work table vs the same table plus
`Store`-keyed history of `Sales`, ranked by the probe. This is the honest,
servable feature: history frozen per fold, age growing across each block.

**2. Outside the harness, on purpose.** The naive backward lag — every row reads
the true previous day's sales, including validation rows, whose labels are
sitting in the validation file. 0048 predicts this scores far better and cannot
be served.

That second table **cannot be built through the harness at all**: a `shift(1)` in
`transform_features` is refused by 0011's stability check, and there is no
lineage step that would reproduce it for a served row. So it is constructed here,
with the probe's own estimator and the run's own validation rows, and it is
labelled as what it is — a measurement of the gap, not a candidate.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import (  # noqa: E402
    _feature_frame,
    fit_history,
    probe_table,
    transform_features,
)
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.probe import fit_probe, probe_predict  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RAW = os.path.join(REPO, "rossmann_store_sales")
OUT = os.path.join(RAW, "train_joined.parquet")
TEST_OUT = os.path.join(RAW, "test_joined.parquet")
RUN_DIR_BASE = os.path.join(REPO, "_runs", "rossmann_history_probe")

METRIC = "rmspe"
SPLIT_ON = "Date"
#: v7's values, so the validation rows are the same period v7 scored on.
#:
#: Overridable from the command line, and the reason is the finding this script
#: produced rather than a convenience: at 0.05 the work half ends 2015-05-01, a
#: public holiday on which 98% of stores were shut, and **the single freeze date
#: decides what every validation row's feature is**. Pass 0.06 for a boundary on
#: an ordinary trading day and the same comparison comes out twelve times
#: smaller. Both readings are in 0049; neither alone is the result.
VAL_FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
HOLDOUT_FRACTION = 0.05
RUN_DIR = f"{RUN_DIR_BASE}_{VAL_FRACTION}"

#: Kaggle's test block starts 2015-08-01 and training ends 2015-07-31, so the
#: *earliest* row asked for is one day past the end of history. That is the floor
#: across the block, which is what `forecast_horizon` means — the rows deeper into
#: the block are 2 to 48 days out, and their `_age` columns say so.
FORECAST_HORIZON = 1.0


def serving_columns() -> list[str]:
    return [c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"]


def make_session(run_dir: str) -> RunSession:
    return RunSession(
        run_dir,
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
        serving_columns=serving_columns(),
    )


def gain(before: float, after: float) -> float:
    """Positive means better. `rmspe` is an error, so improvement is a decrease."""
    return before - after


def naive_lag_reading(session: RunSession) -> dict:
    """The unservable number: every row reads the true previous day, val included.

    Built outside the harness because the harness refuses to build it. Uses the
    probe's estimator, the probe's target transform and the run's own validation
    rows, so the only difference from reading 1 is the feature.
    """
    spec = session.resolved()
    work = load_table(session.data_path)
    val = load_table(session.val_path)

    # The whole file in time order, so validation rows can read the day before
    # them — which is what makes this a lie: at serving time that label does not
    # exist yet.
    both = pd.concat([work, val], ignore_index=True)
    both = both.sort_values(["Store", "Date"], kind="stable")
    both["Store_Sales_naive_lag1"] = both.groupby("Store")["Sales"].shift(1)
    both = both.sort_index()

    n_work = len(work)
    work_l = both.iloc[:n_work]
    val_l = both.iloc[n_work:]

    X_work, _ = _feature_frame(work_l, spec.target)
    fit = fit_probe(X_work, work_l[spec.target], spec.task_type, spec.metric,
                    spec.target_transform)
    X_val = val_l[fit.feature_names]
    preds = probe_predict(fit, X_val, spec.task_type)
    return {
        "probe_score": round(
            compute_metric(spec.metric, val_l[spec.target].to_numpy(), preds), 6
        ),
        "n_features": len(fit.feature_names),
    }


#: The work table holds `Customers`, which is not in `test.csv`, so 0030 refuses
#: to score it. Dropping it is the agent's first move on this dataset and it has
#: to happen here too, or there is no servable baseline to compare against.
DROP_CUSTOMERS = (
    "def transform_features(df):\n"
    "    return df.drop(columns=['Customers'], errors='ignore')\n"
)


def diagnose_freeze(session: RunSession) -> None:
    """What the single freeze date is, and how representative that day was.

    Printed before any score, because it turned out to explain most of the
    result. Every validation row reads history frozen at the *last day of the
    work half* — one date. If that date is a closed day, the feature is a
    constant zero for almost every served row, whatever the training columns look
    like.
    """
    work = load_table(session.data_path)
    end = work["Date"].max()
    that_day = work[work["Date"] == end]
    closed = float((that_day["Sales"] == 0).mean())

    by_date = work.groupby("Date")["Sales"].apply(lambda s: float((s == 0).mean()))
    lottery = float((by_date > 0.9).mean())

    print(f"  freeze date          : {pd.Timestamp(end).date()} "
          f"(day-of-week {that_day['DayOfWeek'].iloc[0]}, "
          f"StateHoliday {that_day['StateHoliday'].unique().tolist()})")
    print(f"  stores shut that day : {closed:.1%}")
    print(f"  freeze dates that bad: {lottery:.1%} of the {len(by_date)} candidates "
          "— this is a lottery, not this split's bad luck")


def main() -> None:
    if not os.path.exists(OUT):
        sys.exit(f"missing {OUT} — run scripts/prep_rossmann.py first")
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)

    session = make_session(RUN_DIR)
    print(f"work {len(load_table(session.data_path)):>8,} rows | "
          f"val {len(load_table(session.val_path)):>7,} rows | metric {METRIC}")

    diagnose_freeze(session)

    print("\n=== 1. through the harness ===")
    base_out = transform_features(session, DROP_CUSTOMERS)
    base_table = base_out["table"]
    base_score = base_out["probe_score"]
    print(f"  baseline (Customers dropped)     : {base_score:.5f}")

    results = {"baseline": base_score}

    variants = [
        ("Store history of Sales", dict(by=["Store"], value="Sales")),
        ("Store+DayOfWeek history of Sales", dict(by=["Store", "DayOfWeek"], value="Sales")),
        ("Store history of Sales, +lag7", dict(by=["Store"], value="Sales", offsets=[7], keep=10)),
    ]
    for label, kwargs in variants:
        out = fit_history(session, table=base_table, **kwargs)
        s = out.get("probe_score")
        if s is None:
            print(f"  {label:<33}: unscored — {out.get('probe_unavailable')}")
            continue
        results[label] = s
        print(f"  {label:<33}: {s:.5f}   "
              f"({gain(base_score, s):+.5f} vs baseline, "
              f"{out['n_rows_without_value']:,} rows without a value)")

    # `Customers` is measurable historically and absent at predict time, so its
    # history has to be taken from the table that still has the column and the
    # column dropped afterwards — two steps, in that order.
    hist_c = fit_history(session, by=["Store"], value="Customers")
    dropped = transform_features(session, DROP_CUSTOMERS, table=hist_c["table"])
    s = dropped.get("probe_score")
    label = "Store history of Customers"
    if s is None:
        print(f"  {label:<33}: unscored — {dropped.get('probe_unavailable')}")
    else:
        results[label] = s
        print(f"  {label:<33}: {s:.5f}   ({gain(base_score, s):+.5f} vs baseline)")

    print("\n=== 2. outside the harness — the number the harness refuses ===")
    naive = naive_lag_reading(session)
    print(f"  naive lag-1 (val rows read their own labels): {naive['probe_score']:.5f}   "
          f"({gain(base_score, naive['probe_score']):+.5f} vs baseline)")

    honest = results.get("Store history of Sales")
    if honest is not None:
        print(f"\n  the gap: {gain(naive['probe_score'], honest):+.5f} "
              "— what peeking at validation labels was worth, not what the "
              "feature costs (0048).")


if __name__ == "__main__":
    main()
