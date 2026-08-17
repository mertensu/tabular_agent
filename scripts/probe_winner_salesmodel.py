"""Rebuild the Rossmann winner's main model by hand, and find out what we are missing.

Ulf, setting the question this answers: *"ich will nur vermeiden eine echte Lücke zu
haben in unserer harness/agent. wenn der agent alles könnte aber es nicht schafft
ist es eine andere sache."*

Two very different problems wear the same symptom of being mid-table:

* **a capability gap** — the harness cannot express a feature at all, so no run
  could ever have built it;
* **an agent gap** — the harness can express it and the agent did not build it.

The first is fixed with library code, the second with something else entirely
(prompt, search budget, stopping rule), and this project has been guessing which
one it has. 0069 and 0070 are both records of a lever whose size was estimated from
a description and then measured much smaller. So this script estimates nothing.

## What is rebuilt

Only the **salesmodel** — the branch of his solution that carries bit 0 of his
feature dict. Parsed from `rossmann_store_sales/winner.md` at run time rather than
transcribed, because transcribing it by eye is exactly how 0070 first claimed the
trend features were his largest, when the dict says they are not in this model at
all. 24 features, and notably **no `store` column**: he excluded store identity
deliberately.

Classified against the tool vocabulary before anything was fitted:

| | n | |
|---|---|---|
| buildable with existing tools | 19 | `transform_features`, `fit_encoding`, `fit_group_stats`, `fit_history` |
| **capability gap** | 3 | `holidays_last/this/nextweek` |
| out by decision | 2 | `weather_maxtemp`, `weather_precip` (0001) |

The three in the middle are one capability, not three: **a window over an exogenous
column that is known in advance.** `fit_history` reaches backward over the target's
own past; `fit_stream_aggregate` restricts to what was knowable a lead ahead. Nothing
reaches *forward* over a calendar that is published years ahead, and nothing computes
a per-row count over neighbouring dates that survives a batch of one row.

## The three arms, and what each difference means

* **A — tools only.** Every column built by calling the real tool functions, so the
  table is one a run *could* have produced. Nothing here is hand pandas except what
  `transform_features` legitimately accepts.
* **B — A plus the three calendar windows**, written by hand because no tool can
  express them. **B − A is the capability gap, measured in rmspe.**
* **v7's `t13`** — what the agent actually built, already on disk. **A − t13 is the
  agent gap**: same vocabulary available, different choices made.

All three are fitted at a real budget and scored on v7's own sealed holdout, so they
are comparable to each other and to 0068's readings.

## The prediction, registered before running

* **A lands closer to `t13` than to his 0.105** — around 0.13 to 0.15 sealed. His
  ~0.105 private came from a model with 5,000 rounds, `colsample_bytree=0.3`, an
  ensemble behind it, and two features we exclude on purpose. A single fit on 19 of
  his columns should not reach it.
* **B − A is small, under 0.005.** Holiday-window counts are three columns out of
  twenty-two on a dataset whose dominant structure is per-store and per-weekday.
* **The interesting number is A − t13, and I expect it to be near zero or negative**
  — `t13` has eight target-encoded interaction columns that his salesmodel does not,
  and 0046 measured that family as worth +0.060 on its own.

**If all three land together around 0.13**, the answer to Ulf's question is that
there is **no large capability gap and no large agent gap in the feature vocabulary**,
and the distance to the winner is in capacity, model configuration and architecture —
which 0068 already measured as bounded at +0.0037 for capacity alone. That would be
the most useful negative result available, and it is the outcome I expect.

**If A beats `t13` by more than 0.01**, the harness could have built a much better
table than the agent did, and the problem is the search rather than the vocabulary.

## What this is not

Not a submission and not a tuning run — Rossmann's board is spent (0045). The
dataset-specific knowledge here is confined to this file, which is where `CLAUDE.md`
sanctions it; what the run produces is an answer about `src/`, which is general.
"""
import ast
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from prep_rossmann import (  # noqa: E402
    ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT, leaderboard, leaderboard_rank,
)
from tabular_agent import models  # noqa: E402
from tabular_agent.harness import journal  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.stop import StopPolicy  # noqa: E402
from tabular_agent.harness.tools import (  # noqa: E402
    filter_rows,
    finalize_run,
    fit_encoding,
    fit_group_stats,
    fit_history,
    inspect_data,
    transform_features,
)
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RUNS = os.path.join(REPO, "_runs")
OUT_ROOT = os.path.join(RUNS, "winner_salesmodel")
SOURCE_RUN = os.path.join(RUNS, "rossmann_v7")
WINNER_MD = os.path.join(REPO, "rossmann_store_sales", "winner.md")

#: The budget the arms are fitted at. `1000 @ lr 0.10` because 0068 measured it as
#: the best of the three on Rossmann's *sealed* rows (+0.00373 against 200 trees,
#: where 5000@0.03 gave only +0.00189). Not a tuned choice and not adopted
#: anywhere — reused so these numbers sit on the same scale as 0068's.
BUDGET = {"n_estimators": 1000, "learning_rate": 0.1}
BUDGETED = ("lgbm", "xgb", "catboost")

#: His salesmodel columns that we exclude by decision rather than by inability.
EXTERNAL = {"weather_maxtemp", "weather_precip", "relativeweather_maxtemp",
            "relativeweather_precip"}


def winner_features(bit: int = 0) -> list[str]:
    """His feature dict, parsed from the writeup rather than transcribed.

    0070 first claimed the trend features were the largest gap; the dict says they
    are `[0,0,0,1]` and therefore not in this model at all. Reading four bits by
    eye is how that happened, so it is done in code here.
    """
    m = re.search(r"features=\{(.*?)\n\}", open(WINNER_MD).read(), re.S)
    d = ast.literal_eval("{" + m.group(1) + "}")
    return [k for k, v in d.items() if v[bit] == 1]


# --------------------------------------------------------------------------
# The steps. Each is a real tool call, and each says which of his features it
# is supposed to produce, so the mapping is auditable rather than asserted.
# --------------------------------------------------------------------------

#: day, dayofweek, dayofyear, week, daysincompetition_unrounded, daysinpromocycle,
#: primpromocycle. Pure row-local arithmetic on columns that exist at serve time,
#: which is precisely what `transform_features` is for.
DATE_PARTS = '''
import numpy as np
def transform_features(df):
    df = df.copy()
    d = df["Date"]
    df["day"] = d.dt.day
    df["dayofyear"] = d.dt.dayofyear
    df["week"] = d.dt.isocalendar().week.astype("int64")
    df["month"] = d.dt.month
    df["year"] = d.dt.year
    # daysincompetition, unrounded: months since the competitor opened, as days.
    comp = (d.dt.year - df["CompetitionOpenSinceYear"]) * 365.25 + (
        d.dt.month - df["CompetitionOpenSinceMonth"]) * 30.4375
    df["daysincompetition_unrounded"] = comp
    # The secondary promotion runs on a quarterly interval named in PromoInterval;
    # primpromocycle is the position of this month within that cycle, and
    # daysinpromocycle the day offset since the cycle's month began.
    months = {"Jan,Apr,Jul,Oct": 1, "Feb,May,Aug,Nov": 2, "Mar,Jun,Sept,Dec": 3}
    start = df["PromoInterval"].map(months).astype("float64")
    df["primpromocycle"] = np.where(
        start.notna(), (d.dt.month.to_numpy() - start.to_numpy()) % 3, np.nan)
    df["daysinpromocycle"] = np.where(
        start.notna(), df["primpromocycle"].to_numpy() * 30.4375 + d.dt.day.to_numpy() - 1,
        np.nan)
    # Promo2 has not started yet for many store-days; a week/year pair is only
    # meaningful once it has.
    started = (d.dt.year > df["Promo2SinceYear"]) | (
        (d.dt.year == df["Promo2SinceYear"])
        & (d.dt.isocalendar().week.astype("int64") >= df["Promo2SinceWeek"]))
    df["promo2_active"] = (df["Promo2"].astype(bool) & started).astype("int64")
    return df
'''

#: The three columns no tool can produce: a count over a *window of dates* on an
#: exogenous calendar. Written by hand for arm B only, which is what makes B − A a
#: measurement of the missing capability rather than an opinion about it.
#:
#: Note what makes it inexpressible rather than merely awkward: the value for one
#: row is a function of *other rows' dates*, so a served batch of one row cannot
#: produce it — 0011's stability check rejects exactly this, and correctly. The
#: information is legitimately available (a school-holiday calendar is published
#: years ahead); the harness simply has no way to be given it.
HOLIDAY_WINDOWS = '''
import numpy as np
import pandas as pd
def transform_features(df):
    df = df.copy()
    d = df["Date"]
    # One row per date: whether any store had a holiday that day. Built from the
    # frame, which is the reason this cannot be a tool — see the comment above.
    flag = (df["SchoolHoliday"].astype(float) > 0) | (df["StateHoliday"].astype(str).isin(["a", "b", "c"]))
    by_date = flag.groupby(d).max().astype(float).sort_index()
    full = by_date.reindex(pd.date_range(by_date.index.min(), by_date.index.max(), freq="D"), fill_value=0.0)
    roll = full.rolling(7, min_periods=1).sum()
    this = roll
    last = roll.shift(7).fillna(0.0)
    nxt = roll.shift(-7).fillna(0.0)
    for name, series in (("holidays_thisweek", this), ("holidays_lastweek", last),
                         ("holidays_nextweek", nxt)):
        df[name] = d.map(series).astype("float64")
    return df
'''

#: He trained on open days with non-zero sales only. v7 did the same; recorded
#: here because it is one of his stated choices, not an incidental.
OPEN_ONLY = '''
def select_rows(df):
    return (df["Open"] == 1) & (df["Sales"] > 0)
'''


def build_session(arm: str) -> RunSession:
    run_dir = os.path.join(OUT_ROOT, arm)
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    # v7's split, copied. These arms have to be comparable to `t13` and to 0068's
    # readings, and a recut split would quietly make them a different experiment.
    shutil.copytree(os.path.join(SOURCE_RUN, "data"), os.path.join(run_dir, "data"))
    session = RunSession(
        run_dir, OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        stop_policy=StopPolicy(max_tables=40),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=[c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"],
        fit_time_budget_s=90 * 60,
    )
    inspect_data(session)
    return session


def build_table(session: RunSession, with_gap_features: bool) -> str:
    """His salesmodel, built with tool calls. Returns the final table path.

    The order matters in one place and it is worth stating: **`fit_history` runs
    before `filter_rows`.** Its windows count *rows*, and on this dataset a row is
    a day only while every store-day is present. Filtering to open days first would
    make a 91-row window mean something different per store — 0054 found exactly
    that on a ragged spine, and it would be invisible.
    """
    steps: list[tuple[str, dict]] = []

    out = transform_features(session, DATE_PARTS)
    steps.append(("transform: date parts, competition age, promo cycle", out))
    table = out["table"]

    if with_gap_features:
        out = transform_features(session, HOLIDAY_WINDOWS, table=table)
        steps.append(("transform: holiday windows (NO TOOL EXPRESSES THIS)", out))
        table = out["table"]

    # assortment, promointerval, stateholiday, storetype — categorical columns his
    # model used directly. Frequency rather than target encoding: he fed them to
    # xgboost as codes, and a target encoding would add information his model did
    # not have.
    out = fit_encoding(
        session, ["Assortment", "PromoInterval", "StateHoliday", "StoreType"],
        method="frequency", table=table,
    )
    steps.append(("fit_encoding(frequency): his 4 categorical columns", out))
    table = out["table"]

    # shopavg_schoolholiday — the proportion of days a store has a school holiday.
    out = fit_group_stats(session, by=["Store"], value="SchoolHoliday",
                          stat="mean", table=table)
    steps.append(("fit_group_stats: shopavg_schoolholiday", out))
    table = out["table"]

    # shopavg_salespercustomer — the feature 0070 notes we throw away in every
    # Rossmann run. `Customers` is not in test.csv, and `fit_group_stats` is
    # explicit that the value column need not exist at predict time; only the `by`
    # columns must.
    out = fit_group_stats(session, by=["Store"], value="Customers",
                          stat="mean", table=table)
    steps.append(("fit_group_stats: shopavg_salespercustomer (train-only column)", out))
    table = out["table"]

    # shopsales_promo — how a store sells during promotions, relative to itself.
    out = fit_encoding(session, ["Store", "Promo"], method="target", table=table)
    steps.append(("fit_encoding(target): shopsales_promo", out))
    table = out["table"]

    # prevquarter_med — the median of the store's own sales over the last quarter.
    out = fit_history(session, by=["Store"], value="Sales",
                      windows=[[91, "median"]], table=table)
    steps.append(("fit_history: prevquarter_med", out))
    table = out["table"]

    # prevyear_dps_med — median by day-of-week / promo / store over the last year.
    out = fit_history(session, by=["Store", "DayOfWeek", "Promo"], value="Sales",
                      windows=[[364, "median"]], table=table)
    steps.append(("fit_history: prevyear_dps_med", out))
    table = out["table"]

    out = filter_rows(session, OPEN_ONLY, table=table)
    steps.append(("filter_rows: open days with non-zero sales", out))
    table = out["table"]

    print(f"\n  built {len(steps)} steps:")
    for label, o in steps:
        probe = o.get("probe_score")
        gain = o.get("probe_vs_source_table")
        cells = "unscored" if probe is None else f"probe {probe:.5f}"
        if gain is not None:
            cells += f"  ({gain:+.5f} vs parent)"
        print(f"    {label:<62} {cells}")
    return table


def _swap_budget():
    import contextlib

    @contextlib.contextmanager
    def cm():
        original = models._make_model

        def patched(model_type, task_type, p, seed, metric=None):
            if model_type in BUDGETED:
                p = dict(p or {})
                p.update(BUDGET)
            return original(model_type, task_type, p, seed, metric)

        models._make_model = patched
        try:
            yield
        finally:
            models._make_model = original

    return cm()


def shipped(run_dir: str) -> dict:
    j = journal.load(run_dir)
    return {
        e.get("model_type"): {"val": e.get("val_score"), "sealed": e.get("holdout_score")}
        for e in (j.get("holdout") or {}).get("scores", [])
    }


def run_arm(arm: str, with_gap_features: bool) -> dict:
    print(f"\n{'=' * 74}\n=== {arm} ===")
    session = build_session(arm)
    build_table(session, with_gap_features)
    with _swap_budget():
        finalize_run(session)
    scores = shipped(session.run_dir)
    for mt, s in scores.items():
        sealed = "n/a" if s["sealed"] is None else f"{s['sealed']:.5f}"
        print(f"    {mt:<42} val {s['val']:.5f}  sealed {sealed}")
    blend = next((k for k in scores if k and k.startswith("blend")), None)
    return {"arm": arm, **{f"{k}_{f}": v[f] for k, v in scores.items() if k
                           for f in ("val", "sealed")},
            "blend_val": scores[blend]["val"] if blend else None,
            "blend_sealed": scores[blend]["sealed"] if blend else None}


def main() -> int:
    if not os.path.isdir(SOURCE_RUN):
        print(f"missing {SOURCE_RUN}", file=sys.stderr)
        return 1
    os.makedirs(OUT_ROOT, exist_ok=True)

    sales = winner_features(0)
    buildable = [f for f in sales if f not in EXTERNAL
                 and not f.startswith("holidays_")]
    print(f"his salesmodel: {len(sales)} features")
    print(f"  buildable with existing tools : {len(buildable)}")
    print(f"  capability gap (calendar windows): "
          f"{[f for f in sales if f.startswith('holidays_')]}")
    print(f"  out by decision (0001)          : "
          f"{sorted(set(sales) & EXTERNAL)}")
    print(f"  note: 'store' in his salesmodel : {'store' in sales}")

    rows = [run_arm("A_tools_only", False), run_arm("B_plus_calendar", True)]

    print(f"\n{'=' * 74}\nTHE READING (rmspe, lower is better; sealed = v7's holdout)")
    t13 = 0.136181  # v7's t13 blend at this budget — 0068, `_runs/capacity_terminal_blend`
    print(f"  {'v7 t13, what the agent built (200 trees)':<46} {t13:.5f}")
    print(f"  {'v7 t13, same budget as here':<46} {0.132450:.5f}   [0068]")
    for r in rows:
        print(f"  {r['arm']:<46} {r['blend_sealed']:.5f}")
    a, b = rows[0]["blend_sealed"], rows[1]["blend_sealed"]
    print(f"\n  capability gap  (B - A)          {a - b:+.5f}")
    print(f"  agent gap       (A - t13@budget) {0.132450 - a:+.5f}")
    print(f"\n  his single un-ensembled model, private: ~0.105 "
          f"({leaderboard_rank(0.105)})")
    print(f"  private board: {leaderboard(True)}")
    print("\n  Sealed scores are not Kaggle scores; 0060 says the seal is a lower")
    print("  bound at the horizon we are graded at. No submission is written.")

    pd.DataFrame(rows).to_csv(os.path.join(RUNS, "winner_salesmodel.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
