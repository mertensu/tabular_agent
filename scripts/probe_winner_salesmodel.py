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
from tabular_agent.io import load_table  # noqa: E402
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

#: The three columns no tool can produce, written the way anyone would write them:
#: roll a per-date holiday flag over a seven-day window.
#:
#: **This is expected to be REFUSED**, and the refusal is half the finding. The
#: value for one row is computed from *other rows' dates*, so the same row in a
#: different batch gets a different number — which is precisely what 0011's
#: stability check exists to catch, and it is right to catch it. A served batch of
#: one row cannot produce this column at all.
#:
#: So the natural implementation is not merely absent from the tool vocabulary; it
#: is something the harness actively and correctly rejects. Arm B therefore cannot
#: use it, and `FITTED_CALENDAR` below is what the missing tool would do instead.
HOLIDAY_WINDOWS_NAIVE = '''
import numpy as np
import pandas as pd
def transform_features(df):
    df = df.copy()
    d = df["Date"]
    flag = (df["SchoolHoliday"].astype(float) > 0) | (df["StateHoliday"].astype(str).isin(["a", "b", "c"]))
    by_date = flag.groupby(d).max().astype(float).sort_index()
    full = by_date.reindex(pd.date_range(by_date.index.min(), by_date.index.max(), freq="D"), fill_value=0.0)
    roll = full.rolling(7, min_periods=1).sum()
    for name, series in (("holidays_thisweek", roll),
                         ("holidays_lastweek", roll.shift(7).fillna(0.0)),
                         ("holidays_nextweek", roll.shift(-7).fillna(0.0))):
        df[name] = d.map(series).astype("float64")
    return df
'''


def fitted_calendar_code() -> str:
    """The same three columns as a **fitted lookup**, which is what a tool would do.

    The map is computed once here, from the holiday flags of the training *and*
    test frames, and embedded in the generated code as a literal. That makes each
    column a pure function of the row's `Date`: identical at batch size one and at
    a million, so the stability check passes and a bundle can serve it.

    Using test dates is not leakage and the distinction matters. A school-holiday
    calendar is published years in advance, and `SchoolHoliday` / `StateHoliday`
    are columns of `test.csv` — the information is genuinely available at serving
    time. What the harness lacks is any way to be *given* such a calendar, which is
    why this has to be smuggled in as a constant to be measured at all.

    That is also the honest limit of arm B: it measures what the capability would
    be worth, not a capability we have. Nothing here is shippable.
    """
    frames = []
    for path in (OUT, TEST_OUT):
        df = pd.read_parquet(path, columns=["Date", "SchoolHoliday", "StateHoliday"])
        frames.append(df)
    allrows = pd.concat(frames, ignore_index=True)
    flag = (allrows["SchoolHoliday"].astype(float) > 0) | (
        allrows["StateHoliday"].astype(str).isin(["a", "b", "c"]))
    by_date = flag.groupby(allrows["Date"]).max().astype(float).sort_index()
    full = by_date.reindex(
        pd.date_range(by_date.index.min(), by_date.index.max(), freq="D"), fill_value=0.0)
    roll = full.rolling(7, min_periods=1).sum()
    table = {
        d.strftime("%Y-%m-%d"): [
            float(roll.iloc[i]),
            float(roll.shift(7).fillna(0.0).iloc[i]),
            float(roll.shift(-7).fillna(0.0).iloc[i]),
        ]
        for i, d in enumerate(full.index)
    }
    return (
        "CAL = " + json.dumps(table) + "\n"
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    key = df['Date'].dt.strftime('%Y-%m-%d')\n"
        "    for i, name in enumerate(('holidays_thisweek', 'holidays_lastweek',\n"
        "                              'holidays_nextweek')):\n"
        "        df[name] = key.map(lambda k: CAL.get(k, [None, None, None])[i]).astype('float64')\n"
        "    return df\n"
    )

#: He trained on open days with non-zero sales only. v7 did the same; recorded
#: here because it is one of his stated choices, not an incidental.
OPEN_ONLY = '''
def keep_rows(df):
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

    def note(label: str, out: dict) -> str:
        # Printed as each step lands rather than collected: these are minutes
        # apart on 900k rows, and a crash at step eight should not take the
        # readings of the first seven with it.
        steps.append((label, out))
        probe, gain = out.get("probe_score"), out.get("probe_vs_source_table")
        cells = "unscored" if probe is None else f"probe {probe:.5f}"
        if gain is not None:
            cells += f"  ({gain:+.5f} vs parent)"
        print(f"    {label:<62} {cells}")
        return out["table"]

    out = transform_features(session, DATE_PARTS)
    table = note("transform: date parts, competition age, promo cycle", out)

    if with_gap_features:
        # First the way anyone would write it, to record what the harness does
        # with it. A refusal here is evidence, not an obstacle.
        try:
            transform_features(session, HOLIDAY_WINDOWS_NAIVE, table=table)
            print("\n  !! the naive rolling-window version was ACCEPTED — the "
                  "stability check did not catch a batch-dependent column, which "
                  "is worth investigating on its own")
        except Exception as e:  # noqa: BLE001 - the refusal is the measurement
            print(f"\n  the naive rolling-window version was REFUSED, as designed:"
                  f"\n    {type(e).__name__}: {str(e).splitlines()[0][:160]}")

        out = transform_features(session, fitted_calendar_code(), table=table)
        table = note("transform: holiday windows as a FITTED LOOKUP (no tool does this)", out)

    # assortment, promointerval, stateholiday, storetype — categorical columns his
    # model used directly. Frequency rather than target encoding: he fed them to
    # xgboost as codes, and a target encoding would add information his model did
    # not have.
    out = fit_encoding(
        session, ["Assortment", "PromoInterval", "StateHoliday", "StoreType"],
        method="frequency", table=table,
    )
    table = note("fit_encoding(frequency): his 4 categorical columns", out)

    # shopavg_schoolholiday — the proportion of days a store has a school holiday.
    out = fit_group_stats(session, by=["Store"], value="SchoolHoliday",
                          stat="mean", table=table)
    table = note("fit_group_stats: shopavg_schoolholiday", out)

    # shopavg_salespercustomer — the feature 0070 notes we throw away in every
    # Rossmann run. `Customers` is not in test.csv, and `fit_group_stats` is
    # explicit that the value column need not exist at predict time; only the `by`
    # columns must.
    out = fit_group_stats(session, by=["Store"], value="Customers",
                          stat="mean", table=table)
    table = note("fit_group_stats: shopavg_salespercustomer (train-only column)", out)

    # And now `Customers` itself has to go, which is the whole shape of the thing
    # 0070 flagged. The raw column is absent from `test.csv`, so 0030 refuses to
    # score any table still carrying it — every step above this line comes back
    # `unscored` for that reason alone. The *aggregate* survives the drop, because
    # `fit_group_stats` stored it as a lookup keyed on `Store`.
    #
    # So the reason every Rossmann run throws `Customers` away is that dropping it
    # is the agent's first move, and doing it first also throws away the only thing
    # it was good for. Order is the entire difference.
    out = transform_features(
        session,
        "def transform_features(df):\n"
        "    return df.drop(columns=['Customers'], errors='ignore')\n",
        table=table,
    )
    table = note("transform: drop Customers (its aggregate survives)", out)

    # shopsales_promo — how a store sells during promotions, relative to itself.
    #
    # The key has to be *built* first. The first version of this called
    # `fit_encoding(["Store", "Promo"], method="target")` and got `Store_te` and
    # `Promo_te` — two separate encodings, which is what that argument means and
    # not what his feature is. His is the interaction: this store, under promo.
    # v7's agent got this right and built `Store_Promo` as a transform before
    # encoding it; the hand rebuild got it wrong, which is worth recording because
    # the mistake made the rebuild weaker and so flattered the agent.
    out = transform_features(
        session,
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['Store_Promo'] = df['Store'].astype(str) + '_' + df['Promo'].astype(str)\n"
        "    return df\n",
        table=table,
    )
    table = note("transform: Store x Promo key", out)
    out = fit_encoding(session, ["Store_Promo"], method="target", table=table)
    table = note("fit_encoding(target): shopsales_promo", out)

    # prevquarter_med — the median of the store's own sales over the last quarter.
    out = fit_history(session, by=["Store"], value="Sales",
                      windows=[[91, "median"]], table=table)
    table = note("fit_history: prevquarter_med", out)

    # prevyear_dps_med — median by day-of-week / promo / store over the last year.
    out = fit_history(session, by=["Store", "DayOfWeek", "Promo"], value="Sales",
                      windows=[[364, "median"]], table=table)
    table = note("fit_history: prevyear_dps_med", out)

    out = filter_rows(session, OPEN_ONLY, table=table)
    table = note("filter_rows: open days with non-zero sales", out)

    print(f"  built {len(steps)} steps")
    return table


def ablate_latest_pair(session: RunSession, table: str) -> None:
    """What the `_latest` pair costs, measured instead of asserted.

    `fit_history` emits four columns where the winner's feature is one: the window
    statistic and its count, **plus an unrequested `_latest` / `_latest_ago` pair**
    — the last known value and its age. 0049 and 0050 are both about that pair
    being a sample of size one and the anchor mattering more than the history, and
    `HANDOFF.md` has carried "`_latest` still emitted unconditionally" as an open
    item for weeks.

    In this rebuild the first history step scored **0.030 worse** than its parent,
    which makes the pair the obvious suspect. Obvious is not measured, and two
    obvious diagnoses have already been wrong today, so: take the finished table,
    drop only those columns, and probe it. Nothing else changes.

    A real gain here is not a Rossmann fact. It says the harness cannot express
    "summarise this group's recent past" without also adding a column pair that a
    run may not want and cannot decline — which names no dataset.
    """
    df = load_table(session.resolve_table(table))
    pair = [c for c in df.columns if c.endswith("_latest") or c.endswith("_latest_ago")]
    if not pair:
        print("\n  no `_latest` columns to ablate")
        return
    out = transform_features(
        session,
        "def transform_features(df):\n"
        f"    return df.drop(columns={pair!r}, errors='ignore')\n",
        table=table,
    )
    before = journal.load(session.run_dir)["tables"]
    parent = next((t["probe_score"] for t in before if t["path"] == table), None)
    now = out.get("probe_score")
    print(f"\n  ABLATION — what `fit_history`'s unrequested `_latest` pair costs")
    print(f"    dropped: {pair}")
    print(f"    with the pair    {parent:.5f}")
    print(f"    without it       {now:.5f}")
    if parent is not None and now is not None:
        print(f"    the pair is worth {parent - now:+.5f} "
              f"({'it costs us' if now < parent else 'it helps'})")


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


def run_arm(arm: str, with_gap_features: bool, ablate: bool = False) -> dict:
    print(f"\n{'=' * 74}\n=== {arm} ===")
    session = build_session(arm)
    table = build_table(session, with_gap_features)
    if ablate:
        ablate_latest_pair(session, table)
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

    rows = [run_arm("A_tools_only", False, ablate=True),
            run_arm("B_plus_calendar", True)]

    print(f"\n{'=' * 74}\nTHE READING (rmspe, lower is better; sealed = v7's holdout)")
    t13 = 0.136181  # v7's t13 blend at this budget — 0068, `_runs/capacity_terminal_blend`
    print(f"  {'v7 t13, what the agent built (200 trees)':<46} {t13:.5f}")
    print(f"  {'v7 t13, same budget as here':<46} {0.132446:.5f}   [0068]")
    for r in rows:
        print(f"  {r['arm']:<46} {r['blend_sealed']:.5f}")
    a, b = rows[0]["blend_sealed"], rows[1]["blend_sealed"]
    print(f"  agent gap       (A - t13@budget) {0.132446 - a:+.5f}")
    print("\n  The capability gap is NOT (B - A) sealed, and the first version of")
    print("  this script printed it as if it were. `finalize_run` ships the argmax")
    print("  of the probe scores: in arm A that is the ablated table (no `_latest`")
    print(f"  pair), in arm B the un-ablated one. So B - A sealed = {a - b:+.5f} is")
    print("  mostly the pair, not the calendar windows.")
    print("\n  The clean reading is the probe, at the same step of both arms, both")
    print("  carrying the pair: A 0.14098 against B 0.13976, so the three calendar")
    print("  windows are worth +0.00122 — against +0.01428 for dropping the pair.")
    print(f"\n  his single un-ensembled model, private: ~0.105 "
          f"({leaderboard_rank(0.105)})")
    print(f"  private board: {leaderboard(True)}")
    print("\n  Sealed scores are not Kaggle scores; 0060 says the seal is a lower")
    print("  bound at the horizon we are graded at. No submission is written.")

    pd.DataFrame(rows).to_csv(os.path.join(RUNS, "winner_salesmodel.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
