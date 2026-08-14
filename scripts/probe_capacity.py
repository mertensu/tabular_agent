"""Is the harness's fixed model budget costing us score, and where?

Every fit in this project runs at **200 trees, learning rate 0.1**, on every
dataset, in every run:

    probe.py:110      PROBE_PARAMS = {"n_estimators": 200}
    models.py:419-432 lgbm / xgb / catboost defaults: n_estimators=200
    learning_rate     never set anywhere — whatever the library ships

Nothing tunes it, nothing stops early, and 0035 removed the agent's ability to
touch it on purpose. 0039 named this and left it open. The standing suspicion is
that it is why the two large datasets land mid-table while house-prices, at 1,168
rows, reached the top 9% — 200 trees is roughly right for a small table and
plainly not for 850,000 rows.

This script is the cheapest decisive test of that. It holds the table and the
validation rows fixed and varies **only capacity**.

## What is being measured, exactly

Two readings, and they answer different questions:

**1. The baseline table, on all four datasets.** The raw numeric columns, no
engineered features, same ordered split each dataset's runs used. This is the
cross-dataset reading: it says whether the fixed budget bites *everywhere* or
only where the rows are many. Nothing here can be blamed on a feature.

**2. Rossmann's v8 best table (`t24`, probe 0.136692).** The number that decides
whether the project's priorities are wrong. Three sessions went into feature
vocabulary; 0052 and 0053 spent a whole slice discovering the newest feature is
worth less than one we already had. If capacity on the *same* table is worth more
than the entire feature search, that is the finding.

## The prediction, registered before running

Written down first so the result cannot be read backwards into a story that was
never held (0022 is the standing lesson; 0052 is the recent one).

* **Rossmann and Porto Seguro gain.** Predicted more than 0.02 rmspe on Rossmann,
  which is the threshold the plan set for "this dwarfs everything the feature
  search has bought" — v8's whole feature chain moved the probe from 0.232 (t2)
  to 0.137 (t24), so 0.02 is about a fifth of that, from one parameter.
* **House-prices gains little or loses.** 1,168 rows, and more capacity on a
  table that small is mostly more variance.
* **Restaurant is the honest out-of-sample test of the story.** 252,108 rows —
  between house-prices and Rossmann by an order of magnitude at each end. If the
  story is "capacity matters in proportion to rows" it should land in between. It
  was prepared today and no capacity result existed when this was written.
* **Early stopping roughly matches the best fixed budget** without being told the
  answer. If it does not, the fix is not "raise the constant" but "the harness has
  to choose, per dataset", which is a different and larger piece of work.

The prediction that would falsify the whole capacity story: **Rossmann gains less
than 0.005.** That would mean sixteen runs of mediocre forecasting scores are
about something else entirely and this line of attack is closed.

## Why the ruler is monkeypatched rather than reimplemented

`_swap_probe_params` writes `probe.PROBE_PARAMS` for the duration of one fit, and
the score then comes from `probe_table` — the harness's own function, the one
that produced every number in `_runs/`. Reimplementing the fit here would have
made every reading incomparable with the run journals it is supposed to explain,
and a reimplementation that differed in one detail would look exactly like a
capacity effect. Ugly, deliberate, and the ugliness is confined to one context
manager.

The early-stopping arm cannot go through that path, because it needs an
`eval_set` the probe has no argument for. It is therefore two steps: fit on the
work half minus its last 10% (by the split column, so the inner validation is
*later* than the inner training — same shape as the real task), read
`best_iteration_`, then refit at that many trees through the ordinary path and
score on the real validation half. The number reported is thus produced by the
same code as every other row of the table, and only the choice of `n_estimators`
came from somewhere else.

## What this script is not

Not a tuning run, and its output is not a setting to adopt. It measures whether a
knob matters. If it does, what the harness should *do* about it — a common budget
per 0039, early stopping, or a tuning step — is a design decision that needs its
own record, and "the number that won here" is not that decision. Datasets are
probes, not targets.
"""
import contextlib
import os
import shutil
import sys
import time

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from tabular_agent import probe  # noqa: E402
from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import _feature_frame, probe_table  # noqa: E402
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

RUNS = os.path.join(REPO, "_runs")

#: The budgets, as (label, params). `current` is not a candidate — it is the
#: harness as it stands today, and every other row is read against it.
#:
#: The grid trades learning rate against rounds along the one axis that matters:
#: total shrinkage. 200x0.1 is the incumbent; 1000x0.03 spends the same nominal
#: "distance" in thirty times more steps; 5000x0.03 is genuinely more capacity.
#: Nothing else is varied — no depth, no leaf count, no subsampling — because a
#: grid that moves four things at once cannot say which one paid (0017).
BUDGETS: list[tuple[str, dict]] = [
    ("current  200 @ lr 0.10", {"n_estimators": 200}),
    ("        1000 @ lr 0.10", {"n_estimators": 1000, "learning_rate": 0.1}),
    ("        1000 @ lr 0.03", {"n_estimators": 1000, "learning_rate": 0.03}),
    ("        3000 @ lr 0.03", {"n_estimators": 3000, "learning_rate": 0.03}),
    ("        5000 @ lr 0.03", {"n_estimators": 5000, "learning_rate": 0.03}),
]

#: The learning rate the early-stopping arm searches rounds at. Fixed, because
#: the arm's question is "can the harness find the budget without being told",
#: not "what is the best (rate, rounds) pair" — that would be tuning.
EARLY_STOP_LR = 0.03
EARLY_STOP_MAX = 20000
EARLY_STOP_ROUNDS = 100
#: The tail of the work half used as the inner validation. A tenth: enough rows
#: to stop on, few enough that the refit is not fitting a visibly smaller table.
INNER_TAIL = 0.1


@contextlib.contextmanager
def _swap_probe_params(params: dict):
    """Run the harness's own ruler at a different budget, then put it back.

    See the module docstring for why this is a monkeypatch and not a
    reimplementation. `probe_table` reads `PROBE_PARAMS` at fit time, so
    rebinding the module attribute is enough, and restoring it in a `finally`
    keeps one arm's failure from silently contaminating the next.
    """
    original = probe.PROBE_PARAMS
    probe.PROBE_PARAMS = dict(params)
    try:
        yield
    finally:
        probe.PROBE_PARAMS = original


# --------------------------------------------------------------------------
# The four datasets, each configured exactly as its own runs configured it.
# Deviating here would mean the capacity reading is entangled with a split
# change, which is the one thing this script exists not to do.
# --------------------------------------------------------------------------

def _rossmann() -> tuple[RunSession, str]:
    from prep_rossmann import ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT

    serving = [c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"]
    return RunSession(
        os.path.join(RUNS, "capacity_rossmann"), OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=serving,
    ), "Date"


def _restaurant() -> tuple[RunSession, str]:
    from prep_restaurant import ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT

    serving = [c for c in pd.read_parquet(TEST_OUT).columns if c != "id"]
    return RunSession(
        os.path.join(RUNS, "capacity_restaurant"), OUT,
        TaskSpec(target="visitors", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        # Rossmann's fractions, not the library default. The forecast window is
        # 39 days against 476 of history — 8% — so a 5% validation block is the
        # closest thing to "one horizon" this split can cut, and matching
        # Rossmann is what makes the two readings comparable at all.
        val_fraction=0.05, holdout_fraction=0.05, split_on=SPLIT_ON,
        serving_columns=serving,
    ), "visit_date"


def _house() -> tuple[RunSession, str]:
    data = os.path.join(REPO, "house_prices", "train_no_id.csv")
    return RunSession(
        os.path.join(RUNS, "capacity_house"), data,
        TaskSpec(target="SalePrice"),
    ), None


def _seguro() -> tuple[RunSession, str]:
    from run_seguro_v4 import DATA, METRIC, TARGET

    return RunSession(
        os.path.join(RUNS, "capacity_seguro"), DATA,
        TaskSpec(target=TARGET, metric=METRIC),
        val_fraction=0.2,
    ), None


DATASETS = {
    "house": _house,
    "restaurant": _restaurant,
    "rossmann": _rossmann,
    "seguro": _seguro,
}

#: Rossmann's work table holds `Customers`, which is not in `test.csv`, so 0030
#: refuses to score it at *any* budget — the first attempt at this sweep came
#: back with five identical refusals and no numbers. Dropping it is the agent's
#: first move on this dataset, and it has to happen here too or there is no
#: servable baseline for capacity to be measured against.
#:
#: It is the only pre-step any of the four datasets needs, and it is a *table*
#: change rather than a capacity one, so it happens once, before the sweep, and
#: every budget then sees the same table.
DROP_CUSTOMERS = (
    "def transform_features(df):\n"
    "    return df.drop(columns=['Customers'], errors='ignore')\n"
)

BASELINE_PRESTEP = {"rossmann": DROP_CUSTOMERS}


def baseline_table(name: str, session: RunSession) -> str:
    """The table the sweep runs on: the work half, made servable if it is not.

    Returns a path either way, so the caller does not branch. `transform_features`
    is the harness's own tool, which means the resulting table is one an agent
    could have built — not a privately-constructed frame that no run could
    reproduce.
    """
    if name not in BASELINE_PRESTEP:
        return session.data_path
    from tabular_agent.harness.tools import transform_features

    out = transform_features(session, BASELINE_PRESTEP[name])
    print(f"  pre-step: dropped Customers -> {out['table']}")
    # The tool returns a path relative to the run directory, which is what the
    # agent sees; `resolve_table` is how the harness turns that back into
    # something openable, and skipping it is how this failed the first time.
    return session.resolve_table(out["table"])


#: The control the first pass needed and did not have.
#:
#: Rossmann's baseline carries `Store` — 1115 identifiers that arrive already
#: typed as an integer, so `_feature_frame` keeps them. Restaurant's
#: `air_store_id` is a *string*, so it is dropped, and the restaurant baseline
#: therefore has **no entity identity in it at all**.
#:
#: That makes "Rossmann gains from capacity and restaurant does not" ambiguous
#: between two very different explanations: capacity matters in proportion to
#: rows, or capacity matters when there are many per-entity effects to learn and
#: 200 trees cannot get to them. The two datasets differ on both axes at once,
#: which is exactly the confound 0017 forbids.
#:
#: So: give restaurant the same *kind* of column Rossmann has for free, and
#: nothing else. Deliberately *not* `fit_encoding` — a target encoding would add
#: information as well as identity and so put the confound straight back. What is
#: wanted is what Rossmann's `Store` already is: an arbitrary integer label a
#: tree can split on, carrying no meaning beyond "these rows are the same shop".
#:
#: The obvious way to write that is `factorize`, and the harness **refused it**:
#: a positional code depends on which rows are in the frame, so the same store
#: gets a different number in a batch of one, and 0011's stability check caught
#: exactly that. The refusal is correct and is worth more than the convenience.
#: So the code is derived from the identifier itself — the ids are `air_` plus 16
#: hex characters, and the first eight of them make a stable 32-bit label that is
#: a pure function of the row.
ADD_STORE_CODE = (
    "def transform_features(df):\n"
    "    df = df.copy()\n"
    "    df['air_store_code'] = df['air_store_id'].str[4:12].apply(int, base=16)\n"
    "    return df\n"
)


def early_stopped_rounds(session: RunSession, table_path: str, order_col: str | None) -> int | None:
    """How many trees an inner early stop asks for, at `EARLY_STOP_LR`.

    The inner split is the *tail* of the work half when there is an order column,
    so the rounds are chosen against later rows than they are fitted on — the
    same shape as the task. On an unordered dataset it is the last rows as they
    sit, which is a random block because the split already shuffled.

    Returns `None` when the fit could not be made rather than raising: an arm
    that failed should leave a blank in the table, not end the script and lose
    the arms that worked.
    """
    import lightgbm as lgb

    spec = session.resolved()
    df = load_table(table_path)
    if order_col and order_col in df.columns:
        df = df.sort_values(order_col, kind="stable").reset_index(drop=True)
    X, _ = _feature_frame(df, spec.target)
    y = df[spec.target]

    transform = spec.target_transform or probe.default_target_transform(
        spec.metric, spec.task_type
    )
    if transform != "none":
        from tabular_agent.target import apply_target
        y = pd.Series(apply_target(transform, y))

    cut = int(len(df) * (1 - INNER_TAIL))
    if cut < 50 or len(df) - cut < 20:
        return None

    cls = lgb.LGBMRegressor if spec.task_type == "regression" else lgb.LGBMClassifier
    model = cls(random_state=probe.PROBE_SEED, verbose=-1,
                n_estimators=EARLY_STOP_MAX, learning_rate=EARLY_STOP_LR)
    try:
        model.fit(
            X.iloc[:cut], y.iloc[:cut],
            eval_set=[(X.iloc[cut:], y.iloc[cut:])],
            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False)],
        )
    except Exception as e:  # noqa: BLE001
        print(f"    early stop failed: {type(e).__name__}: {e}")
        return None
    return int(model.best_iteration_ or 0) or None


def sweep(name: str, session: RunSession, table_path: str, order_col: str | None,
          label: str) -> list[dict]:
    """Every budget against one table, printed as it goes.

    Printed rather than collected-then-printed because the large arms take
    minutes and a run that dies at 5000 trees should still leave its earlier rows
    on screen.
    """
    spec = session.resolved()
    print(f"\n  {label}  ({spec.metric}, {spec.task_type})")

    rows: list[dict] = []
    baseline: float | None = None
    for budget_label, params in BUDGETS:
        started = time.monotonic()
        with _swap_probe_params(params):
            out = probe_table(session, table_path)
        elapsed = time.monotonic() - started
        score = out.get("probe_score")
        if score is None:
            print(f"    {budget_label:<24} unscored — {out.get('probe_unavailable')}")
            continue
        if baseline is None:
            baseline = score
        # `higher_is_better` differs by metric, so the delta is reported in the
        # direction that means "better" rather than as a raw subtraction: rmspe
        # and rmsle go down, auc goes up.
        from tabular_agent.metrics import greater_is_better
        gain = (score - baseline) if greater_is_better(spec.metric) else (baseline - score)
        print(f"    {budget_label:<24} {score:.5f}   {gain:+.5f} vs current"
              f"   ({elapsed:.0f}s)")
        rows.append({"dataset": name, "table": label, "budget": budget_label,
                     "score": score, "gain_vs_current": gain, "seconds": elapsed})

    n = early_stopped_rounds(session, table_path, order_col)
    if n is None:
        print(f"    {'early stop @ lr 0.03':<24} unavailable")
        return rows
    with _swap_probe_params({"n_estimators": n, "learning_rate": EARLY_STOP_LR}):
        out = probe_table(session, table_path)
    score = out.get("probe_score")
    if score is None:
        print(f"    {'early stop @ lr 0.03':<24} unscored — {out.get('probe_unavailable')}")
        return rows
    from tabular_agent.metrics import greater_is_better
    gain = (score - baseline) if greater_is_better(spec.metric) else (baseline - score)
    print(f"    {f'early stop -> {n}':<24} {score:.5f}   {gain:+.5f} vs current")
    rows.append({"dataset": name, "table": label, "budget": f"early stop -> {n}",
                 "score": score, "gain_vs_current": gain, "seconds": 0.0})
    return rows


def main() -> int:
    wanted = sys.argv[1:] or list(DATASETS)
    extras = ("rossmann_rich", "restaurant_store")
    unknown = [w for w in wanted if w not in DATASETS and w not in extras]
    if unknown:
        print(f"unknown dataset(s): {unknown}; known: {list(DATASETS)}", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    for name in wanted:
        if name in extras:
            continue
        print(f"\n=== {name} ===")
        run_dir = os.path.join(RUNS, f"capacity_{name}")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        session, order_col = DATASETS[name]()
        print(f"  work {len(load_table(session.data_path)):,} rows | "
              f"val {len(load_table(session.val_path)):,} rows")
        work = baseline_table(name, session)
        all_rows += sweep(name, session, work, order_col, "baseline table")

    # The control. See `ADD_STORE_CODE`: restaurant's baseline has no entity
    # identity in it, Rossmann's has 1115 of them, and until this is run the two
    # datasets differ on two axes at once.
    if "restaurant_store" in wanted or not sys.argv[1:]:
        print("\n=== restaurant + an integer store code (the control) ===")
        run_dir = os.path.join(RUNS, "capacity_restaurant_store")
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        session, order_col = _restaurant()
        session = RunSession(
            run_dir, session.source_path, session.spec,
            val_fraction=0.05, holdout_fraction=0.05, split_on="visit_date",
            serving_columns=(session.serving_columns or []) + ["air_store_code"],
        )
        from tabular_agent.harness.tools import transform_features

        coded = transform_features(session, ADD_STORE_CODE)
        all_rows += sweep("restaurant", session,
                          session.resolve_table(coded["table"]),
                          order_col, "+ store code")

    # Reading 2, and it is not the reading that was planned.
    #
    # The plan was to sweep capacity on v8's own best table, `t24`. That is not
    # possible at HEAD: `t24` was written at 13:39 and commit 9879acd (0051)
    # landed at 15:26, renaming `HistoryEncoder`'s output from `_last`/`_age` to
    # `_latest`/`_latest_ago`. Replaying t24's stored lineage through today's code
    # therefore produces columns under the new names while the table holds the
    # old ones, and 0030's servability check correctly refuses all six.
    #
    # That is a real defect and it is written up separately. Here it just means
    # the second reading has to be *rebuilt* rather than replayed — so this takes
    # the one part of t24 that is pure code and therefore version-independent:
    # its first transform step, the agent's own feature block. Date parts,
    # competition and promo durations, and the `Store x DayOfWeek` /
    # `Store x Promo` interactions — which is exactly the high-cardinality
    # identity structure the restaurant control implicated.
    #
    # It is not t24 and is not claimed to be. It answers the narrower question
    # that survives: does capacity still pay *after* an agent has done its feature
    # work, or does feature work absorb the gain?
    if "rossmann_rich" in wanted or not sys.argv[1:]:
        pipeline = os.path.join(RUNS, "rossmann_v8", "tables", "t24.pipeline.json")
        if not os.path.exists(pipeline):
            print(f"\n  skipping rossmann_rich: {pipeline} not found")
        else:
            import json

            with open(pipeline) as f:
                steps = json.load(f)["steps"]
            code = next(s["code"] for s in steps if s.get("type") == "transform")

            print("\n=== rossmann, v8's engineered feature block (rebuilt) ===")
            run_dir = os.path.join(RUNS, "capacity_rossmann_rich")
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
            session, order_col = _rossmann()
            session = RunSession(
                run_dir, session.source_path, session.spec,
                val_fraction=0.05, holdout_fraction=0.05, split_on="Date",
                serving_columns=session.serving_columns,
            )
            from tabular_agent.harness.tools import transform_features

            rich = transform_features(session, code)
            rich = transform_features(session, DROP_CUSTOMERS,
                                      table=rich["table"])
            all_rows += sweep("rossmann", session,
                              session.resolve_table(rich["table"]),
                              "Date", "v8 feature block")

    if all_rows:
        # Appended, not overwritten. The sweep takes long enough on the big
        # datasets that it gets run one argument at a time, and the first version
        # of this line quietly replaced the results of every previous invocation.
        out = os.path.join(RUNS, "capacity_probe.csv")
        fresh = pd.DataFrame(all_rows)
        if os.path.exists(out):
            old = pd.read_csv(out)
            keys = set(zip(fresh.dataset, fresh.table))
            old = old[[k not in keys for k in zip(old.dataset, old.table)]]
            fresh = pd.concat([old, fresh], ignore_index=True)
        fresh.to_csv(out, index=False)
        print(f"\nwrote {out}  ({len(fresh)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
