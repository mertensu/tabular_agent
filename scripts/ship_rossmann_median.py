"""v8's full pipeline, rebuilt at HEAD, fitted at a chosen budget, blended.

Two effects have now been confirmed on Kaggle's own rows, separately:

* **the pipeline** is worth ~0.095 at a fixed 200 trees — v7's full chain scored
  0.13960 public where a bare feature block scored 0.23420;
* **the budget** is worth ~0.079 — `ship_rossmann_capacity.py`'s A/B, 0.23048 vs
  0.15166 private, against a sealed-holdout prediction of +0.085.

Neither alone reaches the median (public 0.11677); our best-ever submission sits
at 0.12932, a gap of 0.0126. This script is the one attempt to have both at once.

## What it does

Re-executes v8's `t24` chain **step by step at HEAD**, rather than replaying its
stored lineage. That is the point: 0055 records that 0051's rename left those
artifacts unreplayable, and re-running the steps sidesteps it completely — the
encoders refit here and emit today's column names. The parameters are read out of
`t24.pipeline.json` so they cannot drift from what v8 actually did.

Then each of the three GBDT families is fitted at a budget chosen by early
stopping on a *later* block of the work half, and the three are averaged.

## The discipline, stated before running

This script aims at a leaderboard number, which is the thing CLAUDE.md warns
about, so the boundary has to be explicit:

* **Nothing here is tuned against the leaderboard.** The pipeline is v8's, taken
  as-is. The budget comes from early stopping, which reads no Kaggle score. The
  learning rate is the same 0.03 used in every earlier reading. The blend is
  0044's equal-weight average of everything that trained.
* **One shot, and the result is recorded either way.** If it misses the median it
  is written up as a miss. Iterating on this script until a number goes green is
  exactly the failure 0022 exists to name, and it is the temptation this note is
  here to refuse.
* **No feature search.** Not one column is added beyond what v8 built.

## Known risks, named in advance

* v8's chain includes `fit_history`, and 0049/0050/0052/0053 are a run of negative
  findings about that feature class — including that a window's measured worth is
  substantially an artifact of the fold count. It is kept because the goal is to
  rebuild v8's table faithfully, not to curate it, and curating it *would* be
  fitting to the leaderboard.
* The blend is 0044's arithmetic mean of inverted predictions. `predict_bundle`'s
  geometric-mean defect (HANDOFF, 2026-08-14) is why the inversion order matters
  and why this composes predictions rather than models.
* The early-stopping block is ordered by `Date` explicitly. Rossmann's file is in
  reverse-chronological order, so "the last 10% of rows" is the *oldest* period —
  the bug found in `ship_rossmann_capacity.py`, and the reason `order` is a
  required argument here rather than an assumption.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from tabular_agent.harness.session import RunSession  # noqa: E402
from tabular_agent.harness.tools import (  # noqa: E402
    _feature_frame,
    filter_rows,
    fit_encoding,
    fit_group_stats,
    fit_history,
    transform_features,
)
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.models import _make_model, predict_model  # noqa: E402
from tabular_agent.probe import PROBE_SEED, default_target_transform  # noqa: E402
from tabular_agent.target import apply_target, invert_target  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

from prep_rossmann import ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT  # noqa: E402

RUN_DIR = os.path.join(REPO, "_runs", "ship_rossmann_median")
T24_PIPELINE = os.path.join(REPO, "_runs", "rossmann_v8", "tables", "t24.pipeline.json")

VAL_FRACTION = 0.05
HOLDOUT_FRACTION = 0.05

LR = 0.03
MAX_ROUNDS = 30000
PATIENCE = 200
INNER_TAIL = 0.1

#: The three families `finalize_run` would fit. `elasticnet` is excluded, and
#: this is the one deviation from 0044's "every candidate, no filtering" — it has
#: no early-stopping equivalent, so including it would mean blending a model at a
#: budget nobody chose into a blend whose entire premise is a chosen budget. Named
#: here rather than buried, because 0044's argument against dropping members is
#: good and this is a real exception to it.
FAMILIES = ["lgbm", "xgb", "catboost"]


def serving_columns() -> list[str]:
    return [c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"]


def rebuild_v8_table(session: RunSession, stop_before_history: bool = False) -> str:
    """Re-execute t24's chain at HEAD, reading its parameters off the stored spec.

    Nine steps. Each one goes through the harness tool an agent would have called,
    so what comes out is a table with a lineage this version of the library can
    replay — which is exactly what the stored artifact no longer has (0055).
    """
    with open(T24_PIPELINE) as f:
        steps = json.load(f)["steps"]

    table = None
    for i, st in enumerate(steps):
        kind = st.get("type")
        if kind == "transform":
            out = transform_features(session, st["code"], table=table)
            what = "transform"
        elif kind == "filter":
            out = filter_rows(session, st["code"], table=table)
            what = "filter_rows"
        elif kind == "encode":
            e = st["encoder"]
            if e["kind"] == "group_stats":
                out = fit_group_stats(session, by=e["by"], value=e["value"],
                                      stat=e["stat"], table=table)
                what = f"group_stats {e['by']} of {e['value']}"
            elif e["kind"] == "frequency":
                out = fit_encoding(session, columns=e["columns"],
                                   method="frequency", table=table)
                what = f"frequency {e['columns']}"
            elif e["kind"] == "target":
                out = fit_encoding(session, columns=e["columns"], method="target",
                                   stats=e["stats"], table=table)
                what = f"target {e['columns']} {e['stats']}"
            elif e["kind"] == "history":
                if stop_before_history:
                    print(f"  {i}. history — SKIPPED (this is the no-history arm)")
                    break
                out = fit_history(session, by=e["by"], value=e["value"],
                                  offsets=e["offsets"], keep=e["keep"], table=table)
                what = f"history {e['by']} of {e['value']} offsets {e['offsets']}"
            else:
                raise SystemExit(f"step {i}: unknown encoder {e['kind']}")
        else:
            raise SystemExit(f"step {i}: unknown step type {kind}")

        if out.get("table") is None:
            raise SystemExit(f"step {i} ({what}) produced no table: {out}")
        table = out["table"]
        print(f"  {i}. {what:<44} -> {table}  probe {out.get('probe_score')}")

    return session.resolve_table(table)


def choose_rounds(family: str, X: pd.DataFrame, y_fit: pd.Series,
                  order: pd.Series) -> int:
    """How many rounds this family wants, read off a later block of the work half.

    Per family, because there is no reason three different implementations agree
    on a round count and 0039's whole complaint was that enumerating library
    defaults measures the libraries rather than the models.
    """
    idx = np.argsort(order.to_numpy(), kind="stable")
    Xs, ys = X.iloc[idx].reset_index(drop=True), y_fit.iloc[idx].reset_index(drop=True)
    cut = int(len(Xs) * (1 - INNER_TAIL))

    if family == "lgbm":
        import lightgbm as lgb

        m = lgb.LGBMRegressor(random_state=PROBE_SEED, verbose=-1,
                              n_estimators=MAX_ROUNDS, learning_rate=LR)
        m.fit(Xs.iloc[:cut], ys.iloc[:cut],
              eval_set=[(Xs.iloc[cut:], ys.iloc[cut:])],
              callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        return int(m.best_iteration_ or MAX_ROUNDS)

    if family == "xgb":
        import xgboost as xgb

        m = xgb.XGBRegressor(random_state=PROBE_SEED, verbosity=0,
                             n_estimators=MAX_ROUNDS, learning_rate=LR,
                             early_stopping_rounds=PATIENCE)
        m.fit(Xs.iloc[:cut], ys.iloc[:cut],
              eval_set=[(Xs.iloc[cut:], ys.iloc[cut:])], verbose=False)
        return int(m.best_iteration or MAX_ROUNDS) + 1

    from catboost import CatBoostRegressor

    m = CatBoostRegressor(random_state=PROBE_SEED, verbose=False,
                          n_estimators=MAX_ROUNDS, learning_rate=LR,
                          early_stopping_rounds=PATIENCE)
    m.fit(Xs.iloc[:cut], ys.iloc[:cut],
          eval_set=(Xs.iloc[cut:], ys.iloc[cut:]), verbose=False)
    return int(m.get_best_iteration() or MAX_ROUNDS) + 1


def predict_on(model, frame: pd.DataFrame, features: list[str],
               transform: str) -> np.ndarray:
    """Predictions on the target's own scale.

    Inverted **per model**, before any averaging. That order is load-bearing:
    averaging on a `log1p` scale and inverting once at the end is a *geometric*
    mean of the target, which is the `predict_bundle` defect found on 2026-08-14.
    """
    preds = predict_model(model, frame[features], "regression")
    return invert_target(transform, preds) if transform != "none" else preds


def main() -> int:
    # The second arm, and the reason it exists is a fact rather than a hunch:
    # **v7 has no `fit_history` step anywhere**, and v7 is the run that produced
    # our best external result (public 0.12932). v8 added history, scored better
    # on val, and was never submitted. The sealed holdout here — the period
    # immediately before Kaggle's test window — says history costs 0.019.
    #
    # Both arms are submitted. This is an A/B measured on Kaggle, exactly like
    # the capacity pair, and deliberately *not* a choice made on the sealed set:
    # picking the better one internally is what 0041/0042 forbid.
    no_history = "--no-history" in sys.argv
    run_dir = RUN_DIR + ("_nohistory" if no_history else "")

    for p in (OUT, T24_PIPELINE):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    session = RunSession(
        run_dir, OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        val_fraction=VAL_FRACTION, holdout_fraction=HOLDOUT_FRACTION,
        split_on=SPLIT_ON, serving_columns=serving_columns(),
    )
    spec = session.resolved()
    print(f"work {len(load_table(session.data_path)):,} | "
          f"val {len(load_table(session.val_path)):,} | metric {METRIC}\n")

    label = "no-history (v7-shaped)" if no_history else "full v8 chain"
    print(f"rebuilding {label} at HEAD:")
    table_path = rebuild_v8_table(session, stop_before_history=no_history)

    df = load_table(table_path)
    X, _ = _feature_frame(df, spec.target)
    y = df[spec.target]
    transform = spec.target_transform or default_target_transform(METRIC, spec.task_type)
    y_fit = pd.Series(apply_target(transform, y)) if transform != "none" else y
    order = df[SPLIT_ON]
    print(f"\n{len(df):,} rows, {X.shape[1]} numeric features, target on {transform}")

    val = load_table(session.val_path)
    holdout_path = os.path.join(run_dir, "data", "holdout.parquet")
    hold = load_table(holdout_path) if os.path.exists(holdout_path) else None
    test = pd.read_parquet(TEST_OUT)

    # Every frame the model will be asked about, put through the *same* lineage —
    # the tool's own replay, not a reimplementation. A wrong lineage still yields
    # a plausible CSV, which is why this goes through the pipeline the bundle uses.
    from tabular_agent.pipeline import Pipeline

    with open(table_path.replace(".parquet", ".pipeline.json")) as f:
        pipe = Pipeline.from_dict(json.load(f))
    val_X = pipe.transform(val.drop(columns=[spec.target]))
    hold_X = pipe.transform(hold.drop(columns=[spec.target])) if hold is not None else None
    test_X = pipe.transform(test.drop(columns=["Id"]))

    features = list(X.columns)
    val_p, hold_p, test_p, rows = [], [], [], []
    for family in FAMILIES:
        print(f"\n=== {family}")
        t0 = time.monotonic()
        rounds = choose_rounds(family, X, y_fit, order)
        print(f"  early stop chose {rounds:,} rounds at lr {LR}")
        if rounds >= MAX_ROUNDS:
            print(f"  !! hit the ceiling of {MAX_ROUNDS:,}")

        model = _make_model(family, "regression",
                            {"n_estimators": rounds, "learning_rate": LR},
                            PROBE_SEED, METRIC)
        model.fit(X, y_fit)

        vp = predict_on(model, val_X, features, transform)
        val_p.append(vp)
        v = compute_metric(METRIC, val[spec.target].to_numpy(), vp)
        h = None
        if hold is not None:
            hp = predict_on(model, hold_X, features, transform)
            hold_p.append(hp)
            h = compute_metric(METRIC, hold[spec.target].to_numpy(), hp)
        test_p.append(predict_on(model, test_X, features, transform))

        print(f"  val {v:.5f}" + (f" | sealed holdout {h:.5f}" if h else "")
              + f"  ({time.monotonic() - t0:.0f}s)")
        rows.append({"model": family, "rounds": rounds, "val": v, "holdout": h})

    # 0044: an equal-weight average of every member, nothing fitted, nothing
    # chosen. Members were fixed before any sealed row was read.
    blend_val = np.mean(val_p, axis=0)
    bv = compute_metric(METRIC, val[spec.target].to_numpy(), blend_val)
    bh = None
    if hold is not None:
        bh = compute_metric(METRIC, hold[spec.target].to_numpy(), np.mean(hold_p, axis=0))
    rows.append({"model": "BLEND (equal weight)", "rounds": None, "val": bv, "holdout": bh})

    print("\n=== the board ===")
    print(pd.DataFrame(rows).to_string(index=False))

    blend_test = np.mean(test_p, axis=0)
    out = os.path.join(run_dir, "submission_blend.csv")
    pd.DataFrame({"Id": test["Id"], "Sales": blend_test}).to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"  {int((blend_test < 0).sum())} predictions below zero, left as they are")
    print("\npublic median is 0.11677; our best ever is 0.12932 (v7 blend).")
    print("Whatever this scores is the result — it is not iterated on (see the docstring).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
