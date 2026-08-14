"""Two Rossmann submissions that differ by one thing: the model's budget.

0054 measured capacity on validation blocks and found a large effect, a mechanism
(per-entity identity, not row count), and a sign reversal in the ruler. All of it
is *internal*. Nothing there has been converted into a prediction anyone could
send anywhere, and a finding that never reaches a shipped model has not yet
earned anything.

This script converts it. One table, one seed, one set of rows, two fits:

* **A — 200 trees at lr 0.1.** Not a candidate: it is what this harness ships
  today, reproduced exactly, so it can be the control.
* **B — early-stopped at lr 0.03.** The budget 0054 argues for, chosen by the
  data rather than set here (see `fit_arm_b`).

Both are scored on the run's validation half and on its sealed holdout, and both
write a Kaggle submission. **Both get submitted.** That matters: the point is to
measure the difference against rows nothing in this project has touched, not to
pick a winner internally and send only that one — which is 0042's failure, and
would also make the sealed holdout a selection set (0041).

## Why this is not leaderboard-fitting

The distinction CLAUDE.md draws is whether a change can be defended on its own
terms without the leaderboard. Capacity can: the effect reproduces on four
datasets, the mechanism was isolated by a control that added *identity and nothing
else*, and the treatment arm uses early stopping, which needs no per-dataset
knowledge and asked for wildly different budgets on each. Kaggle is being used to
check whether an internally-measured effect survives contact with a real test
period — which is what 0043 says we cannot assume — not to choose the budget.

If B loses, the finding stands as an internal measurement and the terminal phase
question stays open. That outcome is recorded either way.

## What is rebuilt, and why it is not v8's table

v8's `t24` cannot be replayed at HEAD (0055: 0051's rename landed four hours after
those tables were written). So the table here is rebuilt from the one part of that
pipeline that is pure code and therefore version-independent — the agent's own
feature block — plus the `Customers` drop that makes it servable at all.

**This means B is not comparable to v7's Kaggle 0.13813.** A is the control that
makes the pair meaningful; the across-run comparison is not available and is not
claimed.
"""
import argparse
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
from tabular_agent.harness.tools import _feature_frame, transform_features  # noqa: E402
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.models import _make_model  # noqa: E402
from tabular_agent.pipeline import Pipeline, predict_bundle  # noqa: E402
from tabular_agent.probe import PROBE_SEED, default_target_transform  # noqa: E402
from tabular_agent.target import apply_target  # noqa: E402
from tabular_agent.taskspec import TaskSpec  # noqa: E402

from prep_rossmann import ABOUT, COLUMN_NOTES, METRIC, OUT, SPLIT_ON, TEST_OUT  # noqa: E402

RUN_DIR = os.path.join(REPO, "_runs", "ship_rossmann_capacity")

#: v7/v8's fractions, unchanged. The split is not what is being tested and
#: changing it would entangle the capacity reading with a different exam (0043).
VAL_FRACTION = 0.05
HOLDOUT_FRACTION = 0.05

#: Arm A: the harness as it stands. `learning_rate` is absent on purpose — that
#: is the point, the library's default is what has always been used.
ARM_A = {"n_estimators": 200}

#: Arm B's learning rate. The round count is *not* here: it is read off an early
#: stop, which is the whole argument of 0054 — no constant is defensible.
ARM_B_LR = 0.03
ARM_B_MAX = 30000
ARM_B_PATIENCE = 200
#: The tail of the work half held back to stop on. Later rows than the ones fitted
#: on, which is the same shape as the task; 0054 used 10% and this keeps it.
INNER_TAIL = 0.1

#: v8's feature block, lifted from t24's stored pipeline. Read from disk rather
#: than pasted so it cannot drift from the thing it claims to be, and it is the
#: only step of that pipeline that is pure code — everything else carries fitted
#: state whose column names moved under 0051.
T24_PIPELINE = os.path.join(REPO, "_runs", "rossmann_v8", "tables", "t24.pipeline.json")

DROP_CUSTOMERS = (
    "def transform_features(df):\n"
    "    return df.drop(columns=['Customers'], errors='ignore')\n"
)


def serving_columns() -> list[str]:
    return [c for c in pd.read_parquet(TEST_OUT).columns if c != "Id"]


def build_table(session: RunSession) -> str:
    """The work table plus v8's feature block, minus `Customers`.

    Two `transform_features` calls rather than one concatenated blob, because the
    lineage then records them as two steps and the second — the drop that makes
    the table servable — stays legible as its own decision.
    """
    with open(T24_PIPELINE) as f:
        steps = json.load(f)["steps"]
    code = next(s["code"] for s in steps if s.get("type") == "transform")

    out = transform_features(session, code)
    out = transform_features(session, DROP_CUSTOMERS, table=out["table"])
    path = session.resolve_table(out["table"])
    print(f"  table {out['table']}  probe {out.get('probe_score')}")
    return path


def fit_arm_a(X: pd.DataFrame, y_fit: pd.Series):
    """200 trees at the library's own learning rate. The control."""
    model = _make_model("lgbm", "regression", dict(ARM_A), PROBE_SEED, METRIC)
    model.fit(X, y_fit)
    return model, ARM_A["n_estimators"]


def fit_arm_b(X: pd.DataFrame, y_fit: pd.Series, order: pd.Series):
    """Early-stopped, then refitted on everything at the round count it chose.

    Two steps on purpose. The stop is read from a held-back block of the work
    half, so the number of rounds is chosen against rows *later* than the ones it
    was fitted on — the same shape as the task. It is then refitted on the
    **whole** work half at that count, so arm B trains on exactly the rows arm A
    does; otherwise the comparison would confound budget with training-set size,
    which is the one thing this script exists to avoid.

    The refit gets no early stopping and needs none: the round count is now a
    fixed number like any other.

    ``order`` is not optional and the first version of this function did not have
    it. Rossmann's file is in **reverse-chronological** order, so slicing the last
    10% of rows as they sit hands the stopper the *oldest* period — 2013-01-01 to
    2013-03-24 — while fitting on everything after it. That runs the forecast
    backwards, and it chose 2,989 rounds where the correct ordering chooses a
    different number. The rows are sorted here rather than assumed to be in order,
    because "the tail of the frame" is only "the latest period" on a file that
    happens to be stored that way, and this one is not.
    """
    import lightgbm as lgb

    order_idx = np.argsort(order.to_numpy(), kind="stable")
    X = X.iloc[order_idx].reset_index(drop=True)
    y_fit = y_fit.iloc[order_idx].reset_index(drop=True)
    ordered = order.iloc[order_idx].reset_index(drop=True)

    cut = int(len(X) * (1 - INNER_TAIL))
    print(f"  stopping block: {pd.Timestamp(ordered.iloc[cut]).date()} -> "
          f"{pd.Timestamp(ordered.iloc[-1]).date()} "
          f"(fitted on everything up to {pd.Timestamp(ordered.iloc[cut - 1]).date()})")
    stopper = lgb.LGBMRegressor(
        random_state=PROBE_SEED, verbose=-1,
        n_estimators=ARM_B_MAX, learning_rate=ARM_B_LR,
    )
    stopper.fit(
        X.iloc[:cut], y_fit.iloc[:cut],
        eval_set=[(X.iloc[cut:], y_fit.iloc[cut:])],
        callbacks=[lgb.early_stopping(ARM_B_PATIENCE, verbose=False)],
    )
    rounds = int(stopper.best_iteration_ or ARM_B_MAX)
    print(f"  early stop chose {rounds:,} rounds at lr {ARM_B_LR}")
    if rounds >= ARM_B_MAX:
        print(f"  !! hit the ceiling of {ARM_B_MAX:,} — the budget is still binding")

    model = _make_model(
        "lgbm", "regression",
        {"n_estimators": rounds, "learning_rate": ARM_B_LR}, PROBE_SEED, METRIC,
    )
    model.fit(X, y_fit)
    return model, rounds


def as_bundle(model, feature_names: list[str], table_path: str,
              target_transform: str) -> dict:
    """A bundle-shaped dict, so scoring and prediction go down the real path.

    `predict_bundle` cannot tell this from a bundle `finalize_run` wrote, which is
    the point: a hand-rolled prediction path could disagree with the served one
    and nothing would notice until Kaggle scored it.
    """
    lineage_path = table_path.replace(".parquet", ".pipeline.json")
    with open(lineage_path) as f:
        lineage = json.load(f)
    return {
        "pipeline": Pipeline.from_dict(lineage),
        "models": [model],
        "weights": np.array([1.0]),
        "feature_names": feature_names,
        "task_type": "regression",
        "target_transform": target_transform,
    }


def score_on(bundle: dict, path: str, target: str) -> float | None:
    if not os.path.exists(path):
        return None
    df = load_table(path)
    preds = predict_bundle(bundle, df.drop(columns=[target]))
    return float(compute_metric(METRIC, df[target].to_numpy(), preds))


def write_submission(bundle: dict, out_path: str) -> str:
    test = pd.read_parquet(TEST_OUT)
    ids = test["Id"]
    preds = predict_bundle(bundle, test.drop(columns=["Id"]))

    bad = int((~np.isfinite(preds)).sum())
    if bad:
        print(f"  !! {bad} non-finite predictions")
    neg = int((preds < 0).sum())
    if neg:
        # Left as they are rather than clipped: rmspe does not score zero-target
        # rows, and silently clipping would hide a model that has gone wrong.
        print(f"  note: {neg} predictions below zero, left as they are")

    pd.DataFrame({"Id": ids, "Sales": preds}).to_csv(out_path, index=False)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keep", action="store_true",
                   help="reuse an existing run dir instead of rebuilding it")
    a = p.parse_args()

    if not os.path.exists(OUT):
        sys.exit(f"missing {OUT} — run scripts/prep_rossmann.py first")
    if not os.path.exists(T24_PIPELINE):
        sys.exit(f"missing {T24_PIPELINE} — v8's tables are needed for the feature block")
    if os.path.exists(RUN_DIR) and not a.keep:
        shutil.rmtree(RUN_DIR)

    session = RunSession(
        RUN_DIR, OUT,
        TaskSpec(target="Sales", metric=METRIC, notes=ABOUT,
                 column_notes=COLUMN_NOTES, forecast_horizon=1.0),
        val_fraction=VAL_FRACTION, holdout_fraction=HOLDOUT_FRACTION,
        split_on=SPLIT_ON, serving_columns=serving_columns(),
    )
    spec = session.resolved()
    print(f"work {len(load_table(session.data_path)):,} | "
          f"val {len(load_table(session.val_path)):,} | metric {METRIC}")

    table_path = build_table(session)
    df = load_table(table_path)
    X, dropped = _feature_frame(df, spec.target)
    y = df[spec.target]
    transform = spec.target_transform or default_target_transform(METRIC, spec.task_type)
    y_fit = pd.Series(apply_target(transform, y)) if transform != "none" else y
    print(f"  {X.shape[1]} numeric features, target on {transform}")

    # Arm B needs the split column to order its stopping block by, and it is not
    # in `X` — `_feature_frame` drops it as non-numeric. Taken from the table
    # rather than assumed, per the bug in `fit_arm_b`'s docstring.
    order = df[SPLIT_ON]

    holdout_path = os.path.join(RUN_DIR, "data", "holdout.parquet")
    rows = []
    for name, fit in [("A  200 @ lr 0.10 (today)", fit_arm_a),
                      ("B  early-stopped @ lr 0.03", fit_arm_b)]:
        print(f"\n=== arm {name}")
        t0 = time.monotonic()
        model, rounds = (fit(X, y_fit) if name.startswith("A")
                         else fit(X, y_fit, order))
        elapsed = time.monotonic() - t0

        bundle = as_bundle(model, list(X.columns), table_path, transform)
        val = score_on(bundle, session.val_path, spec.target)
        hold = score_on(bundle, holdout_path, spec.target)
        tag = "a" if name.startswith("A") else "b"
        sub = write_submission(bundle, os.path.join(RUN_DIR, f"submission_{tag}.csv"))

        print(f"  val {val:.5f}" + (f" | sealed holdout {hold:.5f}" if hold else ""))
        print(f"  fitted in {elapsed:.0f}s -> {sub}")
        rows.append({"arm": name, "rounds": rounds, "val": val, "holdout": hold,
                     "seconds": round(elapsed, 1), "submission": sub})

    print("\n=== the pair ===")
    print(pd.DataFrame(rows)[["arm", "rounds", "val", "holdout"]].to_string(index=False))
    a_row, b_row = rows
    print(f"\n  capacity is worth {a_row['val'] - b_row['val']:+.5f} on val", end="")
    if a_row["holdout"] and b_row["holdout"]:
        print(f", {a_row['holdout'] - b_row['holdout']:+.5f} on the sealed holdout")
    else:
        print()
    print("\nSubmit BOTH to Kaggle. Sending only the better one makes the holdout a\n"
          "selection set (0041) and leaves the difference unmeasured on real rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
