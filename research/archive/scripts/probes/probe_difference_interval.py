"""Does the block instrument see what the row bootstrap cannot? Rossmann says yes.

`compare.py` ships two ways of putting an interval on a difference, and the
argument for the second one is entirely structural: under an ordered split the
validation rows are one *period*, so resampling within it cannot see an effect
that changes across periods. That argument predicts something checkable, and this
script checks it on the only case where the answer is already known from outside.

## The case

`_runs/ship_rossmann_median` ends its chain with `t9`, a `fit_history` table built
on `t8`. That one step is the history feature, and 0056 measured it externally:
the arm carrying it scored **0.18438** on Kaggle's private rows against
**0.15429** for the same pipeline without it — a real loss of 0.0301 — while our
own validation block ranked it **best of the four arms** at 0.13276.

0058 explains the discrepancy: the feature decays with distance from its
2015-05-01 freeze, so it looks strong on rows adjacent to the freeze and degrades
on rows further out. That is a difference which is *not constant across the
evaluation window*, which is exactly the shape a row bootstrap is blind to and the
shape blocks are meant to expose.

## The prediction, registered before running

1. The **paired bootstrap** over validation rows will report a small standard
   error around a positive gain — confident, and confidently misleading.
2. The **blocks** will disagree with each other: either `sign_stable` is false, or
   the block gains trend downward across the window. If they do not, the
   structural argument for the second mode is weaker than `compare.py` claims and
   the record should say so.

Run: ``uv run python scripts/probe_difference_interval.py``

## What happened — prediction 1 passed, prediction 2 was half wrong

    t8 (no history)  rmspe 0.15840      t9 (+ history)  rmspe 0.13812

    paired bootstrap over rows      gain +0.02028  se 0.00074   gain/se 27.3
      pairing ratio 0.371 - 0036's 0.18-0.37, reproduced on a second dataset

    blocks over the 44-day val window
      +0.02628 +0.00753 +0.01398 +0.04036 +0.01041
      sd 0.01214   sign_stable TRUE     gain/sd 1.7

    blocks over val + holdout, 90 days
      +0.02613 +0.00739 +0.01374 +0.04160 +0.01092
      -0.01081 +0.03086 +0.00279 +0.00126 -0.01286
      sd 0.01670   sign_stable FALSE    pooled gain +0.01152

**Prediction 2 was wrong on the window the harness actually uses.** Over the
validation half the blocks agreed unanimously — five out of five favouring the
history table — so a run would have reported a stable positive difference for a
feature that went on to lose 0.0301 on Kaggle.

What the blocks did do there is widen the interval by **16x**: `gain/se` falls
from 27.3 to 1.7. The row bootstrap called this difference twenty-seven standard
errors from zero. It is not, and the only thing that was ever wrong about it was
the yardstick.

**Prediction 2 was right once the window was long enough.** Doubled to 90 days the
blocks disagree, the pooled gain nearly halves, and the decay is legible directly:
the first five blocks average **+0.0200** and the last five **+0.0022**, a ninefold
fall with distance from the 2015-05-01 freeze. Kaggle's test window starts two
weeks after the last block here and scores −0.0301, which is the same line
continued.

So the finding is about **window length, not block count**: blocking cannot see
past the last row it has, and this run's validation half stops well inside the
regime where the feature still works. That is the limitation `compare.py` states
in advance, measured — and it is why a longer evaluation window, not more blocks,
is the thing to reach for when a difference has to hold at a horizon.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tabular_agent.compare import blockwise_difference, paired_difference  # noqa: E402
from tabular_agent.io import load_table  # noqa: E402
from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.pipeline import Pipeline  # noqa: E402
from tabular_agent.probe import fit_probe  # noqa: E402

RUN = os.path.join(os.path.dirname(__file__), "..", "_runs", "ship_rossmann_median")
TARGET = "Sales"
METRIC = "rmspe"
ORDER_COLUMN = "Date"


def _lineage(table_path: str) -> Pipeline:
    path = os.path.splitext(table_path)[0] + ".pipeline.json"
    if not os.path.exists(path):
        return Pipeline("regression")
    with open(path) as f:
        return Pipeline.from_dict(json.load(f))


def _score_table(table_id: str, val: pd.DataFrame):
    """Fit the ruler on one table, predict the shared validation rows.

    Identical to what `probe_table` does inside a run — one fit on all the table's
    rows, then the table's lineage replayed over raw validation rows — because the
    point is to measure the instrument the harness actually uses.
    """
    table_path = os.path.join(RUN, "tables", f"{table_id}.parquet")
    df = load_table(table_path)
    X = df.drop(columns=[TARGET]).select_dtypes(include="number")
    fit = fit_probe(X, df[TARGET], "regression", METRIC)

    features = _lineage(table_path).transform(val.drop(columns=[TARGET]))
    missing = [c for c in fit.feature_names if c not in features.columns]
    if missing:
        raise SystemExit(f"{table_id}: lineage replay did not produce {missing[:5]}")

    from tabular_agent.probe import probe_predict

    preds = probe_predict(fit, features[list(fit.feature_names)], "regression")
    return np.asarray(preds)


def main() -> None:
    val = load_table(os.path.join(RUN, "data", "val.parquet"))
    y = val[TARGET].to_numpy()
    order = val[ORDER_COLUMN].to_numpy()
    print(f"validation rows: {len(val)}   span: {order.min()} .. {order.max()}\n")

    with_history = _score_table("t9", val)
    without = _score_table("t8", val)

    for name, preds in (("t8 (no history)", without), ("t9 (+ history)", with_history)):
        print(f"{name:22s} rmspe {compute_metric(METRIC, y, preds):.5f}")

    rows = paired_difference(METRIC, y, with_history, without, seed=0)
    blocks = blockwise_difference(METRIC, y, with_history, without, order, n_blocks=5)

    print("\n--- paired bootstrap over rows (what a random split would report) ---")
    print(f"  gain (t9 over t8) {rows['gain']:+.5f}")
    print(f"  paired se         {rows['paired_se']:.5f}")
    print(f"  independent se    {rows['independent_se']:.5f}")
    print(f"  pairing ratio     {rows['pairing_ratio']}")
    print(f"  gain / se         {rows['gain'] / rows['paired_se']:.1f}")

    print("\n--- across consecutive blocks of the same rows (what ships) ---")
    print(f"  gain (pooled)     {blocks['gain']:+.5f}")
    print(f"  block gains       {[f'{g:+.5f}' for g in blocks['block_gains']]}")
    print(f"  block sd          {blocks['block_sd']:.5f}")
    print(f"  sign stable       {blocks['sign_stable']}")
    print(f"  blocks favouring t9  {blocks['blocks_favouring_a']}/{blocks['n_blocks']}")
    print(f"  gain / block sd   {blocks['gain'] / blocks['block_sd']:.1f}")

    # The validation half is 44 days, all of it adjacent to the freeze. If the
    # decay 0058 describes is real, it should be visible over a *longer* window —
    # so the same two candidates are scored again over validation plus the sealed
    # holdout, which reaches six weeks further out. The seal is spent for this
    # purpose either way (0045), and nothing here selects anything on it.
    holdout_path = os.path.join(RUN, "data", "holdout.parquet")
    if not os.path.exists(holdout_path):
        return
    print("\n--- the same comparison over a window twice as long ---")
    held = load_table(holdout_path)
    wide = pd.concat([val, held], ignore_index=True)
    y_wide = wide[TARGET].to_numpy()
    order_wide = wide[ORDER_COLUMN].to_numpy()
    a_wide = _score_table("t9", wide)
    b_wide = _score_table("t8", wide)
    long = blockwise_difference(METRIC, y_wide, a_wide, b_wide, order_wide, n_blocks=10)
    print(f"  span              {order_wide.min()} .. {order_wide.max()}")
    print(f"  gain (pooled)     {long['gain']:+.5f}")
    print(f"  block gains       {[f'{g:+.5f}' for g in long['block_gains']]}")
    print(f"  block sd          {long['block_sd']:.5f}")
    print(f"  sign stable       {long['sign_stable']}")
    print(f"  blocks favouring t9  {long['blocks_favouring_a']}/{long['n_blocks']}")


if __name__ == "__main__":
    main()
