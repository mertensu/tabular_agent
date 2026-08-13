"""Fit one model type on one of a finished run's tables, and write a submission.

The sibling of `rebuild_submission.py`. That script packages an experiment the
run *did* produce; this one packages a model the run **could** have produced and
did not — a model type on a table, fitted the way the terminal phase would fit
it, without needing the run to be re-driven.

It exists because of a defect and the measurement that priced it. Porto Seguro v3
ranked its tables with the probe (`lgbm`), chose `t3` at gini 0.29066, and then
shipped elasticnet at 0.25389 — because the ruler was not among the terminal
model types. The fix is one line; the *cost* of having been wrong is only
knowable by fitting the model the run should have shipped and scoring it against
ground truth.

    uv run python scripts/fit_table_and_submit.py _runs/seguro_v3 t3 \\
        --model lgbm --test seguro_driver_prediction/test.csv \\
        --id id --target target

**A diagnostic, not part of a run.** It reads a finished run directory and writes
a CSV; it never touches the journal, and nothing in `src/` depends on it. The
model is fitted on the table's rows exactly as the terminal phase would — same
seed, same target scale as the run resolved — and its validation score is printed
so the number can be compared with what the run recorded before anything is
submitted anywhere.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from tabular_agent.harness import journal
from tabular_agent.io import load_table
from tabular_agent.metrics import compute_metric, greater_is_better
from tabular_agent.models import _make_model, predict_model
from tabular_agent.pipeline import Pipeline
from tabular_agent.probe import (
    PROBE_MODEL_TYPE,
    PROBE_PARAMS,
    PROBE_SEED,
    default_target_transform,
)
from tabular_agent.target import apply_target, invert_target


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("table", help="a table id in the run, e.g. t3")
    p.add_argument("--model", default=PROBE_MODEL_TYPE)
    p.add_argument("--params", default=None, help="JSON; defaults to the probe's")
    p.add_argument("--test", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", default=None)
    #: The probe's seed, not the run's. On Porto Seguro's 476k rows lgbm moves
    #: 0.003 of auc between seeds at identical parameters — more than a third of
    #: the gain the whole feature search produced — so reproducing the run's
    #: ranking number means reproducing its seed as well as its model.
    p.add_argument("--seed", type=int, default=PROBE_SEED)
    a = p.parse_args()

    j = journal.load(a.run_dir)
    task = j["task"]
    metric, task_type = task["metric"], task["task_type"]
    # The run's own scale, so this is the procedure the terminal phase applies
    # rather than a variant of it.
    tt = default_target_transform(metric, task_type)
    params = json.loads(a.params) if a.params else dict(PROBE_PARAMS)

    table_path = os.path.join(a.run_dir, "tables", f"{a.table}.parquet")
    lineage = json.load(open(table_path.replace(".parquet", ".pipeline.json")))
    tbl = load_table(table_path)
    X = tbl.drop(columns=[a.target]).select_dtypes(include="number")
    y = pd.Series(tbl[a.target].to_numpy())
    y_fit = pd.Series(apply_target(tt, y)) if tt != "none" else y

    print(f"{a.table}: {len(X):,} rows x {X.shape[1]} features | {a.model} {params} "
          f"| seed {a.seed} | target scale {tt}")

    model = _make_model(a.model, task_type, params, a.seed, metric)
    model.fit(X, y_fit)

    # The same exam the run sat: replay the lineage over raw validation rows.
    val = load_table(os.path.join(a.run_dir, "data", "val.parquet"))
    produced = Pipeline.from_dict(lineage).transform(val.drop(columns=[a.target]))
    missing = [c for c in X.columns if c not in produced.columns]
    if missing:
        raise SystemExit(f"the lineage does not reproduce {missing[:5]} on raw rows")
    preds_val = predict_model(model, produced[list(X.columns)], task_type)
    if tt != "none":
        preds_val = invert_target(tt, preds_val)
    score = compute_metric(metric, val[a.target].to_numpy(), preds_val)

    recorded = next((t.get("probe_score") for t in j.get("tables") or []
                     if t.get("id") == a.table), None)
    print(f"  val {metric} {score:.5f}")
    if recorded is not None:
        if abs(score - recorded) < 1e-9:
            verdict = "identical — this is the probe's own fit"
        else:
            verdict = "better" if (score > recorded) == greater_is_better(metric) else "worse"
        print(f"  the run's probe recorded {recorded:.5f} on this table ({verdict})")

    test = pd.read_csv(a.test)
    ids = test[a.id]
    # The id column is dropped only if the model was not trained with it. Some
    # runs keep it as a feature; that is their business and the replay has to
    # match what was fitted.
    raw = test if a.id in X.columns else test.drop(columns=[a.id])
    features = Pipeline.from_dict(lineage).transform(raw)
    preds = predict_model(model, features[list(X.columns)], task_type)
    if tt != "none":
        preds = invert_target(tt, preds)

    bad = int((~np.isfinite(preds)).sum())
    if bad:
        print(f"  !! {bad} non-finite predictions")

    out = a.out or os.path.join(a.run_dir, f"submission_{a.table}_{a.model}.csv")
    pd.DataFrame({a.id: ids, a.target: preds}).to_csv(out, index=False)
    print(f"  wrote {out}  ({len(preds)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
