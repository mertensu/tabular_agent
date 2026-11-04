"""Do k1's rolling target means go stale past the last label? No LLM calls.

    uv run python research/ieee_cross_row/staleness.py coverage
    uv run python research/ieee_cross_row/staleness.py effect
    uv run python research/ieee_cross_row/staleness.py effect r1
    uv run python research/ieee_cross_row/staleness.py effect r1 k1_split

`coverage`: the shipped k1 bundle's rolling columns on Kaggle's test rows, by distance
from the last labelled row, beside the same columns on the labelled rows.

`effect`: k1's table refitted on the first 90 labelled days only, scored on the later
labelled rows by distance from day 90, with and without the rolling columns, under
one early-stopped LightGBM. The distance Kaggle's private rows sit at cannot be
reached inside the labelled file. With a tag, the same for that run's table instead,
to compare tables on one cut; with `k1_split`, on k1's own training and validation
rows, to see what its validation would have said. Writes `staleness.json`.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runner import DAY, OUT, TARGET, TEST_DAYS  # noqa: E402
import prep_ieee as prep  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness.tools import _load_lineage  # noqa: E402
from tabular_agent.metrics import compute_metric  # noqa: E402
from tabular_agent.pipeline import feature_table_bundle  # noqa: E402
from tabular_agent.probe import fit_fitted_ruler, probe_predict  # noqa: E402
from tabular_agent.refit import refit_lineage  # noqa: E402
from tabular_agent.taskspec import ResolvedSpec  # noqa: E402

TAG = "k1"


def k1(tag: str = TAG) -> Run:
    return Run(sorted((OUT / tag / "runs").glob("*"))[-1])


def rolling(columns) -> list[str]:
    return [c for c in columns if f"_{TARGET}_" in c]


def coverage(run: Run) -> dict:
    b = run.bundle._bundle
    cols = rolling(b["feature_names"])
    means = [c for c in cols if c.endswith("_mean")]
    train = pd.read_parquet(prep.OUT)
    test = pd.read_parquet(prep.TEST_OUT)
    last = train[prep.SPLIT_ON].max()

    def row(X, distance, edges):
        out = {}
        for lo, hi in edges:
            at = (distance >= lo) & (distance < hi)
            if at.any():
                out[f"{lo}-{hi}"] = {
                    "rows": int(at.sum()),
                    "missing": {c.split("_")[-2]: round(float(X.loc[at, c].isna().mean()), 3)
                                for c in means},
                }
        return out

    X = feature_table_bundle(b, run.bundle._prepare(test.drop(columns=["TransactionID"])))
    days = ((test[prep.SPLIT_ON] - last) / DAY).to_numpy()
    edges = [(30, 60), (60, 90), (90, 120), (120, 150), (150, 180), (180, 214)]
    table = pd.read_parquet(Path(run.run_dir) / "refit" / Path(b["training_table"]).name)
    labelled = {c.split("_")[-2]: round(float(table[c].isna().mean()), 3) for c in means}
    return {"windows_in_seconds_missing_on_all_labelled_rows": labelled,
            "kaggle_test_by_days_past_last_label": row(X, days, edges)}


def effect(run: Run, on_k1_split: bool = False) -> dict:
    b = run.bundle._bundle
    j = run.journal
    split = j["split"]
    spec = ResolvedSpec(
        target=TARGET, task_type="binary", metric="auc", positive_class=None, notes=None,
        column_notes=None, n_splits=5, seed=split["seed"], target_transform="none",
        objective=None, forecast_horizon=split["forecast_horizon"],
    )
    df = pd.read_parquet(prep.OUT)
    start = df[prep.SPLIT_ON].min()
    cut = start + 90 * DAY
    train = df[df[prep.SPLIT_ON] < cut].reset_index(drop=True)
    later = df[df[prep.SPLIT_ON] >= cut + prep.FORECAST_HORIZON].reset_index(drop=True)
    edges = [(30, 50), (50, 70), (70, 92)]
    if on_k1_split:
        first_val = pd.read_parquet(
            Path(k1().run_dir) / "data" / "val.parquet", columns=[prep.SPLIT_ON])[prep.SPLIT_ON].min()
        cut = first_val
        train = df[df[prep.SPLIT_ON] < first_val].reset_index(drop=True)
        later = df[df[prep.SPLIT_ON] >= first_val].reset_index(drop=True)
        edges = [(0, 30), (30, 62)]

    def prepared(frame):
        rows = run.bundle._prepare(frame.drop(columns=[TARGET]))
        rows[TARGET] = frame[TARGET].to_numpy()
        return rows

    work = pd.read_parquet(Path(run.run_dir) / "data" / "work.parquet")
    rows = prepared(train)[list(work.columns)]
    source = Path(run.run_dir) / "tables" / Path(b["training_table"]).name
    table, pipeline = refit_lineage(_load_lineage(str(source), "binary"), rows, spec,
                                    order=split["split_on"])
    recipe = {**b, "pipeline": pipeline}
    Xl = feature_table_bundle(recipe, run.bundle._prepare(later.drop(columns=[TARGET])))
    features = list(b["feature_names"])
    without = [c for c in features if c not in rolling(features)]
    days = ((later[prep.SPLIT_ON] - cut) / DAY).to_numpy()
    y = later[TARGET].to_numpy()

    out = {"cut_day": 90, "n_train": len(table), "arms": {}}
    arms = (("with_rolling", features), ("without_rolling", without)) if without != features \
        else (("table", features),)
    for name, cols in arms:
        fit = fit_fitted_ruler(table[cols], table[TARGET], "binary", "auc", "none",
                               order=table[prep.SPLIT_ON])
        p = np.asarray(probe_predict(fit, Xl[cols], "binary"), dtype=float)
        out["arms"][name] = {
            "rounds": fit.spec["rounds"],
            "by_days_past_cut": {
                f"{lo}-{hi}": round(float(compute_metric(
                    "auc", y[(days >= lo) & (days < hi)], p[(days >= lo) & (days < hi)])), 4)
                for lo, hi in edges},
            "all": round(float(compute_metric("auc", y, p)), 4),
        }
        print(name, json.dumps(out["arms"][name]), flush=True)
    means = [c for c in rolling(features) if c.endswith("_mean")]
    out["missing_by_days_past_cut"] = {
        f"{lo}-{hi}": {c.split("_")[-2]: round(float(
            Xl.loc[(days >= lo) & (days < hi), c].isna().mean()), 3) for c in means}
        for lo, hi in edges}
    return out


def main(which: str, tag: str = TAG, split: str | None = None) -> int:
    run = k1(tag)
    path = HERE / "staleness.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    key = "_".join(x for x in (which, None if tag == TAG else tag, split) if x)
    done[key] = coverage(run) if which == "coverage" else effect(run, split == "k1_split")
    path.write_text(json.dumps(done, indent=2) + "\n")
    print(json.dumps(done[key], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
