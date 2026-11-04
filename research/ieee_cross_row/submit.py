"""Kaggle submission files for a run of this experiment, to read the private board.

    uv run python research/ieee_cross_row/submit.py r1

Writes three files under `_runs/ieee_cross_row/<tag>/`, each one submission and a
reading, never a choice (0075: the board is not for tuning against):

- `submission_<tag>_all_rows.csv` — the run's features, models and blend weights
  refitted on every labelled row, including our test block and the gap before it.
  Those days were scored once (`runner.py`) and have nothing left to judge.
- `submission_<tag>_shipped.csv` — the bundle as the product shipped it, which
  never saw the last 60 labelled days. Against `all_rows`: what staleness costs.
- `submission_lgbm_es_all_rows.csv` — `references.py`'s early-stopped LightGBM on
  the raw columns, fitted the same way on every labelled row. Against `all_rows`:
  what the run's features are worth on the private board.

No LLM calls; nothing is re-chosen.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from references import frames  # noqa: E402
from runner import OUT, TARGET  # noqa: E402
import prep_ieee as prep  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness.tools import _fit_recipe, _load_lineage  # noqa: E402
from tabular_agent.models import fit_final_model  # noqa: E402
from tabular_agent.pipeline import predict_bundle  # noqa: E402
from tabular_agent.refit import refit_lineage  # noqa: E402
from tabular_agent.taskspec import ResolvedSpec  # noqa: E402

ID = "TransactionID"


def all_rows_bundle(run: Run, train: pd.DataFrame) -> dict:
    """The shipped bundle's recipe refitted on every labelled row."""
    b = run.bundle._bundle
    j = run.journal
    split = j["split"]
    meta0 = json.loads((Path(run.run_dir) / "experiments" / b["fitted_on"]["refit_of"][0]
                        / "meta.json").read_text())
    spec = ResolvedSpec(
        target=TARGET, task_type=b["task_type"], metric=j["task"]["metric"],
        positive_class=None, notes=None, column_notes=None, n_splits=5,
        seed=split["seed"], target_transform=meta0.get("target_transform", "none"),
        objective=j["task"].get("objective"), forecast_horizon=split["forecast_horizon"],
    )
    rows = run.bundle._prepare(train.drop(columns=[TARGET]))
    rows[TARGET] = train[TARGET].to_numpy()
    work = pd.read_parquet(Path(run.run_dir) / "data" / "work.parquet")
    assert set(rows.columns) == set(work.columns), set(rows.columns) ^ set(work.columns)
    rows = rows[list(work.columns)]

    source = Path(run.run_dir) / "tables" / Path(b["training_table"]).name
    if not source.exists():
        source = Path(run.run_dir) / "data" / Path(b["training_table"]).name
    table, pipeline = refit_lineage(_load_lineage(str(source), spec.task_type), rows, spec,
                                    order=split["split_on"])
    models = []
    for exp_id in b["fitted_on"]["refit_of"]:
        meta = json.loads((Path(run.run_dir) / "experiments" / exp_id / "meta.json").read_text())
        # `_fit_recipe` reads only the run's seed off the session.
        r = _fit_recipe(SimpleNamespace(resolved=lambda: spec), meta)
        models.append(fit_final_model(
            table[r["feature_names"]], table[spec.target], spec.task_type, spec.metric,
            model_type=r["model_type"], params=r["params"] or None, seed=r["seed"],
            target_transform=r["target_transform"], reference_target=r["reference_target"],
            objective=spec.objective,
        ))
        print(f"  refitted {r['model_type']} on {len(table)} rows", flush=True)
    return {**b, "pipeline": pipeline, "models": models,
            "fitted_on": {"fitted_on": "all labelled rows", "n_rows": int(len(table))}}


def lgbm_es(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """`references.py`'s model: rounds stopped on the latest 15%, refitted on all rows."""
    import lightgbm as lgb

    X, Xt = frames(train, test.assign(**{TARGET: 0}))
    y = train[TARGET]
    order = np.argsort(train[prep.SPLIT_ON].to_numpy(), kind="stable")
    n_stop = int(round(0.15 * len(X)))
    tr, va = order[:-n_stop], order[-n_stop:]

    def make(n):
        return lgb.LGBMClassifier(n_estimators=n, learning_rate=0.03, random_state=0,
                                  verbose=-1)

    m = make(5000).fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
    print(f"  lgbm_es: {m.best_iteration_} rounds", flush=True)
    return make(int(m.best_iteration_)).fit(X, y).predict_proba(Xt)[:, 1]


def positive(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p[:, -1] if p.ndim == 2 else p


def main(tag: str = "r1") -> int:
    run = Run(json.loads((HERE / f"result_{tag}.json").read_text())["run_dir"])
    train = pd.read_parquet(prep.OUT)
    test = pd.read_parquet(prep.TEST_OUT)
    raw = test.drop(columns=[ID])
    dest = OUT / tag
    dest.mkdir(parents=True, exist_ok=True)

    def write(label, preds):
        path = dest / f"submission_{label}.csv"
        pd.DataFrame({ID: test[ID], TARGET: preds}).to_csv(path, index=False)
        print(f"{label}: {path} ({len(preds)} rows, mean {preds.mean():.4f})", flush=True)

    write(f"{tag}_shipped", positive(run.bundle.predict(raw)))
    refit = all_rows_bundle(run, train)
    write(f"{tag}_all_rows", positive(predict_bundle(refit, run.bundle._prepare(raw))))
    write("lgbm_es_all_rows", lgbm_es(train, test))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:2]))
