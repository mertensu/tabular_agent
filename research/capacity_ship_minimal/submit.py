"""Kaggle submission files for a forecasting run of this experiment.

    uv run python research/capacity_ship_minimal/submit.py restaurant r2
    uv run python research/capacity_ship_minimal/submit.py rossmann r1

Writes two files under `_runs/capacity_ship_minimal/<name>/`:

- `submission_<tag>_all_rows.csv` — **the one to submit.** The run's features and
  models, with the same blend weights, refitted on every row of the competition's
  training file, including the days this experiment held out as its own test set.
  Those days have been scored once (evaluate.py) and have nothing left to judge,
  so they may go into the final fit; the competition's test set never does.
- `submission_<tag>_shipped.csv` — the bundle as the product shipped it (fitted on
  training + validation, 0092). It never saw our test days, which sit between its
  training rows and the competition's test block; the gap between the two files'
  Kaggle scores is what that staleness costs.

No LLM calls; nothing is re-chosen.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import DATASETS, HERE, OUT  # noqa: E402

from tabular_agent.api import Run  # noqa: E402
from tabular_agent.harness.tools import _fit_recipe, _load_lineage  # noqa: E402
from tabular_agent.models import fit_final_model  # noqa: E402
from tabular_agent.pipeline import predict_bundle  # noqa: E402
from tabular_agent.refit import refit_lineage  # noqa: E402
from tabular_agent.taskspec import ResolvedSpec  # noqa: E402

ID = {"restaurant": "id", "rossmann": "Id"}


def all_rows_bundle(run: Run, cfg: dict) -> dict:
    """The shipped bundle's recipe refitted on every training row, audit included."""
    b = run.bundle._bundle
    j = run.journal
    split = j["split"]
    meta0 = json.loads((Path(run.run_dir) / "experiments" / b["fitted_on"]["refit_of"][0]
                        / "meta.json").read_text())
    spec = ResolvedSpec(
        target=cfg["target"], task_type=b["task_type"], metric=j["task"]["metric"],
        positive_class=None, notes=None, column_notes=None, n_splits=5,
        seed=split["seed"], target_transform=meta0.get("target_transform", "none"),
        objective=j["task"].get("objective"), forecast_horizon=split["forecast_horizon"],
    )
    train = pd.read_parquet(cfg["source"] / "train_joined.parquet")
    y = train[cfg["target"]].to_numpy()
    rows = run.bundle._prepare(train.drop(columns=[cfg["target"]]))
    rows[cfg["target"]] = y
    work = pd.read_parquet(Path(run.run_dir) / "data" / "work.parquet")
    assert set(rows.columns) == set(work.columns), (set(rows.columns) ^ set(work.columns))
    rows = rows[list(work.columns)]

    name = Path(b["training_table"]).name
    source = Path(run.run_dir) / "tables" / name
    if not source.exists():
        source = Path(run.run_dir) / "data" / name
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
    return {**b, "pipeline": pipeline, "models": models,
            "fitted_on": {"fitted_on": "all training rows", "n_rows": int(len(table))}}


def main(name: str, tag: str) -> int:
    cfg = DATASETS[name]
    result = json.loads((HERE / f"result_{name}_{tag}.json").read_text())
    run = Run(result["run_dir"])
    test = pd.read_parquet(cfg["source"] / "test_joined.parquet")
    raw = test.drop(columns=[ID[name]])
    dest = OUT / name

    shipped = np.asarray(run.bundle.predict(raw), dtype=float)
    refit = all_rows_bundle(run, cfg)
    all_rows = np.asarray(predict_bundle(refit, run.bundle._prepare(raw)), dtype=float)

    for label, preds in (("shipped", shipped), ("all_rows", all_rows)):
        preds = np.clip(preds, 0, None)
        path = dest / f"submission_{tag}_{label}.csv"
        pd.DataFrame({ID[name]: test[ID[name]], cfg["target"]: preds}).to_csv(path, index=False)
        print(f"{label}: {path} ({len(preds)} rows, mean {preds.mean():.3f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:3]))
