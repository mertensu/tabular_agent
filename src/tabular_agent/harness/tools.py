"""The agent's entire reachable surface (spec §3, §4, §12).

Every function here is a *tool*: it takes paths + typed parameters, writes
artifacts into ``run_dir/``, and returns a **compact JSON-able summary plus
paths** — never a DataFrame. Two properties follow from that contract:

* Context grows slowly, so a run can span dozens of cycles (§19.1).
* The state is on disk, so the run survives a crash or a context reset.

These are plain Python functions with no LLM dependency. That is deliberate:
they are unit-testable, and a human can drive the whole pipeline by hand before
handing the keys to a model. ``agent.py`` only wraps them.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

from ..cv import make_folds
from ..inspect import profile_dataframe
from ..io import load_table
from ..metrics import greater_is_better
from ..models import train_cv
from ..pipeline import Pipeline, save_bundle
from ..report import build_report, write_report
from . import journal
from .session import RunSession

#: Column-level detail is capped so a wide table cannot blow up the context.
MAX_REPORTED_COLUMNS = 60


def _feature_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Numeric feature matrix + the names of the columns that had to be dropped.

    v1 models are GBDTs fed a numeric matrix; raw strings would crash the fit.
    Dropping them (loudly) is the honest behaviour: the way to *use* a
    categorical column is `fit_encoding`, which learns its parameters per fold
    and persists them — not an ad-hoc cast that would silently skew train/serve.
    """
    features = df.drop(columns=[target])
    numeric = features.select_dtypes(include="number")
    dropped = [c for c in features.columns if c not in numeric.columns]
    return numeric, dropped


def inspect_data(session: RunSession, table: str | None = None) -> dict:
    """Profile a table and pin down the task type and selection metric.

    Always the first call of a run: it resolves the task spec (auto-detecting
    task type and metric from the target when the user left them open) and
    records it in the journal, so every later score is comparable.
    """
    path = session.resolve_table(table)
    df = load_table(path)
    spec = session.resolved()
    prof = profile_dataframe(df, target=spec.target)

    _, dropped = _feature_frame(df, spec.target)
    journal.set_task(
        session.run_dir, target=spec.target, task_type=spec.task_type, metric=spec.metric
    )

    columns = prof["columns"]
    truncated = len(columns) > MAX_REPORTED_COLUMNS
    shown = dict(list(columns.items())[:MAX_REPORTED_COLUMNS])

    return {
        "table": session.rel(path),
        "n_rows": prof["n_rows"],
        "n_cols": prof["n_cols"],
        "task_type": spec.task_type,
        "metric": spec.metric,
        "n_splits": spec.n_splits,
        "target": prof.get("target"),
        "columns": shown,
        "columns_truncated": truncated,
        "non_numeric_columns": dropped,
        "note": (
            "Non-numeric columns are not usable by the models as-is; encode them "
            "with fit_encoding to bring them in."
            if dropped
            else "All feature columns are numeric and usable as-is."
        ),
    }


def train_tabular_model(
    session: RunSession,
    table: str | None = None,
    model_type: str = "lgbm",
    params: dict | None = None,
) -> dict:
    """Cross-validated training — the only way to fit a model in this harness.

    Writes the fold models, OOF predictions and metrics under
    ``experiments/<id>/`` and appends a leaderboard row to the journal as a
    **side effect of training** (§19.2), so the experiment record can never
    drift from the experiments actually run.
    """
    path = session.resolve_table(table)
    spec = session.resolved()
    df = load_table(path)

    X, dropped = _feature_frame(df, spec.target)
    if X.shape[1] == 0:
        raise ValueError(
            "no numeric feature columns available to train on; use transform_features "
            "or fit_encoding to produce some first"
        )
    y = df[spec.target]

    folds = make_folds(y, spec.task_type, n_splits=spec.n_splits, seed=spec.seed)
    result = train_cv(
        X,
        y,
        folds,
        spec.task_type,
        spec.metric,
        model_type=model_type,
        params=params,
        seed=spec.seed,
    )

    cv_score = float(np.mean(result.fold_metrics))
    fold_spread = float(np.std(result.fold_metrics))

    exp_id = journal.next_experiment_id(session.run_dir)
    exp_dir = os.path.dirname(session.ctx.path("experiments", exp_id, "_"))
    joblib.dump(result.final_model, os.path.join(exp_dir, "final_model.joblib"))
    oof = pd.DataFrame(
        result.oof if result.oof.ndim == 2 else result.oof.reshape(-1, 1)
    ).add_prefix("oof_")
    oof.to_parquet(os.path.join(exp_dir, "oof.parquet"), index=False)
    meta = {
        "id": exp_id,
        "model_type": model_type,
        "params": params or {},
        "task_type": spec.task_type,
        "metric": spec.metric,
        "cv_score": cv_score,
        "fold_metrics": [float(m) for m in result.fold_metrics],
        "feature_names": list(result.feature_names),
        "source_table": session.rel(path),
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    baseline = (journal.load(session.run_dir).get("baseline") or {}).get("cv_score")
    row = {
        "id": exp_id,
        "model_type": model_type,
        "params": params or {},
        "features": list(result.feature_names),
        "metric": spec.metric,
        "cv_score": cv_score,
        "fold_spread": fold_spread,
        "vs_baseline": None if baseline is None else round(cv_score - baseline, 6),
        "artifact_path": session.rel(exp_dir),
    }
    journal.append_experiment(session.run_dir, row)

    return {
        "experiment_id": exp_id,
        "model_type": model_type,
        "metric": spec.metric,
        "cv_score": cv_score,
        "fold_spread": fold_spread,
        "fold_metrics": meta["fold_metrics"],
        "n_features": len(result.feature_names),
        "dropped_non_numeric": dropped,
        "vs_baseline": row["vs_baseline"],
        "higher_is_better": greater_is_better(spec.metric),
        "artifact_path": row["artifact_path"],
    }


def finalize_run(session: RunSession) -> dict:
    """Package the best experiment: deployable bundle + human-readable report.

    Called **once**, at the end (§19.1). Reads the journal's ``best`` pointer
    rather than re-deriving anything, and auto-trains the trivial baseline if the
    run never did, so the report can always answer "did this beat doing nothing".
    """
    j = journal.load(session.run_dir)
    if not j["leaderboard"]:
        raise ValueError("nothing to finalize: no model has been trained yet")

    if j.get("baseline") is None:
        train_tabular_model(session, model_type="baseline")
        j = journal.load(session.run_dir)

    best = j["best"]
    exp_dir = os.path.join(session.run_dir, best["artifact_path"])
    with open(os.path.join(exp_dir, "meta.json")) as f:
        meta = json.load(f)
    model = joblib.load(os.path.join(exp_dir, "final_model.joblib"))

    pipeline = Pipeline(meta["task_type"])
    bundle_dir = save_bundle(
        session.ctx,
        pipeline,
        [model],
        np.array([1.0]),
        meta["feature_names"],
        meta["task_type"],
    )

    text = build_report(
        task_type=meta["task_type"],
        metric=meta["metric"],
        selection_score=meta["cv_score"],
        fold_metrics=meta["fold_metrics"],
        baseline_score=j["baseline"]["cv_score"],
        model_types=[meta["model_type"]],
        feature_names=meta["feature_names"],
    )
    report_path = write_report(session.ctx, text)

    return {
        "best_experiment_id": best["experiment_id"],
        "model_type": meta["model_type"],
        "metric": meta["metric"],
        "cv_score": meta["cv_score"],
        "baseline_score": j["baseline"]["cv_score"],
        "n_experiments": len(j["leaderboard"]),
        "bundle_dir": session.rel(bundle_dir),
        "report_path": session.rel(report_path),
    }
