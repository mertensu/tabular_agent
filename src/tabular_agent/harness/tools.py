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
from ..encoders import fit_encoding as _fit_encoding
from ..inspect import profile_dataframe
from ..io import load_table, save_parquet
from ..metrics import greater_is_better
from ..models import train_cv
from ..pipeline import Pipeline, save_bundle
from ..report import build_report, write_report
from ..sandbox import SandboxError
from ..transform import transform_features as _transform_features
from . import journal
from .session import RunSession

#: Column-level detail is capped so a wide table cannot blow up the context.
MAX_REPORTED_COLUMNS = 60


def _lineage_path(table_path: str) -> str:
    """Sidecar recording the pipeline steps that produced a derived table.

    A feature table on disk is only half the artifact: the other half is *how to
    reproduce its columns from raw input at predict time*. Keeping that next to
    the table (rather than in a single run-wide pipeline) means a run can branch
    into several feature tables and each still knows its own provenance, so
    ``finalize_run`` can package whichever one the winning model was trained on.
    """
    return os.path.splitext(table_path)[0] + ".pipeline.json"


def _load_lineage(table_path: str, task_type: str) -> Pipeline:
    """The steps behind ``table_path``; empty for the raw input table."""
    path = _lineage_path(table_path)
    if not os.path.exists(path):
        return Pipeline(task_type)
    with open(path) as f:
        return Pipeline.from_dict(json.load(f))


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
            "with fit_encoding to bring them in — method='ordinal' (stating the "
            "level order) when the categories are genuinely ordered, "
            "method='frequency' or 'target' when they are nominal."
            if dropped
            else "All feature columns are numeric and usable as-is."
        ),
    }


def fit_encoding(
    session: RunSession,
    columns: list[str],
    method: str = "frequency",
    table: str | None = None,
    categories: dict[str, list] | None = None,
    smoothing: float = 10.0,
) -> dict:
    """Encode categorical columns into a new feature table (spec §5.2).

    The only way to make a non-numeric column usable: ``train_tabular_model``
    drops what it cannot feed to a GBDT, and there is no tool that casts a
    column in place. Writes ``tables/<id>.parquet`` (the source table plus the
    encoded columns) and, beside it, the lineage sidecar recording the fitted
    encoder — so whichever table the winning model was trained on carries the
    parameters needed to reproduce its columns at predict time.

    ``method="target"`` is the leaky one, and is why this is a tool rather than
    a free-code transform: the table gets **out-of-fold** values (honest to
    train on) while the sidecar stores the **all-rows** encoder (correct at
    predict). ``method="frequency"`` and ``method="ordinal"`` never look at the
    target; ``ordinal`` needs ``categories`` — the level order per column, which
    is never inferred.
    """
    path = session.resolve_table(table)
    spec = session.resolved()
    df = load_table(path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in {session.rel(path)}: {missing}")
    if spec.target in columns:
        raise ValueError(f"refusing to encode the target column {spec.target!r} as a feature")

    kwargs = {"smoothing": smoothing, "categories": categories}
    if method == "target":
        y = df[spec.target]
        # Same seed and n_splits as training, so the out-of-fold values in the
        # table were held out of exactly the folds the model is scored on.
        kwargs["y"] = y
        kwargs["folds"] = make_folds(y, spec.task_type, n_splits=spec.n_splits, seed=spec.seed)
    train_features, encoder = _fit_encoding(df, columns, method, **kwargs)

    out = df.join(train_features)
    table_id = journal.next_table_id(session.run_dir)
    out_path = session.ctx.path("tables", f"{table_id}.parquet")
    save_parquet(out, out_path)

    lineage = _load_lineage(path, spec.task_type)
    lineage.add_encoder(encoder)
    with open(_lineage_path(out_path), "w") as f:
        json.dump(lineage.to_dict(), f, indent=2)

    new_columns = list(train_features.columns)
    row = {
        "id": table_id,
        "path": session.rel(out_path),
        "source_table": session.rel(path),
        "method": method,
        "encoded_columns": list(columns),
        "new_columns": new_columns,
        "n_steps": len(lineage.steps),
    }
    journal.append_table(session.run_dir, row)

    _, still_dropped = _feature_frame(out, spec.target)
    return {
        "table": row["path"],
        "method": method,
        "new_columns": new_columns,
        "n_rows": int(len(out)),
        "source_table": row["source_table"],
        "non_numeric_columns": still_dropped,
        "note": (
            "Pass this table to train_tabular_model to use the new columns. "
            + (
                "The values stored here are out-of-fold; the encoder fitted on all "
                "rows is kept in the table's lineage and is what runs at predict time."
                if method == "target"
                else "The fitted encoder is kept in the table's lineage and replays at "
                "predict time."
            )
        ),
    }


def transform_features(
    session: RunSession,
    code: str,
    table: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Run agent-written ``df -> df`` code to derive new columns (spec §5.1).

    The escape hatch: everything the structured tools cannot express — ratios,
    distances, date parts, per-row counts — while staying leakage-safe by
    construction. The sandbox blocks the model and sklearn libraries, so this
    code *cannot* learn a parameter from the data; anything stateful has to go
    through ``fit_encoding``, which fits per fold.

    **The target column is not passed to the code.** Two reasons, and the second
    is the one that would actually bite: a derived column cannot be a function of
    the target if the target was never in the frame; and at predict time there is
    no target column, so code that referenced it would either crash or produce
    different features than it did at training. Same frame in both places or the
    replay is a lie.

    Writes ``tables/<id>.parquet`` (transformed features + the untouched target)
    and the lineage sidecar, which stores **the code, not the columns** — the
    step is re-run at predict, so a stored feature can never drift from the code
    that made it.
    """
    path = session.resolve_table(table)
    spec = session.resolved()
    df = load_table(path)

    features = df.drop(columns=[spec.target])
    try:
        out_features, summary = _transform_features(features, code, timeout=timeout)
    except SandboxError as e:
        # Reaching for the target raises a bare `KeyError: 'y'` from inside the
        # child — true, but the model will read it as a typo and try again. Say
        # what actually happened, once, so it stops reaching.
        if spec.target in str(e):
            raise SandboxError(
                f"{e}\n\nNote: the target column {spec.target!r} is deliberately not in the "
                "frame your code receives — it is not available at predict time either. "
                "Derive features from the other columns; anything that has to learn from "
                "the target belongs in fit_encoding(method='target')."
            ) from e
        raise
    if spec.target in out_features.columns:
        raise ValueError(
            f"the transform produced a column named {spec.target!r}, which is the target; "
            "name derived columns something else"
        )

    out = out_features.join(df[spec.target])
    table_id = journal.next_table_id(session.run_dir)
    out_path = session.ctx.path("tables", f"{table_id}.parquet")
    save_parquet(out, out_path)

    lineage = _load_lineage(path, spec.task_type)
    lineage.add_transform(code)
    with open(_lineage_path(out_path), "w") as f:
        json.dump(lineage.to_dict(), f, indent=2)

    row = {
        "id": table_id,
        "path": session.rel(out_path),
        "source_table": session.rel(path),
        "method": "transform",
        # The code is the only record of *what was tried*: an agent resuming
        # after a context reset can see the idea, not just the column names.
        "code": code,
        "new_columns": list(summary["new_columns"]),
        "n_steps": len(lineage.steps),
    }
    journal.append_table(session.run_dir, row)

    _, still_dropped = _feature_frame(out, spec.target)
    return {
        "table": row["path"],
        "new_columns": row["new_columns"],
        "n_rows": int(summary["n_rows"]),
        "n_cols": len(out.columns),
        "source_table": row["source_table"],
        "non_numeric_columns": still_dropped,
        "note": (
            "Pass this table to train_tabular_model to use the new columns. The code "
            "is stored in the table's lineage and re-runs at predict time; the target "
            "column was not visible to it."
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
        # Which feature table this was trained on, so a resuming agent can read
        # the leaderboard alone and know what to reuse or beat — without opening
        # each experiment's meta.json.
        "source_table": session.rel(path),
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
        "source_table": row["source_table"],
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

    # The bundle must be able to rebuild the winning model's features from *raw*
    # input, so it inherits the lineage of whichever table that model was trained
    # on — empty when it trained on the raw table, one step per encoding otherwise.
    pipeline = _load_lineage(session.resolve_table(meta["source_table"]), meta["task_type"])
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
