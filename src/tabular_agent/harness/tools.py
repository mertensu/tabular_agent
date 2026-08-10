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
from ..metrics import compute_metric, greater_is_better
from ..models import predict_model, train_cv
from ..pipeline import Pipeline, save_bundle
from ..report import build_report, write_report
from ..sandbox import SandboxError, run_readonly
from ..transform import transform_features as _transform_features
from . import journal, stop
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


def _score_on_val(
    session: RunSession,
    source_path: str,
    model,
    feature_names,
    spec,
    y_work: pd.Series,
) -> tuple[float | None, str | None]:
    """Score a just-fitted model on the validation half. ``(score, why_not)``.

    The validation rows are **raw**: no tool ever wrote them, so the columns the
    model expects do not exist on them yet and have to be rebuilt by replaying
    the lineage of the table it trained on — the same pipeline ``finalize_run``
    packages into a bundle. Encoders replay their fitted parameters, so nothing
    learned from validation rows enters the features.

    Returns a reason instead of raising when the replay cannot produce the
    model's columns: a leaderboard row with a missing validation score is worth
    more than a training call that dies.
    """
    val = load_table(session.val_path)
    y_val = val[spec.target]
    pipeline = _load_lineage(source_path, spec.task_type)
    # The target is dropped before the pipeline sees the rows (0009): a transform
    # that can read the target is a transform that can leak it.
    features = pipeline.transform(val.drop(columns=[spec.target]))

    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        return None, f"replaying the table's lineage did not produce {missing[:5]}"

    classes = None
    if spec.task_type in ("binary", "multiclass"):
        # Union of both halves: a class present in work but absent from
        # validation (or the reverse) still has to occupy its column, or the
        # probability matrix silently shifts and logloss scores the wrong class.
        classes = sorted(set(pd.unique(y_work)) | set(pd.unique(y_val)))
    preds = predict_model(
        model,
        features[list(feature_names)],
        spec.task_type,
        classes=classes if spec.task_type == "multiclass" else None,
    )
    return float(compute_metric(spec.metric, y_val, preds, labels=classes)), None


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
    journal.set_split(session.run_dir, session.split_record())

    columns = prof["columns"]
    truncated = len(columns) > MAX_REPORTED_COLUMNS
    shown = dict(list(columns.items())[:MAX_REPORTED_COLUMNS])

    return {
        "table": session.rel(path),
        "n_rows": prof["n_rows"],
        "n_cols": prof["n_cols"],
        "task_type": spec.task_type,
        "metric": spec.metric,
        # Fixed for the whole run, so it is stated here rather than repeated on
        # every training result. Every `vs_...` field is already signed as a gain;
        # this is the direction of `cv_score` itself.
        "higher_is_better": greater_is_better(spec.metric),
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
    construction. Two guards keep it stateless: the sandbox blocks the model and
    sklearn libraries, and every produced column is re-run over a subset and
    rejected if its values depend on the rest of the batch (``pd.qcut``, batch
    normalisation — pure pandas, and the reason the import block alone is not
    enough). Anything genuinely stateful goes through ``fit_encoding``.

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


def _columns_by_table(
    session: RunSession, j: dict, source_table: str, feature_names: list[str]
) -> list[dict]:
    """The table's columns, grouped by which table in its history added them.

    One tool call writes one table, so grouping by table *is* grouping by tool
    call — which is the unit the agent can act on — without needing a second word
    for it.

    Restricted to columns that actually reached the model: a table whose columns
    were all dropped as non-numeric contributed nothing to this fit and is left
    out. Whatever no derived table claims belongs to the raw input, reported
    first under the name the agent would use for it.
    """
    usable = set(feature_names)
    claimed = set()
    out = []
    for row in journal.columns_by_table(j, source_table):
        cols = [c for c in row["columns"] if c in usable]
        claimed.update(cols)
        if cols:
            out.append({"table": row["table"], "method": row["method"], "columns": cols})
    original = [c for c in feature_names if c not in claimed]
    if original:
        out.insert(
            0,
            {
                "table": _table_label(session, session.rel(session.data_path)),
                "method": None,
                "columns": original,
            },
        )
    return out


def _gain(cv_score: float, reference: float, metric: str) -> float:
    """How much better than ``reference`` this score is. Positive is always better.

    Every ``vs_*`` field the agent sees goes through here. The raw difference
    would flip meaning with the metric — an improvement is negative under `rmse`
    and positive under `auc` — and two such fields sitting side by side in one
    result, signed differently, is a reading the model gets wrong exactly once
    before it discards a good branch. One direction, no exceptions.
    """
    delta = cv_score - reference
    # `+ 0.0` normalises IEEE negative zero: a change of nothing rounds
    # to -0.0, which serialises as "-0.0" and reads as a regression.
    return round(delta if greater_is_better(metric) else -delta, 6) + 0.0


def _fit_signature(model_type: str, params: dict | None) -> str:
    """Identity of *how* a model was fitted, for deciding what compares to what."""
    return json.dumps([model_type, params or {}], sort_keys=True)


def _table_label(session: RunSession, rel_path: str) -> str:
    """How to *name* a table in a sentence aimed at the agent.

    The raw input lives outside ``run_dir``, so its "relative" path is the
    absolute one — right for matching journal rows, unreadable in prose, and not
    what the agent would type anyway: it names that table by passing nothing.
    """
    return (
        "the work table" if rel_path == session.rel(session.data_path) else rel_path
    )


def _vs_source_table(
    session: RunSession,
    j: dict,
    source_table: str,
    val_score: float,
    metric: str,
    model_type: str,
    params: dict | None,
) -> tuple[float | None, str]:
    """Gain over the same model, same params, on this table's ``source_table``.

    Returns the gain and a plain statement of what it is a gain *over* — or, when
    there is nothing valid to compare against, why not and what would fix it.

    Restricted to an identical fit on purpose. This number exists to answer "did
    that feature call pay", and the only way it can answer that is if the feature
    call is the sole difference between the two runs. Comparing against the source
    table's best score whatever fitted it silently folds a model swap into the
    features' credit: train lgbm on the source, catboost on what was built from
    it, and a catboost-shaped gain gets attributed to the columns you just added.

    ``None`` when no such fit exists on the source table — which is the point: the
    marginal number is the reward for having measured the previous step the same
    way, so holding the model fixed while iterating on features is worth
    something rather than being a rule imposed on the agent. When the agent does
    want to change model and features at once, `explain_table` still separates the
    contributions, because it refits everything itself.
    """
    sources = {t["path"]: t.get("source_table") for t in j.get("tables") or []}
    source = sources.get(source_table)
    if source is None:
        return None, f"{_table_label(session, source_table)} was not derived from another table"

    label = _table_label(session, source)
    signature = _fit_signature(model_type, params)
    on_source = [
        r
        for r in j.get("leaderboard") or []
        if r.get("source_table") == source and r.get("val_score") is not None
    ]
    comparable = [
        r for r in on_source if _fit_signature(r.get("model_type"), r.get("params")) == signature
    ]
    if not comparable:
        detail = (
            f"{label} has been trained, but not with this exact fit"
            if on_source
            else f"{label} has never been trained"
        )
        return None, (
            f"no gain available: {detail}. Train {model_type} with these params on "
            f"{label} to get one, or call explain_table on "
            f"{_table_label(session, source_table)} to measure each table's columns "
            "directly."
        )

    pick = max if greater_is_better(metric) else min
    best = pick(comparable, key=lambda r: r["val_score"])
    return _gain(val_score, best["val_score"], metric), (
        f"vs {best['id']}: the same {model_type} fit on {label}, so the only "
        "difference is the columns added since"
    )


def _progress(session: RunSession) -> dict:
    """This run's stop status, from the leaderboard as it currently stands."""
    j = journal.load(session.run_dir)
    metric = (j.get("task") or {}).get("metric") or session.resolved().metric
    return stop.evaluate(j["leaderboard"], metric, session.stop_policy)


def _refuse_if_table_exhausted(session: RunSession, path: str) -> dict | None:
    """``None`` if this table may still be trained on, otherwise the redirect.

    Distinct from :func:`_refuse_if_finished` in what it costs the agent: nothing.
    The run is not over, and the next call on a *new* table goes straight
    through. That asymmetry is the point — a table is cheap to give up on, a run
    is not.
    """
    rel = session.rel(path)
    label = _table_label(session, rel)
    j = journal.load(session.run_dir)
    metric = (j.get("task") or {}).get("metric") or session.resolved().metric
    status = stop.table_status(j["leaderboard"], metric, session.stop_policy, rel, label)
    if not status["exhausted"]:
        return None

    # A model that keeps asking for the same refused table is not being
    # redirected, it is stuck. Escalate it into the run-level path, which knows
    # how to end things with a bundle on disk.
    repeats = journal.record_table_refusal(session.run_dir, rel)
    if repeats >= session.stop_policy.max_refusals:
        return _refuse_if_finished(session, force=True)

    return {
        "trained": False,
        "table": label,
        "table_exhausted": True,
        "reason": status["reason"],
        "next": "transform_features or fit_encoding, then train on the new table",
        "note": stop.advisory(_progress(session), status),
    }


def _refuse_if_finished(session: RunSession, force: bool = False) -> dict | None:
    """``None`` if training may proceed, otherwise the refusal to hand back.

    The escalation matters. A first refusal explains itself and points at
    ``finalize_run``, which is all a cooperative model needs. A model that keeps
    asking gets the run packaged **for** it — because the alternative is a loop
    that ends with no bundle, and the bundle is the entire deliverable. Only once
    that is safely on disk does the harness raise and end things outright.
    """
    status = _progress(session)
    # Checked before the progress gate, not after. A finalized run is over
    # whatever the stop rule thinks: the bundle is packaged and the winner picked,
    # so another fit could only add a leaderboard row nothing will ever read, or
    # tempt a second `finalize_run` against the same validation rows.
    if journal.stop_state(session.run_dir).get("finalized"):
        raise RuntimeError(
            "this run is finished: the harness has already packaged it and will not "
            "train again"
        )
    if not status["should_stop"] and not force:
        return None
    if force and status["reason"] is None:
        status = dict(status, reason="the same exhausted table was requested repeatedly")

    refusals = journal.record_refusal(session.run_dir)
    policy = session.stop_policy
    finalized = journal.stop_state(session.run_dir).get("finalized")

    if refusals >= policy.max_refusals and not finalized:
        final = finalize_run(session)
        return {
            "trained": False,
            "reason": status["reason"],
            "harness_finalized": True,
            "note": (
                "Asked to train after the run was over, so the harness packaged the "
                "run itself. Nothing further will be trained. Report what you found "
                "and stop."
            ),
            **final,
        }
    if finalized:
        raise RuntimeError(
            "this run is finished: the harness has already packaged it and will not "
            f"train again ({status['reason']})"
        )

    return {
        "trained": False,
        "reason": status["reason"],
        "n_experiments": status["n_experiments"],
        "since_improvement": status["since_improvement"],
        "next": "finalize_run",
        "note": stop.advisory(status),
    }


def train_tabular_model(
    session: RunSession,
    table: str | None = None,
    model_type: str = "lgbm",
    params: dict | None = None,
    enforce_stop: bool = True,
) -> dict:
    """Cross-validated training — the only way to fit a model in this harness.

    Writes the fold models, OOF predictions and metrics under
    ``experiments/<id>/`` and appends a leaderboard row to the journal as a
    **side effect of training** (§19.2), so the experiment record can never
    drift from the experiments actually run.

    Refuses to train once the run has stopped making progress (§19.3). The
    refusal is a normal return value, not an exception, for a practical reason:
    an exception would burn the tool's retry budget, and being asked to train one
    more time is not a malformed argument — it is a model that has not yet read
    the news.

    ``enforce_stop=False`` is the harness training on its own behalf — the
    baseline `finalize_run` fills in when a run never trained one. Refusing that
    would leave the report unable to answer "did this beat doing nothing", which
    is the one question it must always answer. It is not part of the agent-facing
    signature.
    """
    if enforce_stop:
        refusal = _refuse_if_finished(session)
        if refusal is not None:
            return refusal

    path = session.resolve_table(table)
    if enforce_stop:
        refusal = _refuse_if_table_exhausted(session, path)
        if refusal is not None:
            return refusal

    spec = session.resolved()
    journal.set_split(session.run_dir, session.split_record())
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
    val_score, val_unavailable = _score_on_val(
        session, path, result.final_model, result.feature_names, spec, y
    )

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
        "val_score": val_score,
        "fold_metrics": [float(m) for m in result.fold_metrics],
        "feature_names": list(result.feature_names),
        "target": spec.target,
        "source_table": session.rel(path),
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Everything comparative is measured on the validation half, because that is
    # what the leaderboard ranks by. `cv_score` rides along on the row so the
    # cv-vs-val gap stays readable — it is the run's own record of how much the
    # search has been tuning to the folds.
    baseline = (journal.load(session.run_dir).get("baseline") or {}).get("val_score")
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
        "val_score": val_score,
        "val_unavailable": val_unavailable,
        "cv_score": cv_score,
        "fold_spread": fold_spread,
        "vs_baseline": (
            None
            if baseline is None or val_score is None
            else _gain(val_score, baseline, spec.metric)
        ),
        "artifact_path": session.rel(exp_dir),
    }
    journal.append_experiment(session.run_dir, row)
    j_after = journal.load(session.run_dir)
    columns_by_table = [
        {"table": r["table"], "method": r["method"], "n_columns": len(r["columns"])}
        for r in _columns_by_table(session, j_after, row["source_table"], result.feature_names)
    ]
    vs_source, vs_source_basis = _vs_source_table(
        session, j_after, row["source_table"], val_score, spec.metric, model_type, params
    )
    # All three are derived from the journal *after* this row lands, so they
    # cannot be part of the append. Persist them anyway: they are the record of
    # whether an idea paid off, and a fresh agent that only has the scores cannot
    # rebuild them. None of them affects `best`/`baseline`.
    journal.update_experiment(
        session.run_dir,
        exp_id,
        {
            "vs_source_table": vs_source,
            "vs_source_basis": vs_source_basis,
            "columns_by_table": columns_by_table,
        },
    )

    status = _progress(session)
    table_state = stop.table_status(
        journal.load(session.run_dir)["leaderboard"],
        spec.metric,
        session.stop_policy,
        session.rel(path),
    )

    # What is *not* here is deliberate. This dict ships on every training call, so
    # anything constant for the whole run (the metric and its direction) is said
    # once by `inspect_data` instead, anything that only restates another field
    # (the raw fold scores under `fold_spread`, the stop counters under `note`) is
    # dropped, and anything that would read the same every time is prompt text,
    # not payload. A caption repeated on every call buys nothing after the first.
    out = {
        "trained": True,
        "experiment_id": exp_id,
        # First, because it is what the leaderboard ranks by (0019). `cv_score`
        # follows it rather than being dropped: the two are measured on different
        # rows and their disagreement is information — a gain that shows up in the
        # folds and not on the validation half is a gain in the folds only.
        "val_score": val_score,
        "cv_score": cv_score,
        "fold_spread": fold_spread,
        "model_type": model_type,
        "n_features": len(result.feature_names),
        "source_table": row["source_table"],
        "dropped_non_numeric": dropped,
        "vs_baseline": row["vs_baseline"],
        "vs_source_table": vs_source,
        # What this table is made of: one entry per table in its history, each
        # being one tool call's columns. No score attached — which of them earned
        # the gain takes refits to answer, and that is `explain_table`.
        "columns_by_table": columns_by_table,
        "artifact_path": row["artifact_path"],
        # The advisory half of the stop rule (§19.3): where the run stands after
        # every experiment, so a cooperative model never has to be refused at all.
        "note": stop.advisory(status, table_state),
    }
    if vs_source is None:
        # Only when there is no gain. A missing number is a fixable state and the
        # fix is a call the agent will not think to make unless told; a caption on
        # a number that is present would just be describing the obvious.
        out["vs_source_basis"] = vs_source_basis
    if val_unavailable is not None:
        # Same rule: say it only when the number is missing, and say what stopped
        # it. This row cannot win the run, which the agent needs to know now
        # rather than discovering at finalize.
        out["val_unavailable"] = val_unavailable
    return out


def explain_table(
    session: RunSession,
    table: str | None = None,
    model_type: str = "lgbm",
    params: dict | None = None,
) -> dict:
    """Which table's columns earned the score, by refitting without each in turn.

    Feature importance cannot answer this and never could. Importance decomposes
    *one fitted model's attention*; the question is a counterfactual — what would
    the score be if these columns had never been built — and the two come apart
    routinely. A column the trees split on constantly can be worth nothing once
    removed, because something else carries the same signal; a binary column
    offering a single split point can carry the entire gain.

    So this measures the counterfactual directly. Fit once on everything, then
    once more per contributing table with that table's columns withheld, same
    folds, same model, same params. ``contribution`` is the full score minus the
    score without them, signed as a gain: positive means removing them hurt,
    which means they helped.

    Grouped by table because one tool call writes one table, so that grouping is
    already the unit the agent can act on — and per-column numbers would be finer
    than any move available to it.

    Two properties fall out of leave-one-out that the importance version had to
    hedge about in prose:

    * **Interaction partners both get credit.** If ``K`` is worthless alone but
      ``K + L`` is strong, withholding either one costs the score, so both score
      positive. Nothing needs warning the agent off dropping a partner.
    * **Redundancy reads as ~0 honestly.** Two tables encoding the same signal
      each measure as free to remove, and that is the truth about each one *given
      the other* — the pair still has to be tested as a pair.

    Costs one fit per contributing table plus one. It writes no leaderboard row
    and counts against no budget: it produces no model to ship, and it is most
    useful on a table the stop rule has already declined, which is exactly when
    asking "what happened here" is the right move.
    """
    path = session.resolve_table(table)
    spec = session.resolved()
    journal.set_split(session.run_dir, session.split_record())
    df = load_table(path)

    X, dropped = _feature_frame(df, spec.target)
    if X.shape[1] == 0:
        raise ValueError(
            "no numeric feature columns to explain; use transform_features or "
            "fit_encoding to produce some first"
        )
    y = df[spec.target]
    folds = make_folds(y, spec.task_type, n_splits=spec.n_splits, seed=spec.seed)

    def score(columns: list[str]) -> float:
        result = train_cv(
            X[columns],
            y,
            folds,
            spec.task_type,
            spec.metric,
            model_type=model_type,
            params=params,
            seed=spec.seed,
        )
        return float(np.mean(result.fold_metrics))

    all_columns = list(X.columns)
    # Always refit rather than reusing a leaderboard score for this table: a
    # stored score may come from other params or another model, and a difference
    # of protocol would land in the contribution as if it were a difference of
    # features.
    full_score = score(all_columns)

    j = journal.load(session.run_dir)
    report = []
    for row in _columns_by_table(session, j, session.rel(path), all_columns):
        remaining = [c for c in all_columns if c not in set(row["columns"])]
        entry = {
            "table": row["table"],
            "method": row["method"],
            "n_columns": len(row["columns"]),
        }
        if not remaining:
            # Nothing else contributed columns. "Without them" is not a model, and
            # a contribution of "everything" would be true but useless — that is
            # what vs_baseline already says.
            entry["contribution"] = None
            entry["reason"] = "the only columns on this table; see vs_baseline instead"
        else:
            entry["contribution"] = _gain(full_score, score(remaining), spec.metric)
        report.append(entry)

    journal.update_table(session.run_dir, session.rel(path), {"credit_by_table": report})

    return {
        "table": session.rel(path),
        "metric": spec.metric,
        "model_type": model_type,
        "cv_score_all_columns": full_score,
        "dropped_non_numeric": dropped,
        "n_fits": 1 + sum(1 for r in report if r["contribution"] is not None),
        "credit_by_table": report,
    }


def _experiment_row(run_dir: str, exp_id: str) -> dict:
    """The leaderboard row for ``exp_id``, or a ValueError naming the real ones."""
    rows = journal.load(run_dir)["leaderboard"]
    for row in rows:
        if row["id"] == exp_id:
            return row
    raise ValueError(
        f"no such experiment: {exp_id!r}. Available: "
        + (", ".join(repr(r["id"]) for r in rows) or "none — nothing has been trained yet")
    )


def _residual_frame(df: pd.DataFrame, oof: pd.DataFrame, y: pd.Series, task_type: str,
                    n_splits: int, seed: int) -> pd.DataFrame:
    """The source table, plus what the model predicted for each row and its error.

    Row alignment is positional and holds by construction: ``train_cv`` resets
    the index before it fills the out-of-fold array, and ``oof.parquet`` is
    written in that order from the same table this reloads. Nothing here joins on
    a key, so nothing here can silently mismatch.

    ``fold`` is recomputed rather than stored — ``make_folds`` is deterministic in
    ``(y, task_type, n_splits, seed)``, all four of which are fixed for the run.
    It is worth the two lines because without it a "weak slice" and "one fold fit
    badly" look identical.
    """
    out = df.copy()
    values = oof.to_numpy()

    if task_type == "regression":
        out["oof_pred"] = values[:, 0]
        out["residual"] = y.to_numpy() - out["oof_pred"]
    elif task_type == "binary":
        # The OOF column is P(positive class), so the residual is the same
        # quantity as in regression: truth minus what was predicted, signed, so
        # the two directions of being wrong stay distinguishable.
        classes = sorted(pd.unique(y))
        out["oof_pred"] = values[:, 0]
        out["residual"] = (y.to_numpy() == classes[-1]).astype(float) - out["oof_pred"]
    else:
        # Multiclass has a probability per class and no signed error. The useful
        # scalar is how much mass landed on the truth, which reduces to the same
        # "how wrong, per row" question the other branches answer.
        classes = sorted(pd.unique(y))
        for i, cls in enumerate(classes):
            out[f"oof_pred_{cls}"] = values[:, i]
        out["oof_pred_label"] = [classes[i] for i in values.argmax(axis=1)]
        out["p_true"] = [values[r, classes.index(c)] for r, c in enumerate(y)]
        out["correct"] = out["oof_pred_label"].to_numpy() == y.to_numpy()

    fold = np.empty(len(out), dtype=int)
    for i, (_, va) in enumerate(make_folds(y, task_type, n_splits=n_splits, seed=seed)):
        fold[va] = i
    out["fold"] = fold
    return out


def explore_experiment(
    session: RunSession,
    exp_id: str,
    code: str,
    timeout: float = 10.0,
) -> dict:
    """Read-only code over one experiment's rows and its out-of-fold errors.

    This closes the one arc of the loop that was missing. The agent could
    inspect, build, train and score, but the step from a score back to the next
    hypothesis had nothing to work with: it saw a single number per experiment
    and had to guess which feature to try next from column names alone. Here it
    can ask *where* the model is wrong — which fold, which segment, which end of
    which distribution — and let the answer pick the next feature.

    The frame is the **whole source table**, not just the numeric columns the
    model trained on. Deliberate: the string columns a model cannot consume are
    exactly the ones worth grouping residuals by, and "the error is concentrated
    in one city" is how the agent discovers that a column is worth encoding at
    all. They cost nothing here because nothing here becomes a feature.

    Nothing it computes can be persisted. That is the whole guard, and it is the
    same one :mod:`~tabular_agent.explore` documents: the result is shown and
    discarded, so a per-segment target mean read here has no route back into the
    model except `fit_encoding`, which fits per fold. The target and the residual
    are both in the frame and the row cap does not pretend otherwise — the
    protection is the absence of a persistence path, not the absence of the
    number.

    Free, like `explain_table`: no leaderboard row, no budget, and allowed on an
    experiment whose table the stop rule has already declined. Understanding why
    a table failed is most valuable exactly then.
    """
    row = _experiment_row(session.run_dir, exp_id)
    spec = session.resolved()
    path = session.resolve_table(row["source_table"])
    df = load_table(path)
    oof = pd.read_parquet(os.path.join(session.run_dir, row["artifact_path"], "oof.parquet"))
    frame = _residual_frame(
        df, oof, df[spec.target], spec.task_type, spec.n_splits, spec.seed
    )

    out = run_readonly({"df": frame}, code, func_name="explore", timeout=timeout)
    result = {"experiment_id": exp_id, "result": out["result"]}
    if out["stdout"]:
        # Only when there is something to say. An empty field on every call is
        # the kind of constant caption 0018 spent a prompt rewrite removing.
        result["stdout"] = out["stdout"]
    return result


def finalize_run(session: RunSession) -> dict:
    """Package the best experiment: deployable bundle + human-readable report.

    Called **once**, at the end (§19.1). Reads the journal's ``best`` pointer
    rather than re-deriving anything, and auto-trains the trivial baseline if the
    run never did, so the report can always answer "did this beat doing nothing".
    """
    j = journal.load(session.run_dir)
    if journal.stop_state(session.run_dir).get("finalized"):
        raise ValueError(
            "this run has already been finalized: the bundle and report are on disk "
            "and the winner will not be re-picked"
        )
    if not j["leaderboard"]:
        raise ValueError("nothing to finalize: no model has been trained yet")

    if j.get("baseline") is None:
        train_tabular_model(session, model_type="baseline", enforce_stop=False)
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

    credit = journal.attribution(j, meta["source_table"])
    text = build_report(
        attribution=credit,
        task_type=meta["task_type"],
        metric=meta["metric"],
        # The validation half is what the leaderboard ranked by, so it is what
        # picked this winner and what the report leads with. `cv_score` follows as
        # the fold estimate — the gap between them is the run's own record of how
        # far the search drifted from the folds it was optimising.
        selection_score=meta["val_score"],
        cv_score=meta["cv_score"],
        fold_metrics=meta["fold_metrics"],
        baseline_score=j["baseline"]["val_score"],
        model_types=[meta["model_type"]],
        feature_names=meta["feature_names"],
        split=j.get("split"),
    )
    report_path = write_report(session.ctx, text)
    journal.mark_finalized(session.run_dir)

    return {
        "best_experiment_id": best["experiment_id"],
        "model_type": meta["model_type"],
        "metric": meta["metric"],
        "val_score": meta["val_score"],
        "cv_score": meta["cv_score"],
        "baseline_score": j["baseline"]["val_score"],
        "n_experiments": len(j["leaderboard"]),
        "bundle_dir": session.rel(bundle_dir),
        "report_path": session.rel(report_path),
        # Which links in the winning table's history were actually measured. An
        # unmeasured one means the score cannot be attributed to the columns
        # either side of it — worth saying, since the agent is about to write a
        # summary claiming some feature was the one that worked.
        "unmeasured_tables": [s["table"] for s in credit if not s["measured"]],
    }
