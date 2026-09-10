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

import contextlib
import fcntl
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd

from ..compare import compare_predictions
from ..cv import make_folds
from ..encoders import STREAM_STATS, TARGET_STATS, StreamAggregateEncoder
from ..encoders import fit_encoding as _fit_encoding
from ..encoders import fit_group_stats as _fit_group_stats
from ..encoders import fit_history as _fit_history
from ..encoders import to_periods as _to_periods
from ..inspect import profile_dataframe, profile_stream
from ..io import load_table, save_parquet
from ..metrics import compute_metric, describe_metric, greater_is_better
from ..models import _make_model, predict_model, train_cv
from ..pipeline import Pipeline, save_bundle
from ..probe import PROBE_MODEL_TYPE, PROBE_PARAMS, PROBE_SEED, fit_probe, probe_predict
from ..blend import apply_blend, blend_weights
from ..report import (
    accuracy_interval,
    build_report,
    shipped_holdout,
    target_composition,
    write_report,
)
from ..rows import select_rows
from ..sandbox import SandboxError, run_readonly
from ..target import TARGET_TRANSFORMS, invert_target
from ..transform import transform_features as _transform_features
from . import journal, stop
from .session import RunSession

#: Column-level detail is capped so a wide table cannot blow up the context.
#:
#: Raised from 60 once a dataset arrived with a real data dictionary. Descriptions
#: ride on their column (0018's channel), so truncating the profile silently
#: truncates the dictionary — house-prices has 80 columns and the 20 that fell off
#: included four whose notes carry the level order an ordinal encoding needs. The
#: cap exists to stop a 500-column table, not an 80-column one; a full profile here
#: is a few thousand tokens, once per run.
MAX_REPORTED_COLUMNS = 100

#: Absolute floor on what a row filter may leave behind, on top of the
#: fold-count floor. Fewer rows than this is not a smaller training set, it is a
#: predicate that went wrong — and the resulting table exists only to be refused.
MIN_FILTERED_ROWS = 10


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


def _fold_order(session: RunSession, df: pd.DataFrame):
    """The column folds must respect, or ``None`` when they need respect nothing.

    A run split on a column is a run whose validation half is the *later* part of
    that column, so its cross-validation has to ask the same question: train on
    earlier rows, predict later ones. Leaving the inner folds random while the
    outer split is ordered makes `cv_score` measure interpolation and `val_score`
    measure forecasting, and their disagreement then looks like overfitting rather
    than what it is — two different tasks.

    Raises rather than falling back to random folds when the column is missing.
    Silently changing the evaluation protocol because a table happens to have
    dropped a column is the kind of failure that produces a confident number
    nobody can interpret afterwards.
    """
    if session.split_on is None:
        return None
    if session.split_on not in df.columns:
        raise ValueError(
            f"this run is split on {session.split_on!r}, so every table has to keep "
            f"that column for cross-validation to respect the same ordering — and "
            f"it is not in this one. Columns are {list(df.columns)}. Rebuild the "
            f"table without dropping {session.split_on!r}."
        )
    return df[session.split_on].to_numpy()


#: Rows in the serving probe. Enough that a transform doing real arithmetic
#: behaves normally; few enough that the replay costs nothing.
SERVING_PROBE_ROWS = 20


def _unavailable_at_predict_time(session: RunSession, df: pd.DataFrame) -> list[str]:
    """Raw columns of ``df`` that will not exist when a prediction is asked for."""
    if session.serving_columns is None:
        return []
    target = session.resolved().target
    return [
        c for c in df.columns
        if c != target and c not in session.serving_columns
    ]


def _run_streams(session: RunSession) -> dict:
    """The run's declared streams, loaded, for replaying a lineage inside the run.

    Inside a run the stream is the copy pinned in the run directory, so every
    experiment sees the same file and two tables stay comparable. At predict time
    the caller supplies it instead, deliberately — a stream that has moved on
    since the fit is the one a served model should be reading (0062).

    Loaded per call rather than cached on the session: a cached frame is a frame
    that outlives the reason it was read, and these are read on a path that
    already touches disk twice.
    """
    return {
        name: load_table(session.stream_path(name)) for name in session.stream_names()
    }


def _profile_streams(session: RunSession, df: pd.DataFrame) -> dict:
    """Each declared stream, summarised against the table it would attach to.

    The overlap with ``df.columns`` is the useful half: it is the set of possible
    join keys, and computing it here is the difference between the agent being
    told a second file exists and being told how it attaches.
    """
    out = {}
    for name in session.stream_names():
        spec = session.stream_spec(name)
        out[name] = profile_stream(
            load_table(session.stream_path(name)),
            event_time=spec["event_time"],
            observed_at=spec["observed_at"],
            table_columns=list(df.columns),
            notes=spec.get("notes"),
        )
    return out


def _refuse_if_unservable(
    session: RunSession, path: str, df: pd.DataFrame, feature_names: list[str]
) -> dict | None:
    """``None`` if this model could actually be served, otherwise the refusal.

    The check is a **replay, not a name comparison**, and it has to be: almost no
    feature is a raw column. `Store_te`, `Year`, `CompetitionOpenMonths` are all
    derived, and none of them appears in a serving schema either. What matters is
    whether the table's lineage can *produce* the model's columns from the columns
    that will exist — so this builds a serving-shaped frame and runs the lineage
    over it, which is exactly what `predict_bundle` will do later.

    One mechanism, two failures, and the second is why a name comparison would not
    have been enough (Rossmann, 2026-08-11):

    * the model trains on a column that will not exist — the leak;
    * the model does not use the column, but a stored transform *references* it,
      and per 0010 the code is the artifact and gets re-run on rows that never had
      it. `df.drop(columns=["X"])` fails this; `errors="ignore"` survives it.

    Refused rather than silently dropped, and refused *before* the fit: a run that
    discovers this at `finalize_run` has spent its whole budget optimising a
    number measured on rows that will never exist again. Which is not
    hypothetical — it is what happened, and the leaked score won the leaderboard
    while ten genuine improvements were counted as failures.
    """
    if session.serving_columns is None:
        return None

    # The probe is built from **raw** rows, not from the table being trained on.
    # A table's lineage replays every step from raw input — that is what
    # `_score_on_val` does and what `predict_bundle` will do — so handing it an
    # already-transformed frame would re-apply the chain to its own output. It
    # happens to give the right answer for a single step and stops doing so as
    # soon as a table has a parent.
    spec = session.resolved()
    raw = load_table(session.data_path).head(SERVING_PROBE_ROWS)
    absent = _unavailable_at_predict_time(session, raw)
    probe = raw.drop(columns=absent + [spec.target], errors="ignore")

    try:
        produced = _load_lineage(path, spec.task_type).transform(
            probe, streams=_run_streams(session)
        )
    except Exception as e:
        return {
            "trained": False,
            "table": session.rel(path),
            "unservable": True,
            "reason": (
                f"this table cannot be rebuilt at prediction time: replaying its "
                f"steps over rows without {absent} raised {type(e).__name__}: {e}"
            ),
            "columns_not_available_at_predict_time": absent,
            "next": (
                "the stored code is re-run on raw rows at prediction time, and "
                "those rows never had these columns. Guard the reference — "
                "`df.drop(columns=[...], errors='ignore')`, or check membership "
                "first — and build the table again."
            ),
        }

    missing = [c for c in feature_names if c not in produced.columns]
    if not missing:
        return None
    return {
        "trained": False,
        "table": session.rel(path),
        "unservable": True,
        "reason": (
            f"{len(missing)} of this table's feature columns cannot exist at "
            f"prediction time: {missing[:8]}. Training on them would produce a "
            f"score measured on rows that will never occur again, and a model "
            f"that cannot be asked for a prediction at all."
        ),
        "columns_not_available_at_predict_time": absent,
        "next": (
            "build a table without them — transform_features with "
            "`df.drop(columns=[...], errors='ignore')` — and train on that. The "
            "guard matters: the code is stored and re-run on rows that never had "
            "the columns."
        ),
    }


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


#: Bootstrap resamples behind `val_se`. A few hundred is plenty for a standard
#: error (it is a second moment, not a tail quantile), and the whole loop is
#: milliseconds against the seconds a fit already costs.
N_BOOTSTRAP = 200


def _bootstrap_se(metric: str, y, preds, labels, seed: int) -> float | None:
    """The standard error of a metric, by resampling the rows it was computed on.

    This is the yardstick 0022 says every validation comparison needs. The score
    itself is one draw from a distribution — resample the validation rows with
    replacement, rescore, and the spread of those scores is how much the number
    would have moved had the split fallen differently.

    Seeded from the run's own seed so it is reproducible: `val_se` gates the stop
    rule, and a noise estimate that changes between identical runs would make the
    run's control flow non-deterministic.

    ``None`` when it cannot be computed — too few rows to resample meaningfully,
    or a metric that keeps failing on the resamples (a class missing from a draw,
    typically). A missing yardstick is handled by the caller; a wrong one is not
    detectable at all.
    """
    y = np.asarray(y)
    preds = np.asarray(preds)
    n = len(y)
    if n < 10:
        return None

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        try:
            scores.append(
                compute_metric(metric, y[idx], preds[idx], labels=labels)
            )
        except Exception:
            # A resample can be degenerate — one class only, say — and that draw
            # is uninformative rather than fatal. Enough of them failing is what
            # the length check below treats as "no estimate".
            continue
    if len(scores) < N_BOOTSTRAP // 2:
        return None
    return float(np.std(scores))


def _score_on_held_out(
    session: RunSession,
    rows_path: str,
    source_path: str,
    models,
    feature_names,
    spec,
    y_work: pd.Series,
    target_transform: str = "none",
    weights=None,
) -> tuple[float | None, float | None, str | None, np.ndarray | None]:
    """Score just-fitted models on held-out rows. ``(score, se, why_not, preds)``.

    ``weights`` blends the members with the given (already-normalised) vector
    instead of an equal average — used only to score the inverse-fold-spread blend
    the run may ship (0080). ``None`` keeps the equal average every other caller
    relies on.

    The predictions come back with the score because they are the only thing that
    makes a *paired* comparison possible later (`compare.py`): the difference
    between two candidates can only be given an interval if both were evaluated on
    the same rows and both prediction vectors survived. Recomputing them means
    refitting, so they are returned here, where they already exist, and the caller
    decides whether to keep them.

    ``models`` is a list because a run can ship the weighted average of the
    terminal candidates rather than one of them (0044, 0080). One model and several go
    through the same code for the same reason the two row-sets do: a blend's
    score is only comparable to its members' if it was produced identically.
    Predictions are averaged **after** each model's transform is inverted,
    matching `predict_bundle` — see its docstring for why the order matters.

    ``rows_path`` is the validation half for everything the agent can reach, and
    the sealed holdout for the single call `finalize_run` makes (0023). The
    machinery is deliberately identical: the sealed score is only worth printing
    beside `val_score` if it was produced the same way, so that the gap between
    the two measures selection and not a difference in how they were computed.

    The held-out rows are **raw**: no tool ever wrote them, so the columns the
    model expects do not exist on them yet and have to be rebuilt by replaying
    the lineage of the table it trained on — the same pipeline ``finalize_run``
    packages into a bundle. Encoders replay their fitted parameters, so nothing
    learned from validation rows enters the features.

    Returns a reason instead of raising when the replay cannot produce the
    model's columns: a leaderboard row with a missing validation score is worth
    more than a training call that dies.
    """
    val = load_table(rows_path)
    y_val = val[spec.target]
    pipeline = _load_lineage(source_path, spec.task_type)
    # The target is dropped before the pipeline sees the rows (0009): a transform
    # that can read the target is a transform that can leak it.
    features = pipeline.transform(
        val.drop(columns=[spec.target]), streams=_run_streams(session)
    )

    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        return None, None, f"replaying the table's lineage did not produce {missing[:5]}", None

    classes = None
    if spec.task_type in ("binary", "multiclass"):
        # Union of both halves: a class present in work but absent from
        # validation (or the reverse) still has to occupy its column, or the
        # probability matrix silently shifts and logloss scores the wrong class.
        classes = sorted(set(pd.unique(y_work)) | set(pd.unique(y_val)))
    X = features[list(feature_names)]
    each = []
    for model in models:
        p = predict_model(
            model, X, spec.task_type,
            classes=classes if spec.task_type == "multiclass" else None,
        )
        # The model was fitted on f(y), so its predictions are on that scale.
        # Scoring them against raw validation targets without inverting would not
        # merely be wrong, it would be *flatteringly* wrong for `rmsle` — which is
        # why this inversion is tested rather than trusted.
        if target_transform != "none":
            p = invert_target(target_transform, p)
        each.append(p)
    if len(each) == 1:
        preds = each[0]
    else:
        w = np.full(len(each), 1.0 / len(each)) if weights is None else np.asarray(weights, dtype="float64")
        preds = apply_blend(each, w)
    score = float(compute_metric(spec.metric, y_val, preds, labels=classes))
    # Computed here rather than in the stop rule because this is the only place
    # that still holds the predictions: the leaderboard keeps the score, not the
    # rows behind it.
    #
    # Not computed at all on an ordered split. The bootstrap resamples rows as if
    # they were exchangeable draws from one population; under `split_on` they are
    # a *later period*, deliberately chosen to differ from the training rows, and
    # resampling within it measures how much the score wobbles inside that period
    # while saying nothing about the thing that actually varies — which period it
    # is. The number would be confidently too small, and a too-small yardstick is
    # worse than none: `stop._noise` falls back to `fold_spread`, which at least
    # does not claim to measure something it cannot see.
    se = (
        None
        if session.profile == "minimal" or session.split_on
        else _bootstrap_se(spec.metric, y_val, preds, classes, spec.seed)
    )
    return score, se, None, np.asarray(preds)


def _score_on_val(
    session: RunSession,
    source_path: str,
    models,
    feature_names,
    spec,
    y_work: pd.Series,
    target_transform: str = "none",
    weights=None,
) -> tuple[float | None, float | None, str | None, np.ndarray | None]:
    """`_score_on_held_out` against the validation half — the leaderboard's exam.

    Kept as its own name because every caller in the run is this one, and the
    single caller that is *not* has to say so at its call site.
    """
    return _score_on_held_out(
        session, session.val_path, source_path, models, feature_names, spec, y_work,
        target_transform=target_transform, weights=weights,
    )


def _probe_preds_path(session: RunSession, table_path: str) -> str:
    """Where a table's probe predictions on the validation half are kept.

    One vector per table, `float64` over the validation rows — 400KB on a 50k-row
    validation half, bounded by the table count the stop rule already caps. Kept
    because a paired comparison needs both candidates' predictions on the same
    rows (`compare.py`), and the alternative is refitting the parent every time a
    child is scored, which doubles the cost of the one thing that runs on every
    table.

    Named from the table's own filename, so it survives a resume for the same
    reason the tables do: the run directory is the state.
    """
    stem = os.path.splitext(os.path.basename(table_path))[0]
    return session.ctx.path("probe_preds", f"{stem}.npy")


def _agent_score_fields(session: RunSession, scored: dict) -> dict:
    """Public score vocabulary for the selected product profile."""
    if session.profile != "minimal":
        return scored
    out = dict(scored)
    if "probe_score" in out:
        out["evaluation_score"] = out.pop("probe_score")
    if "probe_vs_source_table" in out:
        out["gain_vs_parent"] = out.pop("probe_vs_source_table")
    if "probe_unavailable" in out:
        out["evaluation_unavailable"] = out.pop("probe_unavailable")
    for key in (
        "probe_se", "probe_fit_seconds", "probe_target_transform",
        "probe_vs_source_interval",
    ):
        out.pop(key, None)
    return out


def _load_probe_preds(session: RunSession, table_path: str) -> np.ndarray | None:
    """A table's stored probe predictions, or ``None`` if it has none.

    Missing is normal rather than exceptional: tables probed before this existed
    have no file, and a table the probe could not score never wrote one. Callers
    degrade to reporting the difference without an interval.
    """
    path = _probe_preds_path(session, table_path)
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:  # noqa: BLE001 - a corrupt sidecar must not kill a tool call
        return None


def _val_order_values(session: RunSession) -> np.ndarray | None:
    """The validation half's ordering column, or ``None`` on a random split.

    This is what picks `compare_predictions`' mode, and it is read from the rows
    rather than passed around as a flag — see that function for why. Returns
    ``None`` when the column is missing from the validation file, which would
    otherwise silently produce blocks cut on nothing.
    """
    if not session.split_on:
        return None
    try:
        val = load_table(session.val_path)
    except Exception:  # noqa: BLE001
        return None
    if session.split_on not in val.columns:
        return None
    return val[session.split_on].to_numpy()


def probe_table(session: RunSession, table_path: str) -> dict:
    """Score a just-created table with the run's fixed ruler (0035, 0036).

    Every table is scored the moment it exists, rather than only the ones an
    agent chooses to spend an experiment on. The reason is coverage: agents build
    chains — t12 then t13 then t14 — and train on the endpoint, so across the runs
    in `_runs/` **22 of the 24 tables that recombined an abandoned branch could
    not be compared to the branch they continued**, because that parent was never
    trained. A table with no number cannot take part in any comparison, and the
    missing ones were concentrated exactly where the interesting comparisons are.

    Affordable because the ranking number is the validation score, which comes
    from a model fitted on all of the table's rows: **one fit per table, not
    ``n_splits + 1``.** Cross-validation still happens for the terminal fit and
    for ablation credit; it is not on this path.

    **Never raises.** A table is a real artifact whether or not it could be
    scored, and the tool that produced it has already succeeded by the time this
    runs. Anything that goes wrong is returned as `probe_unavailable` with a
    reason — a table the agent can still build on beats a tool call that dies
    after writing its output.
    """
    spec = session.resolved()
    try:
        df = load_table(table_path)
        X, _ = _feature_frame(df, spec.target)
        if X.empty or not len(X.columns):
            return {"probe_score": None, "probe_unavailable": "no numeric feature columns"}
        # 0030, at its new home. This used to be checked by `train_tabular_model`
        # and refused there; since 0035 removed that tool from the agent, the
        # ruler is the first thing to touch a table and therefore the only place
        # the agent can still be told. Left unscored rather than scored: an
        # unservable table must not be able to win `best_table` and be packaged
        # into a bundle that cannot answer a prediction.
        refusal = _refuse_if_unservable(session, table_path, df, list(X.columns))
        if refusal is not None:
            return {
                "probe_score": None,
                "probe_unavailable": refusal["reason"],
                "unservable": True,
                "columns_not_available_at_predict_time":
                    refusal.get("columns_not_available_at_predict_time"),
                "next": refusal.get("next"),
            }
        y = df[spec.target]
        fit = fit_probe(X, y, spec.task_type, spec.metric, spec.target_transform)
        score, se, why_not, preds = _score_on_val(
            session, table_path, [fit.model], fit.feature_names, spec, y,
            target_transform=fit.target_transform,
        )
    except Exception as e:  # noqa: BLE001 - see the docstring: this cannot fail loudly
        return {"probe_score": None, "probe_unavailable": f"{type(e).__name__}: {e}"}

    if score is None:
        return {"probe_score": None, "probe_unavailable": why_not}
    if preds is not None and session.profile != "minimal":
        # Written after the score, so a table with a number always has its
        # predictions and never the reverse. Failing to persist them costs an
        # interval on a later comparison, never the score itself, so it is not
        # allowed to turn a successful probe into a failed tool call.
        try:
            np.save(_probe_preds_path(session, table_path), np.asarray(preds, dtype=float))
        except Exception:  # noqa: BLE001 - see the docstring: this cannot fail loudly
            pass
    out = {
        "probe_score": round(score, 6),
        "probe_fit_seconds": fit.fit_seconds,
        "probe_target_transform": fit.target_transform,
    }
    if session.profile != "minimal":
        out["probe_se"] = None if se is None else round(se, 6)
    return out



def _probe_and_compare(session: RunSession, table_path: str, source_rel: str) -> dict:
    """`probe_table`, plus the only comparison that is actually a decision.

    A table's score on its own says little; what a run needs to know is whether
    this table improved on **the one it was derived from**, which is the
    comparison the agent is implicitly making when it builds anything. Reporting
    it here means the agent does not have to reconstruct it from a leaderboard,
    and means it is denominated the same way every time.
    """
    out = probe_table(session, table_path)
    if out.get("probe_score") is None:
        return out
    before = _source_reference(session, source_rel)
    if before is not None:
        out["probe_vs_source_table"] = _gain(out["probe_score"], before, session.resolved().metric)
        if session.profile != "minimal":
            out.update(_difference_interval(session, table_path, source_rel))
    return out


def _difference_interval(session: RunSession, table_path: str, source_rel: str) -> dict:
    """An interval around `probe_vs_source_table` — 0036's instrument, applied.

    The gain on its own says a table beat its parent by some amount; it has never
    said whether that amount is bigger than the evaluation rows' own noise. Every
    feature decision this project has recorded was made without one, and several
    were later withdrawn.

    Returns an empty dict rather than nulls when it cannot be computed — a table
    probed before predictions were stored, a parent that was never scored, a
    validation half too small to block. An absent key reads as "not measured",
    where a null field invites the reader to treat it as "measured, and zero".
    """
    parent_path = session.resolve_table(None) if source_rel is None else os.path.join(
        session.run_dir, source_rel
    )
    mine = _load_probe_preds(session, table_path)
    theirs = _load_probe_preds(session, parent_path)
    if mine is None or theirs is None:
        return {}
    try:
        val = load_table(session.val_path)
        spec = session.resolved()
        y_val = val[spec.target]
        labels = (
            sorted(pd.unique(y_val)) if spec.task_type in ("binary", "multiclass") else None
        )
        result = compare_predictions(
            spec.metric, y_val, mine, theirs,
            order_values=_val_order_values(session),
            labels=labels,
            seed=spec.seed,
        )
    except Exception:  # noqa: BLE001 - an interval is a nicety; the score is not
        return {}
    if result is None:
        return {}
    return {"probe_vs_source_interval": result}


def _source_reference(session: RunSession, source_rel: str) -> float | None:
    """The probe score of the table a new one was derived from.

    Derived tables carry theirs on their journal row. The **work table** has no
    row — nothing derived it — so its score is computed the first time anything
    is built on it and cached in the journal. That number is the run's real
    reference point: "the data as given, measured by the same ruler", which is
    what every later table is implicitly claiming to beat.

    Distinct from `baseline`, which is a trivial predictor and answers a
    different question — whether modelling this data beats predicting its mean.
    """
    j = journal.load(session.run_dir)
    if source_rel != session.rel(session.data_path):
        parent = next(
            (t for t in (j.get("tables") or []) if t.get("path") == source_rel), None
        )
        return (parent or {}).get("probe_score")

    cached = (j.get("work_probe") or {}).get("probe_score")
    if cached is not None:
        return cached
    scored = probe_table(session, session.data_path)
    journal.set_work_probe(session.run_dir, scored)
    return scored.get("probe_score")


def inspect_data(session: RunSession, table: str | None = None) -> dict:
    """Profile a table and pin down the task type and selection metric.

    Always the first call of a run: it resolves the task spec (auto-detecting
    task type and metric from the target when the user left them open) and
    records it in the journal, so every later score is comparable.
    """
    path = session.resolve_table(table)
    df = load_table(path)
    spec = session.resolved()
    prof = profile_dataframe(df, target=spec.target, column_notes=spec.column_notes)

    _, dropped = _feature_frame(df, spec.target)
    journal.set_task(
        session.run_dir, target=spec.target, task_type=spec.task_type, metric=spec.metric
    )
    journal.set_split(session.run_dir, session.split_record())
    journal.set_streams(session.run_dir, session.stream_record())

    unavailable = _unavailable_at_predict_time(session, df)
    streams = _profile_streams(session, df)
    columns = prof["columns"]
    truncated = len(columns) > MAX_REPORTED_COLUMNS
    shown = dict(list(columns.items())[:MAX_REPORTED_COLUMNS])

    out = {
        "table": session.rel(path),
        "n_rows": prof["n_rows"],
        "n_cols": prof["n_cols"],
        "task_type": spec.task_type,
        "metric": spec.metric,
        # Fixed for the whole run, so it is stated here rather than repeated on
        # every training result. Every `vs_...` field is already signed as a gain;
        # this is the direction of `cv_score` itself.
        "higher_is_better": greater_is_better(spec.metric),
        # What the metric rewards, not just which way to push it. The direction
        # alone does not say that `rmspe` and `rmsle` punish *relative* error, nor
        # that `rmspe` does not score zero-valued rows at all — and both change
        # what a good model looks like rather than merely how it is graded.
        "metric_measures": describe_metric(spec.metric),
        "n_splits": spec.n_splits,
        "target": prof.get("target"),
        "columns": shown,
        "columns_truncated": truncated,
        "non_numeric_columns": dropped,
        # Only when a serving schema was supplied, and only when something is
        # actually missing from it. This is the one fact no amount of profiling
        # recovers, and training on such a column is refused rather than warned
        # about — the note says so, so the refusal is not a surprise.
        **({"columns_not_available_at_predict_time": unavailable}
           if unavailable else {}),
        # Only when the run declares one. A stream is a *second input* — a related
        # file at a finer grain, whose records say both what they are about and
        # when they became knowable — and it is invisible to every other tool, so
        # if it is not named here the agent never learns it exists.
        **({"streams": streams,
            "streams_note": (
                "These are second inputs, not tables: no tool returns rows from "
                "them. Reach one with fit_stream_aggregate, which aggregates a "
                "stream's records per row of a table, counting only the records "
                "already observed a stated lead before that row. 'notice' says how "
                "far ahead of the event its records typically arrived — where "
                "notice_zero_share is large, a lead you can actually serve at will "
                "discard that share, and the column you would get without one is "
                "richer in training than it can ever be at predict time."
            )} if streams else {}),
        # Present only when the data's owner supplied it. Facts about the dataset
        # as a whole — what a row is, which rows the real scoring ignores, how it
        # was collected — that have no single column to attach to. Per-column
        # descriptions ride on their columns in `columns` above instead.
        **({"about_this_data": spec.notes} if spec.notes else {}),
        "note": (
            "Non-numeric columns are not usable by the models as-is; encode them "
            "with fit_encoding to bring them in — method='ordinal' (stating the "
            "level order) when the categories are genuinely ordered, "
            "method='frequency' or 'target' when they are nominal."
            if dropped
            else "All feature columns are numeric and usable as-is."
        ),
    }
    if session.profile == "minimal":
        out.pop("n_splits", None)
    return out


def fit_encoding(
    session: RunSession,
    columns: list[str],
    method: str = "frequency",
    table: str | None = None,
    categories: dict[str, list] | None = None,
    smoothing: float = 10.0,
    stats: list[str] | None = None,
) -> dict:
    """Encode categorical columns into a new feature table (spec §5.2).

    The only way to make a non-numeric column usable: every fit in this harness
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

    ``stats`` widens ``method="target"`` past the mean. The mean is a lossy
    summary of a group — two categories with the same average can have very
    different target distributions, and an average standing on three rows should
    not be read like one standing on three hundred — so the group's spread,
    middle and size are available as further columns. Only the mean is smoothed;
    ``count`` is what lets the model discount a thin group by itself. See
    :class:`~tabular_agent.encoders.TargetEncoder`.
    """
    refusal = _refuse_if_finished(session)
    if refusal is not None:
        return refusal

    path = session.resolve_table(table)
    spec = session.resolved()
    df = load_table(path)

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in {session.rel(path)}: {missing}")
    if spec.target in columns:
        raise ValueError(f"refusing to encode the target column {spec.target!r} as a feature")

    if stats and method != "target":
        raise ValueError(
            f"stats={list(stats)} applies to method='target' only — it summarises the "
            f"target's distribution within each category, and method={method!r} never "
            "looks at the target. For per-group statistics of some *other* column, use "
            "fit_group_stats."
        )

    kwargs = {"smoothing": smoothing, "categories": categories}
    if method == "target":
        if stats is not None:
            unknown = [s for s in stats if s not in TARGET_STATS]
            if unknown:
                raise ValueError(
                    f"unknown target statistic(s) {unknown}; use one or more of "
                    f"{list(TARGET_STATS)}"
                )
            kwargs["stats"] = list(stats) or ["mean"]
        y = df[spec.target]
        # Same seed and n_splits as training, so the out-of-fold values in the
        # table were held out of exactly the folds the model is scored on.
        kwargs["y"] = y
        kwargs["folds"] = make_folds(y, spec.task_type, n_splits=spec.n_splits,
                                     seed=spec.seed, order=_fold_order(session, df))
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
    if method == "target":
        # On the row rather than only in the lineage, so `explore_journal` can
        # tell two target encodings of the same columns apart — they differ only
        # by which statistics were asked for.
        row["target_stats"] = list(encoder.stats)
    scored = _probe_and_compare(session, out_path, row["source_table"])
    row.update(scored)
    journal.append_table(session.run_dir, row)

    _, still_dropped = _feature_frame(out, spec.target)
    return {
        "table": row["path"],
        "method": method,
        "new_columns": new_columns,
        "n_rows": int(len(out)),
        "source_table": row["source_table"],
        "non_numeric_columns": still_dropped,
        **_agent_score_fields(session, scored),
        "note": (
            "This table is scored above; build on it to keep the new columns. "
            + (
                "The values stored here are out-of-fold; the encoder fitted on all "
                "rows is kept in the table's lineage and is what runs at predict time."
                if method == "target"
                else "The fitted encoder is kept in the table's lineage and replays at "
                "predict time."
            )
        ),
    }


def fit_group_stats(
    session: RunSession,
    by: list[str],
    value: str,
    stat: str = "mean",
    table: str | None = None,
) -> dict:
    """Summarise a column per group into a new feature table (the fitted aggregate).

    "What is typical for the group this row belongs to" — the average for this
    customer, the median for this route, how busy this location usually is. The
    obvious way to write it, ``groupby(...).transform("mean")`` inside
    ``transform_features``, is refused and should be: that recomputes the number
    from whatever rows are in the frame, so one row alone gets a different
    feature than the same row in a batch. Here the group's number is fitted once
    and **stored in the table's lineage**, so the lookup is identical at any batch
    size.

    Always out-of-fold, with no way to turn it off. A value column correlated
    with the target makes its per-group mean nearly a per-group mean of *the
    target*, and including the row's own contribution would be target encoding
    with the guard off. Whether that matters depends on how many rows a group has
    — which is a property of the data, not something this tool can check — so the
    honest default is the safe one.

    The value column does not have to exist at predict time; only the ``by``
    columns are read at lookup. That is the point: a quantity measurable in
    history but not on the day you predict is usable *as its group's history*.
    It is still not usable as itself, so drop it before training if the run's
    serving schema says it will not be there.
    """
    refusal = _refuse_if_finished(session)
    if refusal is not None:
        return refusal

    path = session.resolve_table(table)
    spec = session.resolved()
    df = load_table(path)

    by = list(by)
    missing = [c for c in [*by, value] if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in {session.rel(path)}: {missing}")
    if value == spec.target:
        raise ValueError(
            f"{value!r} is the target column. A per-group statistic of the target is "
            "target encoding, which has its own tool because the mean of the target "
            "needs smoothing as well as out-of-fold fitting: "
            "fit_encoding(columns=[...], method='target'). That tool takes the same "
            f"statistics this one does — fit_encoding(columns={by}, method='target', "
            f"stats=['{stat}']) is what you are asking for here."
        )
    if spec.target in by:
        raise ValueError(
            f"refusing to group by the target column {spec.target!r}: the group key "
            "has to exist at predict time and the target never does"
        )
    # Checked here rather than at training, because this failure is about the
    # *key*, not the feature. A group statistic keyed on a column that will not
    # exist can never be looked up, so the table would be built, trained on, and
    # refused as unservable three calls later for a reason stated here in one.
    unavailable_keys = [
        c for c in by if c in _unavailable_at_predict_time(session, df)
    ]
    if unavailable_keys:
        raise ValueError(
            f"cannot group by {unavailable_keys}: these columns will not exist when a "
            "prediction is asked for, so the stored statistics could never be looked "
            "up. Group by a column that will be there."
        )

    folds = make_folds(df[spec.target], spec.task_type, n_splits=spec.n_splits,
                       seed=spec.seed, order=_fold_order(session, df))
    train_features, encoder = _fit_group_stats(df, by, value, stat, folds=folds)

    out = df.join(train_features)
    table_id = journal.next_table_id(session.run_dir)
    out_path = session.ctx.path("tables", f"{table_id}.parquet")
    save_parquet(out, out_path)

    lineage = _load_lineage(path, spec.task_type)
    lineage.add_encoder(encoder)
    with open(_lineage_path(out_path), "w") as f:
        json.dump(lineage.to_dict(), f, indent=2)

    new_columns = list(train_features.columns)
    scored = _probe_and_compare(session, out_path, session.rel(path))
    journal.append_table(session.run_dir, {
        "id": table_id,
        "path": session.rel(out_path),
        "source_table": session.rel(path),
        "method": "group_stats",
        "group_by": by,
        "value_column": value,
        "stat": stat,
        "new_columns": new_columns,
        "n_steps": len(lineage.steps),
        **scored,
    })

    column = new_columns[0]
    n_missing = int(out[column].isna().sum())
    value_unavailable = value in _unavailable_at_predict_time(session, df)
    note = (
        "This table is scored above; build on it to keep the new column. The values "
        "here are out-of-fold; the statistics fitted on all rows are kept in the "
        "table's lineage and are what a prediction looks up."
    )
    if value_unavailable:
        note += (
            f" {value!r} itself will not exist at predict time, so training on this "
            f"table as it stands is refused — build one without it first: "
            f"transform_features with df.drop(columns=['{value}'], errors='ignore'). "
            f"The new column survives that, because it no longer needs {value!r}."
        )
    return {
        "table": session.rel(out_path),
        "new_columns": new_columns,
        "n_groups": len(encoder.mapping_),
        "n_rows": int(len(out)),
        "source_table": session.rel(path),
        **_agent_score_fields(session, scored),
        # Out-of-fold values are missing wherever a row was in no fold's held-out
        # block, which under an ordered split is the whole first period. Said
        # plainly because the column looks broken otherwise, and the models
        # handle it — this is a real "no estimate", not a zero.
        "n_rows_without_value": n_missing,
        "note": note,
    }


def fit_history(
    session: RunSession,
    by: list[str],
    value: str,
    keep: int | None = None,
    table: str | None = None,
    windows: list | None = None,
) -> dict:
    """Store each group's recent past and replay it as trailing averages.

    The level a group has been running at lately — the one thing about a group
    that depends on *when* as well as *how much*, which `fit_group_stats` and
    `fit_encoding` both average away. Unlike those, this may point at the target:
    a summary of the target's own past can only ever come from stored history,
    never from a served frame, because the target is not in one.

    `windows=[[90, "mean"]]` adds the mean of every observation in the 90 periods
    before each row, beside a count of how many that was. At least one window is
    required.

    **Single-row features are not built here** (0072). There is no lag argument and
    no last-known-value pair, because both are the value of one row, and one row's
    value carries one row's noise where an average over the stretch does not. The
    concrete failure is this tool's own freeze: a single stored observation is a
    per-group **constant** for every row you predict, while the same column varies
    across the folds you train on — so a model learns a feature it is then served
    as a constant. Two datasets priced that at 0.014 rmspe and 0.002 rmsle against
    the table that would otherwise have shipped, and on the first of them the one
    draw fell on a public holiday when 98% of the groups were shut.

    A seasonal value is not lost, only expressed differently and more stably: put
    the season in the **key** rather than in a distance. Group by the entity *and*
    its position in the cycle, then take a window spanning several cycles, so the
    feature averages the comparable periods instead of reading the single one a
    fixed step back.

    The ordering column is the one the run is split on, and there has to be one.
    History frozen at a training fold scattered through time reports an age that
    depends on the fold seed rather than the data, which is why an unordered run
    is refused here rather than served a number that means nothing.
    """
    refusal = _refuse_if_finished(session)
    if refusal is not None:
        return refusal

    path = session.resolve_table(table)
    spec = session.resolved()
    df = load_table(path)

    by = list(by)
    if session.split_on is None:
        raise ValueError(
            "this run is not split on an ordering column, so there is no past to "
            "store: 'the previous observation for this group' has no meaning when the "
            "rows are exchangeable. History is for runs whose task is to predict "
            "later rows from earlier ones."
        )
    order = session.split_on
    missing = [c for c in [*by, value, order] if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in {session.rel(path)}: {missing}")
    if spec.target in by:
        raise ValueError(
            f"refusing to group by the target column {spec.target!r}: the group key "
            "has to exist at predict time and the target never does"
        )
    unavailable_keys = [
        c for c in [*by, order] if c in _unavailable_at_predict_time(session, df)
    ]
    if unavailable_keys:
        raise ValueError(
            f"cannot key history on {unavailable_keys}: these columns will not exist "
            "when a prediction is asked for, so the stored history could never be "
            "looked up. The ordering column in particular has to be there — without "
            "it a served row has no position, and age is a distance from one."
        )

    folds = make_folds(df[spec.target], spec.task_type, n_splits=spec.n_splits,
                       seed=spec.seed, order=_fold_order(session, df))
    train_features, encoder = _fit_history(
        df, by, value, order, folds=folds, keep=keep,
        forecast_horizon=spec.forecast_horizon, windows=windows or (),
    )

    out = df.join(train_features)
    table_id = journal.next_table_id(session.run_dir)
    out_path = session.ctx.path("tables", f"{table_id}.parquet")
    save_parquet(out, out_path)

    lineage = _load_lineage(path, spec.task_type)
    lineage.add_encoder(encoder)
    with open(_lineage_path(out_path), "w") as f:
        json.dump(lineage.to_dict(), f, indent=2)

    new_columns = list(train_features.columns)
    scored = _probe_and_compare(session, out_path, session.rel(path))
    journal.append_table(session.run_dir, {
        "id": table_id,
        "path": session.rel(out_path),
        "source_table": session.rel(path),
        "method": "history",
        "group_by": by,
        "value_column": value,
        "order_column": order,
        "windows": [[w, s] for w, s in encoder.windows],
        "new_columns": new_columns,
        "n_steps": len(lineage.steps),
        **scored,
    })

    n_missing = int(out[new_columns[0]].isna().sum())
    periods = _to_periods(df[order])
    # How wide a block the run is asked to predict, in the ordering column's
    # units. The agent cannot work this out for itself — `inspect_data` reports
    # `n_val` as a *row count*, and nothing anywhere converts that to a span — so
    # telling it "use a window longer than the period you predict" was advice it
    # had no way to act on. The tool knows, so the tool says.
    spans = [
        float(np.nanmax(periods[va]) - np.nanmin(periods[va]))
        for _, va in folds if len(va)
    ]
    block_span = round(float(np.median(spans)), 3) if spans else None

    # What a window actually averaged. A window is a summary only if there was
    # something in it: two observations under a name that says "90 periods" is a
    # single draw wearing different clothes, and the median across rows is the
    # number that shows it at a glance.
    window_counts = {}
    for w in encoder.window_lengths:
        seen = train_features[f"{encoder.base_name}_{w:g}_count"].dropna()
        if len(seen):
            window_counts[f"{w:g}"] = round(float(np.median(seen)), 3)
    thin = [k for k, v in window_counts.items() if v < 2]

    # The shortest training prefix any fold gets, and the reason it is reported.
    #
    # Under forward chaining the first fold trains on a short head of the data, so
    # a window longer than that prefix **cannot be filled there** — while at
    # serving time the whole history is available and it fills completely. The
    # model then learns from a mean over few observations and is served a mean over
    # many: the same column name, a different and less noisy quantity, which is
    # train/serve skew of exactly the kind 0048 exists to prevent.
    #
    # **The count sees only one side of a two-sided quantity**, which is why it is
    # reported next to the prefixes rather than alone: on Rossmann a 728-day window
    # reported *more* observations (59) than a 364-day one (39) while behaving far
    # worse under one fold count. The median also hides shape — 59 was the midpoint
    # of a per-fold ramp from 20 to 93, not a typical value.
    prefixes = [
        float(np.nanmax(periods[tr]) - np.nanmin(periods[tr]))
        for tr, _ in folds if len(tr)
    ]
    shortest_prefix = round(float(min(prefixes)), 3) if prefixes else None
    longest_prefix = round(float(max(prefixes)), 3) if prefixes else None
    # Reported as facts about the fold geometry, and **deliberately without a
    # claim about what they cost**, because the obvious claim was tested and
    # failed (0053).
    #
    # The arithmetic is not in doubt: past the shortest prefix the earliest folds
    # cannot fill the window, and past the longest one no fold can, so the trained
    # column and the served column summarise different numbers of observations.
    # Measured on Rossmann at w=728: per-fold medians ramping 20/39/59/78/93
    # against a served 101.
    #
    # What does *not* follow is that this is what damages a model. The same window
    # at `n_splits=3` has the identical train/serve ratio (59 against 101, 1.71x)
    # and scores **+0.047 where `n_splits=5` scores −0.093**; and w=546 (1.29x)
    # scores −0.000 where w=364 (1.26x) scores +0.045. Across the pairs the ratio
    # is *anti*-correlated with the score. The fold scheme moves these features by
    # a lot, and this is not the route.
    #
    # So the numbers are surfaced — the agent cannot obtain them otherwise, which
    # is the same argument `block_span` rests on — and the note describes the
    # geometry without predicting a cost it cannot predict.
    beyond_some = [
        f"{w:g}" for w in encoder.window_lengths
        if shortest_prefix is not None and w > shortest_prefix
    ]
    beyond_all = [
        f"{w:g}" for w in encoder.window_lengths
        if longest_prefix is not None and w > longest_prefix
    ]
    beyond_some = [w for w in beyond_some if w not in beyond_all]

    note = (
        "This table is scored above; build on it to keep the new columns. Each "
        "fold's history stops at the end of that fold's training rows, so a "
        "validation row reads exactly what a served row the same distance ahead "
        "would read — the values here get older across each block, on purpose. "
        "Expect this to score lower than a lag that peeks; the difference is what "
        "peeking was worth, not what this feature costs. Rows in the first period "
        "have no earlier block to read and keep a missing value."
    )
    if block_span is not None:
        note += (
            f" Each block you are scored on spans about {block_span:g} in "
            f"{order!r}, so a window has to reach well past that before the rows at "
            "the far edge of the block have anything to average."
        )
    if thin:
        note += (
            f" WARNING: window(s) {thin} held a median of fewer than two "
            "observations — see window_observations. An average over one point "
            "is that point: the column has collapsed into a single stored "
            "observation under a name that promises a summary, which is the whole "
            "thing 0072 refuses. A window has to reach well past the horizon you "
            "forecast before it contains anything to average."
        )
    if beyond_some:
        note += (
            f" NOTE: window(s) {beyond_some} are longer than the shortest training "
            f"prefix ({shortest_prefix:g} in {order!r}), so the earliest folds "
            "cannot fill them while later folds and a served row can."
        )
    if beyond_all:
        note += (
            f" NOTE: window(s) {beyond_all} are longer than every fold's training "
            f"prefix (the longest is {longest_prefix:g} in {order!r}), so no fold "
            "fills them while a served row does. Each fold's column then summarises "
            "a different number of observations from the column you will serve. "
            "Whether that costs anything is not settled — on Rossmann the same "
            "window was worth +0.047 under one fold count and −0.093 under "
            "another, with identical counts on both sides — so treat window length "
            "as a parameter this harness cannot yet advise you on, and read the "
            "probe score rather than these numbers."
        )

    return {
        "table": session.rel(out_path),
        "new_columns": new_columns,
        "n_groups": len(encoder.history_),
        "n_rows": int(len(out)),
        "source_table": session.rel(path),
        **_agent_score_fields(session, scored),
        "n_rows_without_value": n_missing,
        # {window: median number of observations that fell inside it}. Read it
        # beside `shortest_training_prefix`: on its own it sees only the training
        # side, and a window too long to fill reports a healthy-looking count.
        "window_observations": window_counts,
        # The narrowest and widest history any fold was fitted on. Past the first,
        # the earliest folds are starved; past the second, no fold can fill the
        # window at all and its per-fold counts ramp instead of settling.
        "shortest_training_prefix": shortest_prefix,
        "longest_training_prefix": longest_prefix,
        # In the ordering column's units, so an offset or a window length can be
        # chosen against it.
        "block_span": block_span,
        "note": note,
    }


def fit_stream_aggregate(
    session: RunSession,
    stream: str,
    by: list[str],
    value: str,
    leads: list[float],
    stats: list[str] | None = None,
    table: str | None = None,
) -> dict:
    """Aggregate a declared stream per row, counting only what was knowable in time.

    The tool for a **second input**: a related file at a finer grain than the
    table, whose records carry both the time they are *about* and the time they
    became *knowable*. Several records collapse into one number per row, and the
    obvious collapse keeps the first time and discards the second — which builds a
    column that is systematically richer in training than at serving, by an amount
    that grows with the horizon, and that nothing else in this harness can see.

    Each ``lead`` is a separate feature, not a competing setting: ``leads=[7, 30]``
    answers both "what could a forecast issued a week ahead see" and "what could
    one issued a month ahead see", and the model may use either.

    Every statistic comes with a ``_count``, always. A sum over one record and a
    sum over forty are the same number wearing the same name, and the count is
    what says which one a row got.

    Leads shorter than the declared ``forecast_horizon`` are refused. Not because
    they would leak — the restriction is backward from each row either way — but
    because no served row could ever have one, so the column would be built out of
    knowledge the forecast does not get.
    """
    refusal = _refuse_if_finished(session)
    if refusal is not None:
        return refusal

    path = session.resolve_table(table)
    spec = session.resolved()
    df = load_table(path)
    declared = session.stream_spec(stream)
    stream_df = load_table(session.stream_path(stream))

    by = list(by)
    leads = [float(x) for x in leads]
    stats = list(stats) if stats else ["sum"]

    if session.split_on is None:
        raise ValueError(
            "this run is not split on an ordering column, so 'what was knowable a "
            "week before this row' has no meaning — the rows are exchangeable and "
            "none of them is before another. A stream aggregate is for runs whose "
            "task is to predict later rows from earlier ones."
        )
    order = session.split_on
    if not leads:
        raise ValueError(
            "leads must name at least one lead time. Without one this is the naive "
            "collapse of the stream, which counts records no forecast could have "
            "seen — the thing this tool exists to avoid."
        )
    if spec.forecast_horizon is None:
        raise ValueError(
            f"leads={leads} need the task to declare how far ahead it forecasts. Set "
            "TaskSpec.forecast_horizon to the gap between the end of history and the "
            f"earliest row you are asked to predict, in {order!r}'s units. Without it "
            "there is no way to tell a lead a forecast could actually be issued at "
            "from one that reads records nobody would have had."
        )
    too_short = [x for x in leads if x < spec.forecast_horizon]
    if too_short:
        raise ValueError(
            f"leads {too_short} are shorter than the declared forecast horizon of "
            f"{spec.forecast_horizon:g}. Every row you are scored on sits at least "
            "that far past the end of history, so a record observed later than that "
            "cannot exist for any of them — the column would be built from knowledge "
            "the forecast never gets, and validation would confirm it because "
            f"validation is drawn from the same file. Use leads of at least "
            f"{spec.forecast_horizon:g}."
        )
    bad_stats = [s for s in stats if s not in STREAM_STATS]
    if bad_stats:
        raise ValueError(
            f"unknown statistic(s) {bad_stats}; pick from {list(STREAM_STATS)}"
        )

    missing_table = [c for c in [*by, order] if c not in df.columns]
    if missing_table:
        raise ValueError(f"columns not in {session.rel(path)}: {missing_table}")
    # The key has to name the same columns on both sides, and saying which side is
    # missing them is the whole of the message: "no rows matched" sends a caller
    # looking at the data when the answer is a column name.
    missing_stream = [c for c in [*by, value] if c not in stream_df.columns]
    if missing_stream:
        raise ValueError(
            f"columns not in stream {stream!r}: {missing_stream}. It has "
            f"{list(stream_df.columns)}. The grouping columns are matched by name "
            f"across both files; the table has {[c for c in by if c in df.columns]}."
        )
    if spec.target in by:
        raise ValueError(
            f"refusing to group by the target column {spec.target!r}: the group key "
            "has to exist at predict time and the target never does"
        )
    unavailable_keys = [
        c for c in [*by, order] if c in _unavailable_at_predict_time(session, df)
    ]
    if unavailable_keys:
        raise ValueError(
            f"cannot key a stream aggregate on {unavailable_keys}: these columns will "
            "not exist when a prediction is asked for, so the stream could never be "
            "attached to a served row."
        )

    encoder = StreamAggregateEncoder(
        stream, by, value, declared["event_time"], declared["observed_at"], order,
        leads=leads, stats=stats,
    ).fit(df, stream=stream_df)
    # No fold loop, and that is the design rather than an omission. `fit_history`
    # re-freezes per fold because *last known* is defined relative to the fitting
    # set, so fitting it once on everything lets a validation row read the future.
    # Here the restriction is `observed_at <= row - lead`, which is backward from
    # each row individually: a row cannot reach past its own position whichever
    # fold it lands in, and the serving asymmetry a freeze would reproduce is
    # exactly what the lead already is.
    out = encoder.transform(df, stream_df)
    new_columns = encoder.output_columns

    table_id = journal.next_table_id(session.run_dir)
    out_path = session.ctx.path("tables", f"{table_id}.parquet")
    save_parquet(out, out_path)

    lineage = _load_lineage(path, spec.task_type)
    lineage.add_encoder(encoder)
    with open(_lineage_path(out_path), "w") as f:
        json.dump(lineage.to_dict(), f, indent=2)

    scored = _probe_and_compare(session, out_path, session.rel(path))
    journal.append_table(session.run_dir, {
        "id": table_id,
        "path": session.rel(out_path),
        "source_table": session.rel(path),
        "method": "stream_aggregate",
        "stream": stream,
        "group_by": by,
        "value_column": value,
        "order_column": order,
        "leads": leads,
        "stats": stats,
        "new_columns": new_columns,
        "n_steps": len(lineage.steps),
        **scored,
    })

    # What each lead actually left. This is the readout the whole tool turns on: a
    # lead the data cannot support does not fail, it quietly empties the
    # aggregate.
    #
    # **Two numbers on two different denominators, and saying which is which is
    # the point.** `rows_answered` is over *every* row of the table, so it is
    # directly comparable with `matched_rows` — the share the stream could reach
    # at best, before any lead. The gap between them is what this lead cost.
    # `records_per_row` is over the answered rows only, because a median across
    # all rows of a feed covering a tenth of them is 0 whatever the lead, which
    # says something about the feed and nothing about the choice being made.
    records_per_row = {}
    answered_share = {}
    for lead in leads:
        counts = out[encoder.count_column(lead)]
        answered = counts >= 1
        answered_share[f"{lead:g}"] = round(float(answered.mean()), 4)
        if answered.any():
            records_per_row[f"{lead:g}"] = round(float(counts[answered].median()), 3)
    starved = [k for k, v in answered_share.items() if v == 0.0]

    # How wide a scored block is, so a lead can be read against something. The
    # agent cannot get this anywhere else — `inspect_data` reports `n_val` as a row
    # count and nothing converts that to a span — and it is the number that shows
    # the limit below.
    periods = _to_periods(df[order])
    folds = make_folds(df[spec.target], spec.task_type, n_splits=spec.n_splits,
                       seed=spec.seed, order=_fold_order(session, df))
    spans = [
        float(np.nanmax(periods[va]) - np.nanmin(periods[va]))
        for _, va in folds if len(va)
    ]
    block_span = round(float(np.median(spans)), 3) if spans else None

    note = (
        "This table is scored above; build on it to keep the new columns. Each row "
        "counts only the stream records observed at least `lead` before it, so a "
        "training row is restricted exactly as a served row at the same lead would "
        f"be. Every statistic has a `_count` beside it — read them together, since "
        "a sum over one record and a sum over forty look identical otherwise."
    )
    if block_span is not None:
        note += (
            f" The horizon you declared ({spec.forecast_horizon:g} in {order!r}) is "
            f"the *floor* of a scored block, and a block spans about "
            f"{block_span:g}. A lead at the floor is therefore honest for the "
            "block's earliest rows and more generous than the far edge could ever "
            "be; a lead nearer the far edge is honest throughout and sees less. "
            "Both are legitimate and the counts say what each costs."
        )
    note += (
        f" Read rows_answered against matched_rows ({encoder.matched_rows_:.3f}): the "
        "second is the share of rows this stream could reach at all, the first is "
        "what survived the lead. The distance between them is what the restriction "
        "cost, and a large one is a finding rather than a fault — it is the size of "
        "the gap between what training could see and what a forecast can."
    )
    if starved:
        note += (
            f" WARNING: lead(s) {starved} answered no rows at all. The columns are "
            "constant and carry nothing; a shorter lead, or none of this stream, is "
            "the choice here."
        )

    return {
        "table": session.rel(out_path),
        "new_columns": new_columns,
        "stream": stream,
        "n_records": encoder.n_records_,
        # Distinct (key, event-time) buckets the stream forms — the granularity it
        # can actually answer at, which is not the number of rows in the table.
        "n_buckets": encoder.n_buckets_,
        # Share of this table's rows the stream has any bucket for at all. Low is
        # not an error — a feed covering a third of the shops is a real feed — but
        # it bounds what the feature can be worth before any lead is applied.
        "matched_rows": encoder.matched_rows_,
        # The bucket width measured off the table's own row spacing, in `order`'s
        # units, and stored in the lineage so predict time buckets identically.
        "event_grain": encoder.grain_,
        "n_rows": int(len(out)),
        "source_table": session.rel(path),
        **_agent_score_fields(session, scored),
        # Share of *all* rows this lead could put at least one record behind —
        # the number to compare against `matched_rows`.
        "rows_answered": answered_share,
        # {lead: median records behind a row that got any}. Conditioned on being
        # answered, so it says how much the aggregate summarises when it speaks,
        # rather than restating how sparse the feed is.
        "records_per_row": records_per_row,
        "block_span": block_span,
        "note": note,
    }


def filter_rows(
    session: RunSession,
    code: str,
    table: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Train on fewer rows: write a table with the rows a predicate keeps.

    The only step in the harness that is **not** replayed at prediction time, and
    it must not be — the validation half is scored by replaying lineage over its
    rows, and a filter there would drop some and misalign the score; a served
    model has to answer for every row it is given. So the rows are dropped once,
    here, and the lineage records the filter as train-time-only.

    What that buys: rows the real metric never scores, rows from a regime the
    model is not being asked to predict, and outliers whose squared error drowns
    out everything else, can all be kept out of the fit without touching how the
    model is graded.

    Unlike ``transform_features`` this code **can** see the target, and unlike
    ``transform_features`` it needs no stability check. Both follow from the same
    fact: the predicate is evaluated exactly once, against one frame, and never
    again. Nothing it computes is replayed on a row that has no target, and it
    cannot disagree with a version of itself run on a different batch.

    Only the work half is affected. The validation half is untouched, so
    ``val_score`` still measures the same exam and a filtered experiment stays
    comparable to an unfiltered one. ``cv_score`` does not: it is now the mean
    over the kept rows only.
    """
    refusal = _refuse_if_finished(session)
    if refusal is not None:
        return refusal

    path = session.resolve_table(table)
    spec = session.resolved()
    df = load_table(path)

    kept, summary = select_rows(df, code, timeout=timeout)

    if summary["n_dropped"] == 0:
        raise ValueError(
            f"this predicate kept all {summary['n_rows']} rows, so the table would be "
            "a copy of its source. Either the condition is not the one you meant, or "
            "there is nothing to drop here — train on the source table instead."
        )
    # A fold needs rows on both sides of it, and an ordered run needs one more
    # block than it has folds. Below that the table cannot be trained on at all,
    # and a table that exists only to be refused later is worse than a refusal now.
    floor = max(spec.n_splits + 1, MIN_FILTERED_ROWS)
    if summary["n_kept"] < floor:
        raise ValueError(
            f"this predicate kept {summary['n_kept']} of {summary['n_rows']} rows, which "
            f"is below the {floor} a {spec.n_splits}-fold cross-validation needs. Filtering "
            "removes rows from the fit, not from the problem — keep the condition to "
            "rows the model genuinely should not learn from."
        )

    table_id = journal.next_table_id(session.run_dir)
    out_path = session.ctx.path("tables", f"{table_id}.parquet")
    # Reset, because every other table in the run is written from a fresh index
    # and `train_cv` addresses folds positionally. A gappy index would still work
    # here and would be a trap the first time something aligned by label.
    save_parquet(kept.reset_index(drop=True), out_path)

    lineage = _load_lineage(path, spec.task_type)
    lineage.add_filter(code)
    with open(_lineage_path(out_path), "w") as f:
        json.dump(lineage.to_dict(), f, indent=2)

    scored = _probe_and_compare(session, out_path, session.rel(path))
    journal.append_table(session.run_dir, {
        "id": table_id,
        "path": session.rel(out_path),
        "source_table": session.rel(path),
        "method": "filter",
        "code": code,
        # A filter adds nothing, and the key has to be present anyway: every
        # reader of `tables` (attribution, columns_by_table, explore_journal)
        # reads `new_columns` and would otherwise report this table as missing
        # rather than as empty-by-construction.
        "new_columns": [],
        "n_rows": int(summary["n_kept"]),
        "n_dropped": int(summary["n_dropped"]),
        "n_steps": len(lineage.steps),
        **scored,
    })

    return {
        "table": session.rel(out_path),
        "n_rows": int(summary["n_kept"]),
        "n_rows_before": int(summary["n_rows"]),
        "n_dropped": int(summary["n_dropped"]),
        "fraction_dropped": round(summary["n_dropped"] / summary["n_rows"], 4),
        "source_table": session.rel(path),
        "new_columns": [],
        **_agent_score_fields(session, scored),
        "note": (
            "The kept rows are what anything built on this table fits on. The filter "
            "is recorded as train-time-only and does not run at predict time, so "
            "probe_score is still measured on the whole validation half and stays "
            "comparable to unfiltered tables."
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
    refusal = _refuse_if_finished(session)
    if refusal is not None:
        return refusal

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
    lineage.add_transform(code, new_columns=list(summary["new_columns"]))
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
    scored = _probe_and_compare(session, out_path, row["source_table"])
    row.update(scored)
    journal.append_table(session.run_dir, row)

    _, still_dropped = _feature_frame(out, spec.target)
    return {
        "table": row["path"],
        "new_columns": row["new_columns"],
        "n_rows": int(summary["n_rows"]),
        "n_cols": len(out.columns),
        "source_table": row["source_table"],
        "non_numeric_columns": still_dropped,
        **_agent_score_fields(session, scored),
        "note": (
            "This table is scored above; build on it to keep the new columns. The code "
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


def _fit_signature(model_type: str, params: dict | None,
                   target_transform: str = "none") -> str:
    """Identity of *how* a model was fitted, for deciding what compares to what.

    The target scale belongs in here for the same reason the params do: two
    experiments that differ by it differ by more than their features, and 0017's
    rule is that a comparison must differ by exactly one thing or it is not
    evidence about that thing.
    """
    return json.dumps([model_type, params or {}, target_transform], sort_keys=True)


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
    target_transform: str = "none",
) -> tuple[float | None, str]:
    """Gain over the same model, same params, same target scale, on the source table.

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
    signature = _fit_signature(model_type, params, target_transform)
    on_source = [
        r
        for r in j.get("leaderboard") or []
        if r.get("source_table") == source and r.get("val_score") is not None
    ]
    comparable = [
        r
        for r in on_source
        if _fit_signature(
            r.get("model_type"), r.get("params"), r.get("target_transform", "none")
        )
        == signature
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
    """The run's stop status, read from the journal's tables (0035)."""
    j = journal.load(session.run_dir)
    metric = (j.get("task") or {}).get("metric") or session.resolved().metric
    return stop.evaluate(j.get("tables") or [], metric, session.stop_policy)


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
    # so another table could only add a row nothing will ever read, or tempt a
    # second `finalize_run` against the same validation rows.
    if journal.stop_state(session.run_dir).get("finalized"):
        raise RuntimeError(
            "this run is finished: the harness has already packaged it and will not "
            "build on it again"
        )
    if not status["should_stop"] and not force:
        return None

    refusals = journal.record_refusal(session.run_dir)
    policy = session.stop_policy
    finalized = journal.stop_state(session.run_dir).get("finalized")

    if refusals >= policy.max_refusals and not finalized:
        final = finalize_run(session)
        return {
            "refused": True,
            "reason": status["reason"],
            "harness_finalized": True,
            "note": (
                "Asked to keep building after the run was over, so the harness "
                "packaged it. Nothing further will be built. Report what you found "
                "and stop."
            ),
            **final,
        }
    if finalized:
        raise RuntimeError(
            "this run is finished: the harness has already packaged it and will not "
            f"build on it again ({status['reason']})"
        )

    return {
        "refused": True,
        "reason": status["reason"],
        "n_tables": status["n_tables"],
        "since_improvement": status["since_improvement"],
        "next": "finalize_run",
        "note": stop.advisory(status),
    }


def train_tabular_model(
    session: RunSession,
    table: str | None = None,
    model_type: str = "lgbm",
    params: dict | None = None,
    target_transform: str | None = None,
    enforce_stop: bool = True,
    seed: int | None = None,
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
    spec = session.resolved()
    # `None` means "whatever the run resolved", which since 0035 is derived from
    # the metric rather than chosen per experiment. An explicit value still wins,
    # because the baseline and the tests both need to pin it.
    if target_transform is None:
        target_transform = spec.target_transform
    # The *model's* seed, which is not the run's exam. Folds below still use
    # `spec.seed`, because which rows are held out is the harness's business
    # (0029) and must not move between fits. This exists so the terminal phase
    # can reproduce the probe exactly — see `_terminal_fits`.
    model_seed = spec.seed if seed is None else seed
    journal.set_split(session.run_dir, session.split_record())
    df = load_table(path)

    X, dropped = _feature_frame(df, spec.target)
    if X.shape[1] == 0:
        raise ValueError(
            "no numeric feature columns available to train on; use transform_features "
            "or fit_encoding to produce some first"
        )
    # Before the fit, not after: a refusal that arrives once the model is trained
    # has already spent the experiment it was meant to save.
    if enforce_stop:
        refusal = _refuse_if_unservable(session, path, df, list(X.columns))
        if refusal is not None:
            return refusal

    y = df[spec.target]

    folds = make_folds(y, spec.task_type, n_splits=spec.n_splits, seed=spec.seed,
                       order=_fold_order(session, df))
    result = train_cv(
        X,
        y,
        folds,
        spec.task_type,
        spec.metric,
        model_type=model_type,
        params=params,
        seed=model_seed,
        target_transform=target_transform,
        # `None` unless this run set one. Raises `TrainingTooSlow` (a ValueError,
        # so the agent gets a correction rather than the run ending) and records
        # nothing — an abandoned fit has no score to put on the leaderboard.
        time_budget_s=session.fit_time_budget_s,
    )

    cv_score = float(np.mean(result.fold_metrics))
    fold_spread = float(np.std(result.fold_metrics))
    val_score, val_se, val_unavailable, _ = _score_on_val(
        session, path, [result.final_model], result.feature_names, spec, y, target_transform
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
        "target_transform": target_transform,
        "task_type": spec.task_type,
        "metric": spec.metric,
        "cv_score": cv_score,
        "val_score": val_score,
        "val_se": val_se,
        "fit_seconds": result.fit_seconds,
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
        # On the row, not just in meta.json: it is a dimension an experiment can
        # differ by, so `vs_source_table`'s "differ by exactly one thing" rule
        # (0017) has to be able to see it.
        "target_transform": target_transform,
        "features": list(result.feature_names),
        # Which feature table this was trained on, so a resuming agent can read
        # the leaderboard alone and know what to reuse or beat — without opening
        # each experiment's meta.json.
        "source_table": session.rel(path),
        "metric": spec.metric,
        "val_score": val_score,
        "val_unavailable": val_unavailable,
        # The yardstick every comparison on this row is judged against (0022).
        # `fold_spread` stays below because it is still the honest answer to "how
        # much did this model wobble across folds" — it simply stopped being the
        # ruler, since nothing has been ranked by `cv_score` since 0019.
        "val_se": val_se,
        "cv_score": cv_score,
        "fold_spread": fold_spread,
        # On the row so a run can be asked what its experiments *cost*, not only
        # what they scored. Nothing else in the journal measures time, and the one
        # time it mattered it had to be reconstructed from file timestamps.
        "fit_seconds": result.fit_seconds,
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
        session, j_after, row["source_table"], val_score, spec.metric, model_type, params,
        target_transform,
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
        # Beside the score it qualifies, always — not only when `note` warns.
        # A validation score reported bare invites comparison digit by digit,
        # which is the failure 0022 exists to record: a run once celebrated a
        # 0.008 gain that was worth 0.0007, on a score whose own standard error
        # was 0.012. This is the one field that says how many digits are real.
        "val_se": val_se,
        "cv_score": cv_score,
        "fold_spread": fold_spread,
        # What this experiment cost in wall clock. Reported because the agent is
        # the only party that can choose something cheaper, and it cannot weigh a
        # score against a cost it is never shown.
        "fit_seconds": result.fit_seconds,
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
        "note": stop.advisory(status),
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
    target_transform: str = "none",
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
    # `None` means "whatever the run resolved", which since 0035 is derived from
    # the metric rather than chosen per experiment. An explicit value still wins,
    # because the baseline and the tests both need to pin it.
    if target_transform is None:
        target_transform = spec.target_transform
    journal.set_split(session.run_dir, session.split_record())
    df = load_table(path)

    X, dropped = _feature_frame(df, spec.target)
    if X.shape[1] == 0:
        raise ValueError(
            "no numeric feature columns to explain; use transform_features or "
            "fit_encoding to produce some first"
        )
    y = df[spec.target]
    folds = make_folds(y, spec.task_type, n_splits=spec.n_splits, seed=spec.seed,
                       order=_fold_order(session, df))

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
            target_transform=target_transform,
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


#: Columns of the `lb` frame, declared rather than inferred from whatever rows
#: happen to exist. An empty run must hand over a *well-formed* empty frame:
#: `lb[lb.val_score < 0.2]` should return nothing, not raise KeyError, or the
#: agent's first call on a fresh run fails for a reason that has nothing to do
#: with what it asked.
_LB_COLUMNS = [
    "id", "model_type", "params", "target_transform", "source_table", "metric",
    "val_score", "val_se", "cv_score", "fold_spread", "vs_baseline", "vs_source_table",
    "n_features", "fit_seconds",
]

_TABLE_COLUMNS = [
    "id", "path", "method", "source_table", "n_columns", "columns", "code", "n_dropped",
    # Every table carries a score now (0035): the ruler runs on creation, so this
    # frame is the run's feature search, complete, and `tables.sort_values(
    # "probe_score")` is the question the search is actually asking.
    "probe_score", "probe_se", "probe_vs_source_table",
    # The difference's own uncertainty (0036), flattened out of the nested record
    # for the same reason `params` is a JSON string: a dict cell is unhashable and
    # cannot be grouped, sorted or compared, which is all anyone wants to do with
    # it. The full record stays on the journal row.
    "probe_vs_source_se", "probe_vs_source_mode", "probe_vs_source_sign_stable",
]


def _interval_columns(t: dict) -> dict:
    """The scalar view of a table's difference interval, for `explore_journal`.

    ``probe_vs_source_se`` deliberately carries a *different* quantity in each
    mode — the paired standard error across resampled rows on a random split, the
    standard deviation across consecutive blocks on an ordered one — because in
    each case it is the spread of the same decision quantity, `vs_source_table`,
    over the thing that actually varies there. ``probe_vs_source_mode`` is beside
    it so nobody has to guess which, and comparing the two across split types is
    the reader's error to avoid rather than a number to reconcile.
    """
    rec = t.get("probe_vs_source_interval") or {}
    se = rec.get("paired_se") if rec.get("mode") == "paired_bootstrap" else rec.get("block_sd")
    return {
        "probe_vs_source_se": se,
        "probe_vs_source_mode": rec.get("mode"),
        # Ordered splits only: a random split has no periods to disagree.
        "probe_vs_source_sign_stable": rec.get("sign_stable"),
    }


def _journal_frames(session: RunSession) -> dict:
    """The run's own records as frames: what happened, not what it happened to.

    Deliberately **no data**. Every value here was either produced by a tool the
    agent called or written by the agent itself, so there is nothing to gate: no
    rows, no target values, and no per-column statistic conditioned on the
    target. That is what makes this the cheap sibling of `explore_table` —
    it needs no leakage argument at all, because it never touches the table.

    ``params`` is a JSON *string* rather than a dict. A dict cell is unhashable,
    so `lb.groupby("params")` — one of the two or three things worth doing with
    this frame — would raise; as a string it groups, sorts, and is still one
    `json.loads` away from the values.
    """
    j = journal.load(session.run_dir)

    lb = pd.DataFrame([
        {
            "id": r.get("id"),
            "model_type": r.get("model_type"),
            "params": json.dumps(r.get("params") or {}, sort_keys=True),
            "target_transform": r.get("target_transform", "none"),
            "source_table": r.get("source_table"),
            "metric": r.get("metric"),
            "val_score": r.get("val_score"),
            "val_se": r.get("val_se"),
            "cv_score": r.get("cv_score"),
            "fold_spread": r.get("fold_spread"),
            "vs_baseline": r.get("vs_baseline"),
            "vs_source_table": r.get("vs_source_table"),
            "n_features": len(r.get("features") or []),
            # Null on rows written before this was recorded.
            "fit_seconds": r.get("fit_seconds"),
        }
        for r in j.get("leaderboard") or []
    ], columns=_LB_COLUMNS)

    tables = pd.DataFrame([
        {
            "id": t.get("id"),
            "path": t.get("path"),
            "method": t.get("method"),
            "source_table": t.get("source_table"),
            # `new_columns` is what the table row actually carries — one tool
            # call produces one table, so this is precisely that call's output.
            # Reading a `columns` key here silently yielded [] for every table.
            "n_columns": len(t.get("new_columns") or []),
            "columns": t.get("new_columns") or [],
            "code": t.get("code"),
            # Only a `filter` table drops rows, so this is null everywhere else —
            # which is also how to find the filters. Without it a filter looks
            # like a table that added no columns, i.e. like a wasted call.
            "n_dropped": t.get("n_dropped"),
            # Null on tables written before the ruler existed, and on tables it
            # could not score — a missing number here is not a bad score.
            "probe_score": t.get("probe_score"),
            "probe_se": t.get("probe_se"),
            "probe_vs_source_table": t.get("probe_vs_source_table"),
            **_interval_columns(t),
        }
        for t in j.get("tables") or []
    ], columns=_TABLE_COLUMNS)

    notes = pd.DataFrame(j.get("log") or [])

    # Scalars the frames cannot express, and which decide how to read them —
    # notably the metric's direction, without which every comparison is a guess.
    run = {
        "task": j.get("task"),
        "split": j.get("split"),
        "stop": j.get("stop"),
        "best": j.get("best"),
        "baseline": j.get("baseline"),
        "higher_is_better": (
            greater_is_better(j["task"]["metric"]) if j.get("task") else None
        ),
    }
    return {"lb": lb, "tables": tables, "notes": notes, "run": run}


def explore_journal(session: RunSession, code: str, timeout: float = 10.0) -> dict:
    """Read-only code over the run's own records — the leaderboard, not the rows.

    The problem this solves is arithmetic by eye. By experiment 20 the agent is
    comparing twenty summaries it has read one at a time, each still sitting in
    its context, and answering "did target encoding ever actually help" by
    impression. ``lb.groupby("method")["val_score"].mean()`` answers it exactly,
    and the nineteen experiments not currently being thought about stay on disk.

    Needs no leakage argument, unlike `explore_table`: it never sees a row.
    Everything in these frames was produced by a tool the agent called or written
    by the agent itself.

    Free — no leaderboard row, no budget — and available after `finalize_run`,
    which is when "what did this run actually learn" is the whole question.
    """
    frames = _journal_frames(session)
    out = run_readonly(frames, code, func_name="explore", timeout=timeout)
    result = {"result": out["result"]}
    if out["stdout"]:
        result["stdout"] = out["stdout"]
    return result


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
                    n_splits: int, seed: int, order=None) -> pd.DataFrame:
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

    # -1 for a row no fold validated. Under forward chaining the earliest block is
    # never held out, and calling those rows "fold 0" would put them in a group
    # whose residuals were computed a completely different way.
    fold = np.full(len(out), -1, dtype=int)
    for i, (_, va) in enumerate(
        make_folds(y, task_type, n_splits=n_splits, seed=seed, order=order)
    ):
        fold[va] = i
    out["fold"] = fold
    return out





def explore_table(
    session: RunSession,
    table: str | None = None,
    code: str = "",
    timeout: float = 10.0,
) -> dict:
    """Read-only code over a table's rows and the ruler's out-of-fold errors.

    This closes the one arc of the loop that would otherwise be missing. The
    agent can inspect, build and score, but the step from a score back to the
    next hypothesis needs something to work with: a single number per table
    leaves it guessing which feature to try next from column names alone. Here it
    can ask *where* the model is wrong — which fold, which segment, which end of
    which distribution — and let the answer pick the next table.

    **This is the one place cross-validation is still run during a search.** The
    ranking number needs one fit (see `probe.py`); honest residuals need
    out-of-fold predictions, and there is no way around ``n_splits`` fits for
    that. It is charged to nothing and refused by nothing, but it is not free in
    wall clock, which is why it happens when asked for rather than on every
    table.

    The frame is the **whole source table**, not just the numeric columns the
    model trained on. Deliberate: the string columns a model cannot consume are
    exactly the ones worth grouping residuals by, and "the error is concentrated
    in one city" is how the agent discovers that a column is worth encoding at
    all. They cost nothing here because nothing here becomes a feature.

    Nothing it computes can be persisted. That is the whole guard, and it is the
    same one :mod:`~tabular_agent.explore` documents: the result is shown and
    discarded, so a per-segment target mean read here has no route back into the
    model except `fit_encoding`, which fits per fold.
    """
    spec = session.resolved()
    path = session.resolve_table(table)
    df = load_table(path)
    X, _ = _feature_frame(df, spec.target)
    if X.shape[1] == 0:
        raise ValueError(
            "no numeric feature columns on this table, so there is no model to "
            "take residuals from; build features first"
        )
    y = df[spec.target]
    folds = make_folds(y, spec.task_type, n_splits=spec.n_splits, seed=spec.seed,
                       order=_fold_order(session, df))
    # The same ruler the table was ranked with, so the residuals belong to the
    # number the agent is reacting to rather than to some other model.
    result = train_cv(
        X, y, folds, spec.task_type, spec.metric,
        model_type=PROBE_MODEL_TYPE, params=dict(PROBE_PARAMS), seed=PROBE_SEED,
        target_transform=spec.target_transform,
        time_budget_s=session.fit_time_budget_s,
    )
    oof = pd.DataFrame({"oof_pred": result.oof}) if result.oof.ndim == 1 else pd.DataFrame(
        result.oof, columns=[f"oof_pred_{c}" for c in range(result.oof.shape[1])]
    )
    frame = _residual_frame(
        df, oof, y, spec.task_type, spec.n_splits, spec.seed,
        order=_fold_order(session, df),
    )

    out = run_readonly({"df": frame}, code, func_name="explore", timeout=timeout)
    res = {"table": session.rel(path), "result": out["result"]}
    if out["stdout"]:
        res["stdout"] = out["stdout"]
    return res


#: Fitted on the winning table when the search is over, and never during it
#: (0035). Enumerated rather than searched: the set is small enough to train all
#: of, and spending search queries choosing among four things is the waste that
#: record exists to stop. `elasticnet` earns its place by being from a different
#: family — on house v8's winning table it scored 0.11217 against catboost's
#: 0.11207, and a linear model matching a boosted one says the signal is in the
#: features.
#:
#: **The probe's own fit must be in this set** — its model type, its parameters
#: and its seed. That is what makes the invariant hold: *the shipped model is
#: never worse than the number the search was ranked on*, because the very fit
#: that produced that number is a candidate. Anything less is a near-copy of the
#: ruler and not the ruler: v4 put the model *type* on the ballot with the run's
#: seed underneath, and the terminal lgbm came in 0.00279 of gini below the probe
#: on the table the probe had chosen.
#: Porto Seguro v3 measured a table at gini 0.29066 with the probe and shipped
#: elasticnet at 0.25389 — a loss three times the largest selection-noise effect
#: this project has recorded.
#:
#: Note what that episode is *not* evidence for. Refitted with comparable
#: capacity on that table, the families are close — xgb 0.27409 and catboost
#: 0.28351 against lgbm's 0.29066 — so the 0.08 spread at library defaults was
#: mostly `n_estimators=200` meaning very different things to three libraries,
#: not one family being better. Enumerating *defaults* measures the libraries;
#: comparing families fairly needs a common budget or early stopping, which is a
#: design and a record, not a constant picked off this table.
TERMINAL_MODEL_TYPES = (PROBE_MODEL_TYPE, "xgb", "catboost", "elasticnet")


def _best_table_path(session: RunSession, run_journal: dict) -> str:
    """The raw or derived table with the best fixed evaluation score."""
    metric = (run_journal.get("task") or {}).get("metric") or session.resolved().metric
    candidates = list(run_journal.get("tables") or [])
    work_probe = run_journal.get("work_probe") or {}
    if work_probe.get("probe_score") is not None:
        candidates.insert(0, {
            **work_probe,
            "id": "work",
            "path": session.rel(session.data_path),
        })
    winner = stop.best_table(candidates, metric)
    return winner["path"] if winner else session.rel(session.data_path)


def _terminal_fits(session: RunSession) -> list[dict]:
    """Fit the enumerated set on the best table, and leave them all on the board.

    Since 0044 this is **not** a selection event: `finalize_run` ships the
    weighted average of everything fitted here, so no candidate has to win.
    Since 0080 those weights are inverse to cross-fitting fold spread.
    The leaderboard still ranks them, because the ranking is the measurement —
    the argmax winner's sealed score against the blend's is the A/B this design
    was chosen on, and every run recomputes it.

    A model type that fails is skipped, not fatal: three candidates exist so that
    the run survives one of them being wrong for this data, and the blend simply
    averages whatever survived.
    """
    j = journal.load(session.run_dir)
    table = _best_table_path(session, j)

    done = []
    for model_type in TERMINAL_MODEL_TYPES:
        # The probe's candidate is the probe's *fit*, not merely its model type:
        # same parameters and same seed. Porto Seguro v4 is why the distinction
        # is spelled out — with the type alone on the ballot and the run's seed
        # underneath it, the terminal lgbm scored 0.00279 of gini below the probe
        # on the very table the probe had chosen, because lgbm moves 0.003 of auc
        # between seeds at that scale. A near-copy of the ruler is not the ruler.
        probe_fit = model_type == PROBE_MODEL_TYPE
        try:
            out = train_tabular_model(
                session, table=table, model_type=model_type, enforce_stop=False,
                params=dict(PROBE_PARAMS) if probe_fit else None,
                seed=PROBE_SEED if probe_fit else None,
            )
        except Exception as e:  # noqa: BLE001 - one bad family must not end the run
            done.append({"model_type": model_type, "trained": False, "error": str(e)})
            continue
        done.append(out)
    return done


BLEND_ID = "blend"


def _blend_members(session: RunSession, terminal: list[dict]) -> list[dict]:
    """The terminal fits that may be averaged together, with their models loaded.

    Averaging is only meaningful across models that predict the *same quantity
    from the same rows*, so a candidate joins only if it shares the first one's
    source table, feature list and target transform. Under 0035 they always do —
    `_terminal_fits` fits every family on one winning table — and the check is
    here because the day that stops being true, a silent mismatch would produce a
    bundle whose members disagree about what column three means.

    **Every candidate that trained, with no filtering by score.** Dropping the
    weak ones would be a selection, which is the cost 0044 exists to avoid
    paying; and the obvious filter — keep the boosted trees, drop the linear
    model — is refuted by house v9, where elasticnet was the best model on the
    board. Rossmann v7 is the case against: elasticnet sat 0.054 of rmspe behind
    the next-worst candidate and the four-way average still landed within 0.0001
    of the three-way one under the equal-weight rule then in force. Since 0080
    all members remain, but their pre-seal fold stability determines their weight.
    """
    members = []
    for out in terminal:
        exp_id = out.get("experiment_id")
        if not exp_id:
            continue
        exp_dir = session.ctx.path("experiments", exp_id)
        try:
            with open(os.path.join(exp_dir, "meta.json")) as f:
                meta = json.load(f)
            model = joblib.load(os.path.join(exp_dir, "final_model.joblib"))
        except Exception:  # noqa: BLE001 - a member that cannot be loaded is not a member
            continue
        members.append({"experiment_id": exp_id, "meta": meta, "model": model})

    if not members:
        return []
    head = members[0]["meta"]
    key = ("source_table", "target_transform")
    return [
        m for m in members
        if all(m["meta"].get(k) == head.get(k) for k in key)
        and list(m["meta"]["feature_names"]) == list(head["feature_names"])
    ]


def _score_on_holdout(
    session: RunSession,
    exp_ids: list[str],
    selected_id: str | None = None,
    blend: dict | None = None,
) -> dict | None:
    """Break the seal: score already-fitted experiments on the third split (0023).

    **The only reader of `session.holdout_path` in the library**, called once,
    from `finalize_run`, after the winner has already been picked. Nothing here
    can change which model ships — the argument is the list of experiments that
    were *already* chosen, and the return value goes to the journal and the
    report, never to `best`.

    Every terminal candidate is scored, and since 0044 the blend that ships is
    scored alongside them — ``selected`` marks the one the search crowned,
    ``shipped`` the one the bundle predicts with, and on a blended run those are
    different rows. Keeping both is what lets the argmax-versus-average question
    stay open across runs instead of being settled by the run that raised it.

    The fits exist either way, so the extra numbers are free, and they are the
    measurement three records in a row have wanted and could only get from a
    Kaggle submission:
    0039 and 0040 argued that optimism attaches to the *selected* measurement
    rather than to the model or the table, on three pairs that all happened to be
    lgbm-vs-elasticnet. A winner and its runners-up on the same table, scored on
    the same sealed rows, is that comparison without the confound.

    The rest of the harness must keep treating these as unrankable. They are
    written under `holdout`, not onto the leaderboard, and `selected` records
    which one the run had already crowned — so a later reader can tell the
    difference between "the number the search steered towards" and "the number
    that judged it".

    The models are **not refitted**. Each is scored exactly as it shipped, fitted
    on the work half alone, so the difference between its `val_score` and its
    holdout score is two sets of rows and nothing else. Refitting on work+val
    would produce a better model and a meaningless gap.
    """
    holdout_path = session.holdout_path
    if holdout_path is None:
        return None

    spec = session.resolved()
    scored = []
    for exp_id in exp_ids:
        row = _experiment_row(session.run_dir, exp_id)
        exp_dir = session.ctx.path("experiments", exp_id)
        entry = {
            "experiment_id": exp_id,
            "model_type": row.get("model_type"),
            "val_score": row.get("val_score"),
            # Which one the run had already crowned, on the row rather than by
            # position, so the distinction survives anything that reorders this
            # list. It is the variable 0039's hypothesis is about.
            "selected": exp_id == selected_id,
            # Set on every row, not only where it is true, so a reader can filter
            # on it without a missing key meaning two different things. Overwritten
            # below when there is no blend and the crowned candidate ships itself.
            "shipped": False,
        }
        try:
            with open(os.path.join(exp_dir, "meta.json")) as f:
                meta = json.load(f)
            model = joblib.load(os.path.join(exp_dir, "final_model.joblib"))
            source_path = session.resolve_table(meta["source_table"])
            y_work = load_table(source_path)[spec.target]
            score, se, why_not, _ = _score_on_held_out(
                session, holdout_path, source_path, [model], meta["feature_names"], spec,
                y_work, target_transform=meta.get("target_transform", "none"),
            )
        except Exception as e:  # noqa: BLE001 - a run must still finalize
            score, se, why_not = None, None, f"{type(e).__name__}: {e}"
        if score is None:
            entry["holdout_unavailable"] = why_not
        else:
            entry["holdout_score"] = round(score, 6)
            entry["holdout_se"] = None if se is None else round(se, 6)
            if entry["val_score"] is not None:
                # How much the validation half flattered this model. Signed the
                # same way as every other `vs_*` number in the harness — positive
                # is "the val score was better than the truth", which is the
                # direction every measurement we have has pointed so far.
                entry["optimism"] = _gain(entry["val_score"], score, spec.metric)
        scored.append(entry)

    # The blend last, so it reads as what it is: a combination of the rows above
    # it, not a candidate that competed with them. It has an `optimism` like the
    # rest, and the expectation on it is that it is small — an average selects
    # nothing, so there is no luck in its validation score to give back.
    if blend is not None:
        entry = {
            "experiment_id": BLEND_ID,
            "model_type": blend["model_type"],
            "val_score": blend.get("val_score"),
            "members": blend["members"],
            "selected": False,
            "shipped": True,
        }
        head = blend["meta"]
        try:
            source_path = session.resolve_table(head["source_table"])
            y_work = load_table(source_path)[spec.target]
            score, se, why_not, _ = _score_on_held_out(
                session, holdout_path, source_path, blend["models"],
                head["feature_names"], spec, y_work,
                target_transform=head.get("target_transform", "none"),
            )
        except Exception as e:  # noqa: BLE001 - a run must still finalize
            score, se, why_not = None, None, f"{type(e).__name__}: {e}"
        if score is None:
            entry["holdout_unavailable"] = why_not
        else:
            entry["holdout_score"] = round(score, 6)
            entry["holdout_se"] = None if se is None else round(se, 6)
            if entry["val_score"] is not None:
                entry["optimism"] = _gain(entry["val_score"], score, spec.metric)
        scored.append(entry)
    else:
        for entry in scored:
            entry["shipped"] = entry["selected"]
    return {"n_rows": len(load_table(holdout_path)), "scores": scored}


@contextlib.contextmanager
def _minimal_finalize_lock(session: RunSession):
    """Serialize minimal finalization across processes without stale lock state."""
    path = os.path.join(session.run_dir, ".minimal-finalize.lock")
    os.makedirs(session.run_dir, exist_ok=True)
    with open(path, "w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("this run is already finalizing") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _finalize_minimal_run(session: RunSession) -> dict:
    roots = [
        os.path.join(session.run_dir, "experiments"),
        os.path.join(session.run_dir, "bundle"),
    ]
    before = {
        root: set(os.listdir(root)) if os.path.isdir(root) else set()
        for root in roots
    }
    report = os.path.join(session.run_dir, "run_report.md")
    report_existed = os.path.exists(report)
    with _minimal_finalize_lock(session):
        try:
            return _finalize_minimal_run_unlocked(session)
        except Exception:
            for root in roots:
                if not os.path.isdir(root):
                    continue
                for name in set(os.listdir(root)) - before[root]:
                    path = os.path.join(root, name)
                    shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            if not report_existed and os.path.exists(report):
                os.remove(report)
            raise


def _finalize_minimal_run_unlocked(session: RunSession) -> dict:
    """One selected table, one fixed model, one evaluation score."""
    if journal.stop_state(session.run_dir).get("finalized"):
        raise ValueError("this run has already been finalized")

    j = journal.load(session.run_dir)
    if not j.get("task"):
        inspect_data(session)
        j = journal.load(session.run_dir)
    if not (j.get("work_probe") or {}).get("probe_score"):
        journal.set_work_probe(session.run_dir, probe_table(session, session.data_path))
        j = journal.load(session.run_dir)

    spec = session.resolved()
    table_rel = _best_table_path(session, j)
    path = session.resolve_table(table_rel)
    df = load_table(path)
    X, _ = _feature_frame(df, spec.target)
    if X.empty or not len(X.columns):
        raise ValueError("nothing to finalize: no numeric feature columns")
    refusal = _refuse_if_unservable(session, path, df, list(X.columns))
    if refusal is not None:
        raise ValueError(refusal["reason"])
    y = df[spec.target]

    fit = fit_probe(X, y, spec.task_type, spec.metric, spec.target_transform)
    evaluation_score, _, unavailable, _ = _score_on_val(
        session, path, [fit.model], fit.feature_names, spec, y,
        target_transform=fit.target_transform,
    )
    if evaluation_score is None:
        raise ValueError(f"nothing to finalize: {unavailable}")
    evaluation_score = round(float(evaluation_score), 6)

    baseline_model = _make_model("baseline", spec.task_type, {}, spec.seed, spec.metric)
    baseline_model.fit(X, y)
    baseline_score, _, baseline_unavailable, _ = _score_on_val(
        session, path, [baseline_model], list(X.columns), spec, y,
        target_transform="none",
    )
    if baseline_score is None:
        raise ValueError(f"baseline could not be evaluated: {baseline_unavailable}")
    baseline_score = round(float(baseline_score), 6)

    baseline_row = {
        "id": "baseline",
        "model_type": "baseline",
        "metric": spec.metric,
        "source_table": table_rel,
        "val_score": baseline_score,
        "evaluation_score": baseline_score,
        "artifact_path": "",
    }

    exp_id = journal.next_experiment_id(session.run_dir)
    exp_dir = os.path.dirname(session.ctx.path("experiments", exp_id, "_"))
    joblib.dump(fit.model, os.path.join(exp_dir, "final_model.joblib"))
    meta = {
        "id": exp_id,
        "model_type": PROBE_MODEL_TYPE,
        "params": dict(PROBE_PARAMS),
        "target_transform": fit.target_transform,
        "task_type": spec.task_type,
        "metric": spec.metric,
        "evaluation_score": evaluation_score,
        "fit_seconds": fit.fit_seconds,
        "feature_names": list(fit.feature_names),
        "target": spec.target,
        "source_table": table_rel,
    }
    with open(os.path.join(exp_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)
    model_row = {
        "id": exp_id,
        "model_type": PROBE_MODEL_TYPE,
        "params": dict(PROBE_PARAMS),
        "target_transform": fit.target_transform,
        "features": list(fit.feature_names),
        "source_table": table_rel,
        "metric": spec.metric,
        # Internal compatibility for old journal readers; the minimal public
        # vocabulary is `evaluation_score`.
        "val_score": evaluation_score,
        "evaluation_score": evaluation_score,
        "fit_seconds": fit.fit_seconds,
        "artifact_path": session.rel(exp_dir),
    }

    pipeline = _load_lineage(path, spec.task_type)
    bundle_dir = save_bundle(
        session.ctx, pipeline, [fit.model], np.array([1.0]), fit.feature_names,
        spec.task_type, fit.target_transform,
        training_table=table_rel,
    )
    j = journal.load(session.run_dir)
    credit = journal.attribution(j, table_rel)
    split = j.get("split") or {}
    split_mode = split.get("mode", "unknown")
    if split.get("stratified"):
        split_mode += ", stratified"
    report_lines = [
        "# Run report",
        "",
        f"- **Task:** {spec.task_type}",
        f"- **Metric:** {spec.metric}",
        f"- **Evaluation score:** {evaluation_score:.6g}",
    ]
    if spec.metric == "accuracy" and not split.get("split_on") and split.get("n_val"):
        low, high = accuracy_interval(evaluation_score, split["n_val"])
        report_lines.append(
            f"- **Evaluation row-level 95% interval:** {low:.3f}-{high:.3f}"
        )
    report_lines += [
        f"- **Baseline score:** {baseline_score:.6g}",
        f"- **Model:** {PROBE_MODEL_TYPE}",
        f"- **Selected table:** `{table_rel}`",
        f"- **Features:** {', '.join(fit.feature_names)}",
        f"- **Split:** {split.get('n_work')} work / {split.get('n_val')} evaluation "
        f"({split_mode})",
    ]
    composition = target_composition(split)
    if composition:
        report_lines.append(f"- **Target composition:** {composition}")
    report_lines += [
        "",
        "The evaluation rows were used repeatedly to rank candidate tables. This is",
        "an internal search score, not an independent estimate for a future population.",
        "The bundle replays the selected table's fitted lineage from raw rows.",
    ]
    text = "\n".join(report_lines)
    report_path = write_report(session.ctx, text)
    journal.append_experiments(session.run_dir, [baseline_row, model_row])
    journal.mark_finalized(session.run_dir)
    return {
        "best_experiment_id": exp_id,
        "model_type": PROBE_MODEL_TYPE,
        "metric": spec.metric,
        "evaluation_score": evaluation_score,
        "baseline_score": baseline_score,
        "n_experiments": len(j["leaderboard"]),
        "bundle_dir": session.rel(bundle_dir),
        "report_path": session.rel(report_path),
        **({"required_streams": pipeline.required_streams()}
           if pipeline.required_streams() else {}),
        "unmeasured_tables": [item["table"] for item in credit if not item["measured"]],
    }


def finalize_run(session: RunSession) -> dict:
    """Package the best experiment: deployable bundle + human-readable report.

    Called **once**, at the end (§19.1). Reads the journal's ``best`` pointer
    rather than re-deriving anything, and auto-trains the trivial baseline if the
    run never did, so the report can always answer "did this beat doing nothing".
    """
    if session.profile == "minimal":
        return _finalize_minimal_run(session)

    j = journal.load(session.run_dir)
    if journal.stop_state(session.run_dir).get("finalized"):
        raise ValueError(
            "this run has already been finalized: the bundle and report are on disk "
            "and the winner will not be re-picked"
        )

    if j.get("baseline") is None:
        train_tabular_model(session, model_type="baseline", enforce_stop=False)
        j = journal.load(session.run_dir)

    terminal = _terminal_fits(session)
    j = journal.load(session.run_dir)
    if not j["leaderboard"]:
        raise ValueError("nothing to finalize: nothing could be fitted")

    best = j["best"]
    spec = session.resolved()

    # What ships is the inverse-fold-spread-weighted average of every terminal
    # candidate, not the one with the best validation score (0044, 0080). An argmax
    # over k candidates costs
    # about `sqrt(2*ln k)` standard errors of luck by construction and an average
    # costs none, and on Rossmann v7 the argmax's pick was the *worst* of the
    # three boosted trees on rows it had never seen. One candidate is not a blend,
    # so the winner still ships when only one family survived.
    members = _blend_members(session, terminal)
    blend = None
    if len(members) > 1:
        head = members[0]["meta"]
        source_path = session.resolve_table(head["source_table"])
        models = [m["model"] for m in members]
        # Scored on the validation half like every other candidate, so the run has
        # a number for it in the same units — and so its `optimism` on the sealed
        # rows is comparable to theirs. This is a measurement, not a selection:
        # nothing downstream compares it to anything before deciding.
        val_score, val_se, _, _ = _score_on_val(
            session, source_path, models, head["feature_names"], spec,
            load_table(source_path)[spec.target],
            target_transform=head.get("target_transform", "none"),
        )
        blend = {
            "members": [m["experiment_id"] for m in members],
            "model_types": [m["meta"]["model_type"] for m in members],
            "model_type": "blend(" + "+".join(m["meta"]["model_type"] for m in members) + ")",
            "models": models,
            "meta": head,
            "val_score": None if val_score is None else round(val_score, 6),
            "val_se": None if val_se is None else round(val_se, 6),
        }

    # After `best` is read, and it could not be otherwise: the sealed split is
    # allowed to judge the choice, never to make it (0023). The winner first, then
    # the runners-up it was picked over, then the blend that shipped instead.
    candidates = [best["experiment_id"]] + [
        out["experiment_id"] for out in terminal
        if out.get("experiment_id") and out["experiment_id"] != best["experiment_id"]
    ]
    holdout = _score_on_holdout(
        session, candidates, selected_id=best["experiment_id"], blend=blend,
    )
    if holdout is not None:
        journal.set_holdout(session.run_dir, holdout)
    exp_dir = os.path.join(session.run_dir, best["artifact_path"])
    with open(os.path.join(exp_dir, "meta.json")) as f:
        meta = json.load(f)
    model = joblib.load(os.path.join(exp_dir, "final_model.joblib"))

    ship = blend["meta"] if blend else meta
    ship_models = blend["models"] if blend else [model]
    # 0080 — the blend ships weighted by inverse out-of-sample instability, not
    # equally, so a member that will be optimistic out of sample cannot inflate the
    # average by contributing 1/k of every prediction regardless of how wrong it is
    # (the failure 0045 registered and the next sealed run confirmed). The weights
    # come from each member's cross-fitting fold-spread — fixed before the sealed
    # set is read, from neither `val` nor the seal. Every member still ships; this
    # reweights, it does not select (0044).
    if blend is not None:
        spreads = [float(np.std(m["meta"].get("fold_metrics") or [np.nan])) for m in members]
        ship_weights = blend_weights(spreads)
    else:
        ship_weights = np.array([1.0])
    # The bundle must be able to rebuild the shipped models' features from *raw*
    # input, so it inherits the lineage of whichever table they were trained on —
    # empty when trained on the raw table, one step per encoding otherwise. Every
    # blend member shares that table by construction (`_blend_members`), which is
    # what makes one pipeline enough for all of them.
    pipeline = _load_lineage(session.resolve_table(ship["source_table"]), ship["task_type"])
    bundle_dir = save_bundle(
        session.ctx,
        pipeline,
        ship_models,
        ship_weights,
        ship["feature_names"],
        ship["task_type"],
        ship.get("target_transform", "none"),
        training_table=ship["source_table"],
    )

    credit = journal.attribution(j, ship["source_table"])
    text = build_report(
        attribution=credit,
        task_type=ship["task_type"],
        metric=ship["metric"],
        # The validation half is what the leaderboard ranked by, so it is what
        # picked the winner and what the report leads with. `cv_score` follows as
        # the fold estimate — the gap between them is the run's own record of how
        # far the search drifted from the folds it was optimising. A blend has
        # neither: it was never cross-validated, because it was never fitted.
        selection_score=blend["val_score"] if blend else meta["val_score"],
        val_se=blend["val_se"] if blend else meta.get("val_se"),
        cv_score=None if blend else meta["cv_score"],
        fold_metrics=[] if blend else meta["fold_metrics"],
        baseline_score=j["baseline"]["val_score"],
        model_types=blend["model_types"] if blend else [meta["model_type"]],
        feature_names=ship["feature_names"],
        split=j.get("split"),
        holdout=holdout,
    )
    report_path = write_report(session.ctx, text)
    journal.mark_finalized(session.run_dir)

    shipped = shipped_holdout(holdout)
    return {
        "best_experiment_id": best["experiment_id"],
        "model_type": blend["model_type"] if blend else meta["model_type"],
        # Which fits the bundle averages, and their ids, so a reader of the return
        # value alone can tell a blend from a single model without opening it.
        **({} if blend is None else {"blend_members": blend["members"]}),
        "metric": ship["metric"],
        "val_score": blend["val_score"] if blend else meta["val_score"],
        # Present only on a run that was cut with a third split, and the run is
        # over by the time the agent reads it — there is no call left to make
        # with it, which is the point. `optimism` is the number 0023 exists for:
        # how much the validation half flattered the model that shipped.
        **(
            {}
            if shipped is None
            else {
                "holdout_score": shipped["holdout_score"],
                "optimism": shipped.get("optimism"),
            }
        ),
        # The argmax winner's, always — a blend has no folds of its own, and this
        # is still the run's record of how far the search drifted from them.
        "cv_score": meta["cv_score"],
        "baseline_score": j["baseline"]["val_score"],
        "n_experiments": len(j["leaderboard"]),
        "bundle_dir": session.rel(bundle_dir),
        # Second inputs this bundle cannot predict without. Named in the return
        # value and not only in the manifest, because whoever ships from this run
        # reads the finalize result — and a bundle that needs a file nobody knew
        # to pass looks broken rather than under-supplied.
        **({"required_streams": pipeline.required_streams()}
           if pipeline.required_streams() else {}),
        "report_path": session.rel(report_path),
        # Which links in the winning table's history were actually measured. An
        # unmeasured one means the score cannot be attributed to the columns
        # either side of it — worth saying, since the agent is about to write a
        # summary claiming some feature was the one that worked.
        "unmeasured_tables": [s["table"] for s in credit if not s["measured"]],
    }
