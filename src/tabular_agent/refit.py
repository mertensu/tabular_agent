"""Refit a table's lineage on more rows than it was built on.

A run builds every table on its work rows, because its evaluation rows have to
stay unseen while the search ranks tables. Once the search is over that reason
is gone, and the rows it held back are the most recent the run has: in a
forecast they sit between the end of training and the first served row. A
bundle fitted without them serves stale group histories and stale encodings.

:func:`refit_lineage` replays a lineage's *recipe* — the step types and the
parameters the agent chose — on a new set of rows, fitting each step afresh with
the same entry point the tool that built it used. It never re-chooses anything:
which columns, which windows, which smoothing were decided on the work rows, and
only the fitted state (mappings, stored history, out-of-fold values) is redone.

The result has the contract every table in a run has: the frame holds honest
training values (out-of-fold where the tool's were) and the returned pipeline
holds the encoders fitted on all of the new rows, which is what runs at predict.
"""

import pandas as pd

from .cv import make_folds
from .encoders import (
    StreamAggregateEncoder,
    fit_encoding,
    fit_future_covariate_window,
    fit_group_stats,
    fit_history,
    fit_stream_aggregate,
)
from .pipeline import Pipeline
from .rows import select_rows
from .sandbox import run_transform
from .taskspec import ResolvedSpec


class RefitError(ValueError):
    """A lineage step whose recipe cannot be fitted again from what it stored."""


def refit_lineage(
    pipeline: Pipeline,
    rows: pd.DataFrame,
    spec: ResolvedSpec,
    *,
    order: str | None = None,
    streams: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, Pipeline]:
    """Fit every step of ``pipeline`` again on ``rows``; return ``(table, pipeline)``.

    ``rows`` are raw rows with the target, in the shape the lineage's first step
    expects. ``order`` is the column the run is split on, or ``None`` for a random
    split: it decides whether folds are forward-chaining, exactly as it does for
    the tools. Folds are rebuilt on ``rows`` with the spec's seed and fold count,
    so an out-of-fold value here is held out of the same kind of fold the tool's
    was. ``streams`` supplies each declared stream by name.

    Train-only filters are re-evaluated on ``rows``, so rows the agent kept out of
    the fit stay out of it. Sandbox transforms are replayed, not stability-checked
    again: the check is a property of the code and the code has not changed.

    Raises :class:`RefitError` for a step that stored state no current tool can
    produce (a history encoder with lags from before 0072, say): refitting it
    with today's entry point would build a different column under the same name.
    """
    streams = streams or {}
    table = rows.reset_index(drop=True)
    refit = Pipeline(pipeline.task_type)
    for step in pipeline.steps:
        if step["type"] == "transform":
            features = table.drop(columns=[spec.target])
            out, _ = run_transform(features, step["code"], func_name=step["func_name"])
            table = out.join(table[spec.target])
            refit.steps.append(dict(step))
        elif step["type"] == "filter":
            kept, _ = select_rows(table, step["code"])
            if not len(kept):
                raise RefitError("a train-only filter kept none of the refit rows")
            table = kept.reset_index(drop=True)
            refit.steps.append(dict(step))
        elif step["type"] == "encode":
            features, encoder = _refit_encoder(step["encoder"], table, spec, order, streams)
            table = table.join(features)
            refit.add_encoder(encoder)
        else:
            raise RefitError(f"unknown lineage step type {step['type']!r}")
    return table, refit


def _refit_encoder(e: dict, df: pd.DataFrame, spec: ResolvedSpec, order: str | None,
                   streams: dict) -> tuple[pd.DataFrame, object]:
    """``(training columns, encoder)`` for one stored encoder, fitted on ``df``."""
    kind = e["kind"]

    def folds():
        return make_folds(df[spec.target], spec.task_type, n_splits=spec.n_splits,
                          seed=spec.seed, order=None if order is None else df[order].to_numpy())

    if kind == "target":
        return fit_encoding(df, e["columns"], "target", y=df[spec.target], folds=folds(),
                            smoothing=e["smoothing"], stats=e.get("stats", ["mean"]))
    if kind == "frequency":
        return fit_encoding(df, e["columns"], "frequency", normalize=e["normalize"])
    if kind == "ordinal":
        return fit_encoding(df, e["columns"], "ordinal", categories=e["categories"],
                            unknown=e["unknown"])
    if kind == "group_stats":
        return fit_group_stats(df, e["by"], e["value"], e["stat"], folds=folds())
    if kind == "history":
        if order is None:
            raise RefitError("a history step needs the run's ordering column")
        if e.get("offsets") or e.get("latest", True) or not e.get("windows"):
            raise RefitError(
                "this history step stores lags or a last-known value, which the "
                "history tool no longer fits (0072); it cannot be refitted"
            )
        return fit_history(df, e["by"], e["value"], e["order"], folds=folds(),
                           keep=e["keep"], forecast_horizon=spec.forecast_horizon,
                           windows=e["windows"])
    if kind == "stream_aggregate":
        if e["stream"] not in streams:
            raise RefitError(f"stream {e['stream']!r} was not supplied")
        if order is None:
            raise RefitError("a stream aggregate step needs the run's ordering column")
        encoder = StreamAggregateEncoder(
            e["stream"], e["by"], e["value"], e["event_time"], e["observed_at"],
            e["order"], leads=e["leads"], stats=e["stats"],
            event_grain=e.get("event_grain"),
        )
        return fit_stream_aggregate(df, streams[e["stream"]], encoder, folds=folds(),
                                    forecast_horizon=spec.forecast_horizon)
    if kind == "future_window":
        return fit_future_covariate_window(df, e["by"], e["value"], e["order"],
                                           e["block_length"], e.get("aggregates", ()))
    raise RefitError(f"unknown encoder kind {kind!r}")
