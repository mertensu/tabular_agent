"""Task specification: the single source of truth; auto-detect fills gaps."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TaskSpec:
    """What the data's owner knows and the harness cannot work out for itself.

    ``notes`` and ``column_notes`` are the data dictionary — the one channel by
    which facts about the *meaning* of the data reach the agent. Everything else
    here can be detected from the target; these two cannot be detected from
    anything, which is exactly why they are worth carrying.

    The failure they exist to prevent is not obscurity, it is a column that looks
    usable and is not. A profile can say a column holds 1115 distinct integers; it
    cannot say whether those are store identifiers worth encoding or row numbers
    to discard, and it cannot say that a column present in training will simply
    not exist at predict time. Both of those are facts about the world, and both
    have already cost this project a run.
    """

    target: str
    task_type: str | None = None
    metric: str | None = None
    positive_class: object | None = None
    #: Free text about the dataset as a whole: how it was collected, what a row
    #: is, which rows the real scoring ignores. For anything column-shaped use
    #: `column_notes`, where it reaches the reader beside the column.
    notes: str | None = None
    #: ``{column: what it means}``. Partial by design — describe the columns whose
    #: meaning is not obvious and leave the rest alone; an entry that only restates
    #: the column name is tokens spent on every run for nothing.
    column_notes: dict[str, str] | None = None
    n_splits: int = 5
    seed: int = 42
    #: Scale the target is fitted on, overriding what the metric implies (0035).
    #:
    #: `None` means derive it: `rmsle` and `rmspe` measure relative error, and
    #: squared loss on `log1p` is that metric, so the derivation is available.
    #: Nothing derives a scale for `rmse` or `mae` — a target that should be
    #: logged under `mae` is a fact about the data, which is what this field and
    #: this class are for. Names come from the same validated menu 0021 defined;
    #: free-form code over the target is still impossible.
    target_transform: str | None = None
    #: How far ahead this task actually forecasts: the gap between the end of
    #: history and the *earliest* row a prediction is asked for, in the ordering
    #: column's units (days, usually).
    #:
    #: It is a **minimum**, not a constant, and that distinction is the whole
    #: reason it can be one number. Within a single predict batch the rows sit at
    #: different distances from the end of history — the first is one period out,
    #: the last many — so a design that stored one horizon and treated every
    #: served row as sitting at it would be wrong for every row but the first.
    #: Per-row distance is reported by history's ``_ago`` columns; this is the
    #: floor across the block, and its only job is to say which lag offsets are
    #: answerable at all.
    #:
    #: Left `None` — the default — no lag offsets are legal and history serves the
    #: last known value with its age, which needs no declaration to be honest.
    #: Nothing can derive this: it is a fact about the serving contract, which
    #: makes it the data owner's to state (0029).
    forecast_horizon: float | None = None


@dataclass
class ResolvedSpec:
    target: str
    task_type: str
    metric: str
    positive_class: object | None
    notes: str | None
    column_notes: dict[str, str] | None
    n_splits: int
    seed: int
    #: Always concrete after resolution — derived from the metric when the spec
    #: left it open, so no caller has to repeat the derivation.
    target_transform: str = "none"
    #: Carried through unresolved: nothing can fill it in, so `None` stays `None`
    #: and means "no lag offsets are legal".
    forecast_horizon: float | None = None


def detect_task_type(y: pd.Series) -> str:
    n_unique = int(y.nunique(dropna=True))
    if n_unique <= 2:
        return "binary"
    if pd.api.types.is_float_dtype(y):
        return "regression"
    if pd.api.types.is_integer_dtype(y):
        return "multiclass" if n_unique <= 20 else "regression"
    return "multiclass"


def default_metric(task_type: str, y: pd.Series) -> str:
    if task_type == "regression":
        if pd.api.types.is_numeric_dtype(y) and bool((y.dropna() >= 0).all()):
            return "rmsle"
        return "rmse"
    return "logloss"


def resolve_taskspec(spec: TaskSpec, y: pd.Series) -> ResolvedSpec:
    from .probe import default_target_transform
    from .target import TARGET_TRANSFORMS

    task_type = spec.task_type or detect_task_type(y)
    metric = spec.metric or default_metric(task_type, y)
    if spec.target_transform is not None and spec.target_transform not in TARGET_TRANSFORMS:
        raise ValueError(
            f"unknown target_transform: {spec.target_transform!r} "
            f"(use one of {sorted(TARGET_TRANSFORMS)})"
        )
    if spec.forecast_horizon is not None and spec.forecast_horizon <= 0:
        raise ValueError(
            f"forecast_horizon must be positive, got {spec.forecast_horizon}. It is the "
            "gap to the earliest row you predict; zero would mean the row's own period "
            "is already observed, in which case there is nothing to forecast."
        )
    target_transform = spec.target_transform or default_target_transform(metric, task_type)
    return ResolvedSpec(
        target=spec.target,
        task_type=task_type,
        metric=metric,
        positive_class=spec.positive_class,
        notes=spec.notes,
        column_notes=spec.column_notes,
        n_splits=spec.n_splits,
        seed=spec.seed,
        target_transform=target_transform,
        forecast_horizon=spec.forecast_horizon,
    )
