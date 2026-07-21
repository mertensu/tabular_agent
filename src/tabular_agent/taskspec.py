"""Task specification: the single source of truth; auto-detect fills gaps."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TaskSpec:
    target: str
    task_type: str | None = None
    metric: str | None = None
    positive_class: object | None = None
    notes: str | None = None
    n_splits: int = 5
    seed: int = 42


@dataclass
class ResolvedSpec:
    target: str
    task_type: str
    metric: str
    positive_class: object | None
    notes: str | None
    n_splits: int
    seed: int


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
    task_type = spec.task_type or detect_task_type(y)
    metric = spec.metric or default_metric(task_type, y)
    return ResolvedSpec(
        target=spec.target,
        task_type=task_type,
        metric=metric,
        positive_class=spec.positive_class,
        notes=spec.notes,
        n_splits=spec.n_splits,
        seed=spec.seed,
    )
