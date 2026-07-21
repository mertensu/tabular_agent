import pandas as pd

from tabular_agent.taskspec import (
    TaskSpec,
    detect_task_type,
    default_metric,
    resolve_taskspec,
)


def test_detect_binary():
    assert detect_task_type(pd.Series([0, 1, 1, 0])) == "binary"


def test_detect_multiclass_integer_low_cardinality():
    assert detect_task_type(pd.Series([0, 1, 2, 3, 2, 1])) == "multiclass"


def test_detect_regression_float():
    assert detect_task_type(pd.Series([0.5, 1.7, 3.2, 9.1])) == "regression"


def test_detect_regression_high_cardinality_integer():
    assert detect_task_type(pd.Series(range(100))) == "regression"


def test_default_metric_regression_nonneg_is_rmsle():
    assert default_metric("regression", pd.Series([0.0, 1.0, 2.0])) == "rmsle"


def test_default_metric_regression_negative_falls_back_to_rmse():
    assert default_metric("regression", pd.Series([-1.0, 2.0])) == "rmse"


def test_default_metric_classification_is_logloss():
    assert default_metric("binary", pd.Series([0, 1])) == "logloss"
    assert default_metric("multiclass", pd.Series([0, 1, 2])) == "logloss"


def test_resolve_fills_gaps_but_respects_override():
    y = pd.Series([0.0, 1.0, 2.0])
    resolved = resolve_taskspec(TaskSpec(target="t"), y)
    assert resolved.task_type == "regression"
    assert resolved.metric == "rmsle"

    overridden = resolve_taskspec(TaskSpec(target="t", task_type="regression", metric="mae"), y)
    assert overridden.metric == "mae"
