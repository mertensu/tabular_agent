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


def test_objective_defaults_to_squared_error_for_regression_and_none_for_classes():
    assert resolve_taskspec(TaskSpec(target="t"), pd.Series([0.0, 1.5, 2.0])).objective == "squared_error"
    assert resolve_taskspec(TaskSpec(target="t"), pd.Series([0, 1, 0])).objective is None


def test_a_poisson_objective_brings_its_own_yardstick_and_no_transform():
    """0022: rank what you fit. Unset metric -> poisson_deviance; the rmsle-derived
    log1p is dropped because the Poisson loss already works on the log of the mean."""
    y = pd.Series(range(40))
    resolved = resolve_taskspec(TaskSpec(target="t", objective="poisson"), y)
    assert resolved.objective == "poisson"
    assert resolved.metric == "poisson_deviance"
    assert resolved.target_transform == "none"

    ranked_on_rmsle = resolve_taskspec(
        TaskSpec(target="t", objective="poisson", metric="rmsle"), y
    )
    assert ranked_on_rmsle.metric == "rmsle"
    assert ranked_on_rmsle.target_transform == "none"


def test_the_poisson_metric_implies_the_poisson_objective():
    resolved = resolve_taskspec(
        TaskSpec(target="t", metric="poisson_deviance"), pd.Series(range(40))
    )
    assert resolved.objective == "poisson"


def test_invalid_objective_combinations_are_refused():
    import pytest

    y = pd.Series(range(40))
    with pytest.raises(ValueError, match="unknown objective"):
        resolve_taskspec(TaskSpec(target="t", objective="gamma"), y)
    with pytest.raises(ValueError, match="regression only"):
        resolve_taskspec(
            TaskSpec(target="t", task_type="binary", objective="poisson"), pd.Series([0, 1, 0])
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_taskspec(
            TaskSpec(target="t", objective="poisson", target_transform="log1p"), y
        )


def test_a_declared_count_is_a_regression_even_with_few_distinct_values():
    """Sparse daily sales take a handful of integers; the detector alone would call
    them class labels."""
    y = pd.Series([0, 1, 0, 2, 0, 0, 3, 1])
    assert resolve_taskspec(TaskSpec(target="t"), y).task_type == "multiclass"
    assert resolve_taskspec(TaskSpec(target="t", objective="poisson"), y).task_type == "regression"
