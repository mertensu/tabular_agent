import numpy as np
import pandas as pd
import pytest

from tabular_agent.probe import (
    PROBE_MODEL_TYPE,
    PROBE_PARAMS,
    PROBE_SEED,
    default_target_transform,
    describe_probe,
    fit_probe,
    probe_predict,
)


def _regression_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = np.abs(2.0 * x[:, 0] - x[:, 1]) + 1.0
    return pd.DataFrame(x, columns=["a", "b", "c"]), pd.Series(y)


def _binary_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = pd.Series((x[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int))
    return pd.DataFrame(x, columns=["a", "b", "c"]), y


# --------------------------------------------------------------- target scale


@pytest.mark.parametrize(
    "metric,expected",
    [("rmsle", "log1p"), ("rmspe", "log1p"), ("rmse", "none"), ("mae", "none")],
)
def test_regression_target_scale_follows_the_metric(metric, expected):
    """0035: the scale is derived, not searched. rmsle/rmspe measure relative
    error, and squared loss on a log scale is that metric (exactly, then
    approximately); nothing derives a scale for rmse or mae."""
    assert default_target_transform(metric, "regression") == expected


@pytest.mark.parametrize("task_type", ["binary", "multiclass"])
@pytest.mark.parametrize("metric", ["auc", "logloss", "accuracy", "rmsle"])
def test_classification_is_never_transformed(metric, task_type):
    assert default_target_transform(metric, task_type) == "none"


def test_unknown_metric_is_refused():
    with pytest.raises(ValueError, match="unknown metric"):
        default_target_transform("f1", "regression")


# ---------------------------------------------------------------- the ruler


def test_the_ruler_is_a_constant_not_the_run_seed():
    """The probe must not vary between runs, or cross-run comparison measures
    the seed as much as the change being studied."""
    assert describe_probe() == {
        "model_type": PROBE_MODEL_TYPE,
        "params": dict(PROBE_PARAMS),
        "seed": PROBE_SEED,
    }


def test_same_table_twice_gives_the_same_predictions():
    X, y = _regression_data()
    a = probe_predict(fit_probe(X, y, "regression", "rmse"), X, "regression")
    b = probe_predict(fit_probe(X, y, "regression", "rmse"), X, "regression")
    np.testing.assert_allclose(a, b)


def test_fit_records_the_ruler_it_used():
    X, y = _regression_data()
    fit = fit_probe(X, y, "regression", "rmse")
    assert fit.spec["model_type"] == PROBE_MODEL_TYPE
    assert fit.spec["seed"] == PROBE_SEED
    assert fit.feature_names == ["a", "b", "c"]
    assert fit.fit_seconds >= 0.0


def test_metric_chooses_the_scale_and_the_caller_can_override():
    X, y = _regression_data()
    assert fit_probe(X, y, "regression", "rmsle").target_transform == "log1p"
    assert fit_probe(X, y, "regression", "rmse").target_transform == "none"
    forced = fit_probe(X, y, "regression", "rmsle", target_transform="none")
    assert forced.target_transform == "none"


def test_predictions_come_back_on_the_original_scale():
    """The model is fitted on log1p(y); nothing outside this module should have
    to know that, exactly as train_cv guarantees for an experiment."""
    X, y = _regression_data()
    logged = probe_predict(fit_probe(X, y, "regression", "rmsle"), X, "regression")
    assert logged.min() > 0.5  # y lives in [1, ~8); log-scale output would not
    assert abs(np.median(logged) - np.median(y)) < 1.0


def test_binary_probe_returns_positive_class_probabilities():
    X, y = _binary_data()
    fit = fit_probe(X, y, "binary", "auc")
    assert fit.target_transform == "none"
    np.testing.assert_array_equal(fit.classes, np.array([0, 1]))
    p = probe_predict(fit, X, "binary")
    assert p.shape == (len(X),)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_a_target_outside_the_transform_domain_is_refused():
    """log1p of a negative target is not defined; the probe must fail loudly
    rather than fit on NaN."""
    X, y = _regression_data()
    y = y - y.max() - 2.0
    with pytest.raises(ValueError):
        fit_probe(X, y, "regression", "rmsle")
