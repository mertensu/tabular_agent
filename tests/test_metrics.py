import numpy as np
import pytest

from tabular_agent.metrics import compute_metric, greater_is_better, SUPPORTED_METRICS


def test_rmse_known_value():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])  # errors: 0,0,2 -> mse=4/3 -> rmse=1.1547
    assert compute_metric("rmse", y, p) == pytest.approx(1.1547005, rel=1e-5)


def test_rmsle_rejects_negative():
    with pytest.raises(ValueError):
        compute_metric("rmsle", np.array([-1.0, 2.0]), np.array([1.0, 2.0]))


def test_mae_known_value():
    assert compute_metric("mae", np.array([1.0, 3.0]), np.array([2.0, 3.0])) == pytest.approx(0.5)


def test_binary_logloss_and_auc():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert compute_metric("logloss", y, p) < 0.3
    assert compute_metric("auc", y, p) == pytest.approx(1.0)


def test_accuracy_from_binary_probs():
    y = np.array([0, 1, 1])
    p = np.array([0.2, 0.9, 0.4])  # -> preds 0,1,0 -> 2/3 correct
    assert compute_metric("accuracy", y, p) == pytest.approx(2 / 3)


def test_multiclass_accuracy_from_proba_matrix():
    y = np.array([0, 1, 2])
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    assert compute_metric("accuracy", y, p) == pytest.approx(1.0)


def test_greater_is_better_directions():
    assert greater_is_better("auc") is True
    assert greater_is_better("accuracy") is True
    assert greater_is_better("rmse") is False
    assert greater_is_better("logloss") is False


def test_supported_metrics_set():
    assert SUPPORTED_METRICS == {"rmse", "rmsle", "mae", "logloss", "auc", "accuracy"}


def test_greater_is_better_unknown_metric_raises_value_error():
    with pytest.raises(ValueError):
        greater_is_better("bogus")
