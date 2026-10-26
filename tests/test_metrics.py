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


def test_accuracy_maps_probabilities_back_to_real_class_labels():
    binary_y = np.array(["no", "yes", "yes"])
    binary_p = np.array([0.2, 0.9, 0.4])
    assert compute_metric(
        "accuracy", binary_y, binary_p, labels=["no", "yes"]
    ) == pytest.approx(2 / 3)

    multiclass_y = np.array([10, 20, 30])
    multiclass_p = np.array([
        [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]
    ])
    assert compute_metric(
        "accuracy", multiclass_y, multiclass_p, labels=[10, 20, 30]
    ) == pytest.approx(1.0)


def test_greater_is_better_directions():
    assert greater_is_better("auc") is True
    assert greater_is_better("accuracy") is True
    assert greater_is_better("rmse") is False
    assert greater_is_better("logloss") is False


def test_supported_metrics_set():
    assert SUPPORTED_METRICS == {
        "rmse", "rmsle", "rmspe", "mae", "poisson_deviance", "logloss", "auc", "accuracy"
    }


def test_every_metric_says_what_it_measures():
    """A name and a direction say which way to push a number, not what it rewards.

    The registry and the descriptions must not drift apart: a metric added without
    one leaves the agent optimising something nobody explained to it.
    """
    from tabular_agent.metrics import METRIC_NOTES, describe_metric

    assert set(METRIC_NOTES) == SUPPORTED_METRICS
    assert "relative" in describe_metric("rmspe")
    with pytest.raises(ValueError):
        describe_metric("bogus")


def test_rmspe_is_relative_and_ignores_zero_valued_rows():
    """Both halves are the definition, not a convenience.

    Rows whose true value is zero have no relative error — it is undefined, not
    large — and the competitions using this metric exclude them explicitly. That
    exclusion belongs in the metric, where the model can still see the rows and
    decide what to do about them, rather than in a prep script that hides them.
    """
    y = np.array([100.0, 200.0])

    # Same absolute error, different relative error: 10/100 vs 10/200.
    small = compute_metric("rmspe", y, np.array([110.0, 200.0]))
    large = compute_metric("rmspe", y, np.array([100.0, 210.0]))
    assert small > large

    # A zero-valued row is not scored, so any prediction on it is free.
    with_zero = np.array([100.0, 0.0])
    assert compute_metric("rmspe", with_zero, np.array([110.0, 0.0])) == pytest.approx(
        compute_metric("rmspe", with_zero, np.array([110.0, 99999.0]))
    )
    # And it agrees with the same row scored alone.
    assert compute_metric("rmspe", with_zero, np.array([110.0, 5.0])) == pytest.approx(
        compute_metric("rmspe", np.array([100.0]), np.array([110.0]))
    )

    assert greater_is_better("rmspe") is False
    with pytest.raises(ValueError, match="none"):
        compute_metric("rmspe", np.array([0.0, 0.0]), np.array([1.0, 2.0]))


def test_greater_is_better_unknown_metric_raises_value_error():
    with pytest.raises(ValueError):
        greater_is_better("bogus")


def test_poisson_deviance_matches_sklearn_and_scores_zeros():
    from sklearn.metrics import mean_poisson_deviance

    y = np.array([0.0, 0.0, 3.0, 10.0])
    pred = np.array([0.5, 2.0, 3.0, 8.0])
    assert compute_metric("poisson_deviance", y, pred) == pytest.approx(
        mean_poisson_deviance(y, pred)
    )
    assert greater_is_better("poisson_deviance") is False


def test_poisson_deviance_floors_non_positive_predictions_instead_of_crashing():
    """A squared-error member can predict <= 0; the deviance of that is undefined,
    so it is charged a steep finite price rather than ending the comparison."""
    y = np.array([0.0, 2.0])
    score = compute_metric("poisson_deviance", y, np.array([-1.0, 2.0]))
    assert np.isfinite(score)
    assert score == compute_metric("poisson_deviance", y, np.array([0.0, 2.0]))


def test_poisson_deviance_leaves_negative_true_values_unscored():
    """A negative count is outside the metric's domain, excluded like rmspe's zeros."""
    y = np.array([-5.0, 0.0, 3.0])
    pred = np.array([100.0, 0.5, 2.0])
    assert compute_metric("poisson_deviance", y, pred) == pytest.approx(
        compute_metric("poisson_deviance", y[1:], pred[1:])
    )
    with pytest.raises(ValueError, match="none"):
        compute_metric("poisson_deviance", np.array([-1.0]), np.array([1.0]))
