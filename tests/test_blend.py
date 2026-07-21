import numpy as np
import pytest

from tabular_agent.blend import optimize_blend, apply_blend
from tabular_agent.metrics import compute_metric


def test_weights_are_nonnegative_and_sum_to_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    oof_a = np.array([1.1, 2.1, 2.9, 3.8])
    oof_b = np.array([3.0, 3.0, 3.0, 3.0])
    w = optimize_blend([oof_a, oof_b], y, "rmse", "regression")
    assert np.all(w >= -1e-9)
    assert abs(w.sum() - 1.0) < 1e-6


def test_blend_prefers_the_better_model():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    good = y + np.array([0.05, -0.05, 0.05, -0.05, 0.0])  # near perfect
    bad = np.full(5, 3.0)  # constant
    w = optimize_blend([good, bad], y, "rmse", "regression")
    assert w[0] > w[1]


def test_blend_beats_or_matches_best_single_model():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = y + np.array([0.3, -0.3, 0.3, -0.3, 0.3, -0.3])
    b = y + np.array([-0.3, 0.3, -0.3, 0.3, -0.3, 0.3])
    w = optimize_blend([a, b], y, "rmse", "regression")
    blended = apply_blend([a, b], w)
    best_single = min(compute_metric("rmse", y, a), compute_metric("rmse", y, b))
    assert compute_metric("rmse", y, blended) <= best_single + 1e-6


def test_apply_blend_rejects_mismatched_weight_length():
    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    with pytest.raises(ValueError):
        apply_blend(preds, np.array([1.0]))
