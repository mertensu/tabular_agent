import numpy as np
import pytest

from tabular_agent.blend import apply_blend, blend_weights


def test_apply_blend_rejects_mismatched_weight_length():
    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    with pytest.raises(ValueError):
        apply_blend(preds, np.array([1.0]))


def test_blend_weights_are_a_normalised_distribution():
    w = blend_weights([0.01, 0.02, 0.03])
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


def test_blend_weights_downweight_the_unstable_member():
    # Larger fold-spread -> smaller weight, monotonically.
    w = blend_weights([0.01, 0.05, 0.20])
    assert w[0] > w[1] > w[2]


def test_blend_weights_equal_when_spreads_equal():
    w = blend_weights([0.03, 0.03, 0.03])
    assert np.allclose(w, 1.0 / 3.0)


def test_blend_weights_a_zero_spread_member_cannot_take_all_the_weight():
    # A constant predictor's folds can be identical (spread 0); eps defaults to the
    # median spread so it does not collect weight 1.0 and starve the others.
    w = blend_weights([0.0, 0.04, 0.04])
    assert w[0] < 1.0
    assert (w[1:] > 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_blend_weights_tolerate_nan_or_missing_spread():
    w = blend_weights([float("nan"), 0.02, 0.04])
    assert w.sum() == pytest.approx(1.0)
    assert np.isfinite(w).all()
