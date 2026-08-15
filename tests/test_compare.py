"""The difference instrument (0036): does it measure a difference, and refuse
to measure one it cannot see."""

import numpy as np
import pytest

from tabular_agent.compare import (
    BLOCK_MIN_ROWS,
    blockwise_difference,
    compare_predictions,
    paired_difference,
)
from tabular_agent.harness.tools import _gain as _harness_gain


def _two_predictors(n=400, seed=0, gap=0.5):
    """Truth, a good prediction, and a worse one — same rows, in that order."""
    rng = np.random.default_rng(seed)
    y = rng.normal(10.0, 2.0, size=n)
    good = y + rng.normal(0, 0.5, size=n)
    worse = good + rng.normal(0, gap, size=n)
    return y, good, worse


# ------------------------------------------------------------ paired bootstrap


def test_the_better_predictor_gets_a_positive_gain():
    y, good, worse = _two_predictors()
    out = paired_difference("rmse", y, good, worse)
    assert out["gain"] > 0
    assert out["mode"] == "paired_bootstrap"


def test_pairing_beats_treating_the_two_scores_as_independent():
    """The whole reason this module exists: both candidates are scored on the
    *same* resample, so the shared 'this validation half is easy' component
    cancels. 0036 measured the ratio at 0.18-0.37 on a real run; here it only has
    to be materially below 1."""
    y, good, worse = _two_predictors()
    out = paired_difference("rmse", y, good, worse)
    assert out["paired_se"] < out["independent_se"]
    assert out["pairing_ratio"] < 0.9


def test_two_identical_predictors_have_a_zero_gain_and_no_spread():
    """A comparison of a thing with itself must not manufacture uncertainty —
    every resample scores both sides identically, so the difference is exactly
    zero on every draw."""
    y, good, _ = _two_predictors()
    out = paired_difference("rmse", y, good, good)
    assert out["gain"] == 0.0
    assert out["paired_se"] == 0.0


def test_the_gain_convention_matches_the_harness():
    """`compare._gain` is deliberately duplicated rather than imported (it would
    be a cycle). If the two ever disagree, every interval in the run is attached
    to a difference with the opposite sign."""
    from tabular_agent.compare import _gain as _compare_gain

    for metric in ("rmse", "rmsle", "auc", "accuracy"):
        for a, b in ((0.3, 0.5), (0.5, 0.3), (0.4, 0.4)):
            assert _compare_gain(a, b, metric) == pytest.approx(
                _harness_gain(a, b, metric)
            )


def test_a_greater_is_better_metric_keeps_the_same_direction():
    rng = np.random.default_rng(3)
    y = (rng.normal(size=600) > 0).astype(int)
    good = np.clip(y * 0.7 + rng.normal(0, 0.15, size=600) + 0.15, 0.01, 0.99)
    worse = np.clip(0.5 + rng.normal(0, 0.2, size=600), 0.01, 0.99)
    out = paired_difference("auc", y, good, worse)
    assert out["gain"] > 0


def test_too_few_rows_returns_no_estimate_rather_than_a_bad_one():
    y, good, worse = _two_predictors(n=8)
    assert paired_difference("rmse", y, good, worse) is None


def test_misaligned_predictions_are_refused():
    """Silently comparing predictions for different rows would produce an
    interval for a comparison nobody made."""
    y, good, worse = _two_predictors(n=100)
    with pytest.raises(ValueError, match="same rows"):
        paired_difference("rmse", y, good, worse[:50])


# --------------------------------------------------------------- ordered blocks


def test_blocks_report_one_gain_each_oldest_first():
    y, good, worse = _two_predictors(n=1000)
    order = np.arange(1000)
    out = blockwise_difference("rmse", y, good, worse, order, n_blocks=5)
    assert out["mode"] == "ordered_blocks"
    assert len(out["block_gains"]) == 5
    assert out["sign_stable"] is True
    assert out["blocks_favouring_a"] == 5


def test_a_difference_that_decays_is_visible_as_unstable_blocks():
    """The shape 0058 found: a feature real early in the validation window and
    gone later. The pooled gain is positive and says nothing; the blocks say the
    winner changed."""
    n = 1000
    rng = np.random.default_rng(1)
    y = rng.normal(10.0, 2.0, size=n)
    order = np.arange(n)
    # A is much better over the first half and slightly worse over the second.
    a = y + rng.normal(0, 0.2, size=n)
    a[n // 2 :] = y[n // 2 :] + rng.normal(0, 1.4, size=n - n // 2)
    b = y + rng.normal(0, 1.0, size=n)

    out = blockwise_difference("rmse", y, a, b, order, n_blocks=4)
    assert out["gain"] > 0, "pooled, A still looks like the winner"
    assert out["sign_stable"] is False, "the blocks disagree, which is the finding"
    assert 0 < out["blocks_favouring_a"] < out["n_blocks"]


def test_blocks_are_cut_on_the_ordering_column_not_on_row_order():
    """The validation half is written in the input file's row order, not in its
    ordering column's order (RunSession sorts *positions*). Blocking on file
    order would cut arbitrary slices and report a reassuring number, so the
    shuffled and sorted inputs must agree exactly."""
    n = 800
    rng = np.random.default_rng(5)
    y = rng.normal(10.0, 2.0, size=n)
    a = y + rng.normal(0, 0.3, size=n)
    b = y + rng.normal(0, 0.9, size=n)
    order = np.arange(n)

    sorted_out = blockwise_difference("rmse", y, a, b, order, n_blocks=4)

    shuffle = rng.permutation(n)
    shuffled_out = blockwise_difference(
        "rmse", y[shuffle], a[shuffle], b[shuffle], order[shuffle], n_blocks=4
    )
    assert shuffled_out["block_gains"] == sorted_out["block_gains"]


def test_a_validation_half_too_small_to_block_returns_nothing():
    n = BLOCK_MIN_ROWS * 2
    y, good, worse = _two_predictors(n=n)
    assert blockwise_difference("rmse", y, good, worse, np.arange(n), n_blocks=5) is None


def test_one_block_is_refused():
    y, good, worse = _two_predictors(n=500)
    with pytest.raises(ValueError, match="at least 2 blocks"):
        blockwise_difference("rmse", y, good, worse, np.arange(500), n_blocks=1)


def test_an_ordering_column_of_the_wrong_length_is_refused():
    y, good, worse = _two_predictors(n=500)
    with pytest.raises(ValueError, match="one ordering value per row"):
        blockwise_difference("rmse", y, good, worse, np.arange(10))


# ------------------------------------------------------------------ the router


def test_the_ordering_column_selects_the_mode():
    """Passing the column is both the evidence that the run is ordered and the
    data the blockwise mode needs, so the two cannot come apart."""
    y, good, worse = _two_predictors(n=1000)
    assert compare_predictions("rmse", y, good, worse)["mode"] == "paired_bootstrap"
    assert (
        compare_predictions("rmse", y, good, worse, order_values=np.arange(1000))["mode"]
        == "ordered_blocks"
    )


def test_both_modes_agree_on_the_pooled_gain():
    """The two instruments differ in what uncertainty they can see, never in the
    difference they are putting an interval around."""
    y, good, worse = _two_predictors(n=1000)
    rows = compare_predictions("rmse", y, good, worse)
    blocks = compare_predictions("rmse", y, good, worse, order_values=np.arange(1000))
    assert rows["gain"] == pytest.approx(blocks["gain"])
