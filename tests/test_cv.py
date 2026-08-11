import numpy as np

from tabular_agent.cv import make_folds


def test_folds_cover_all_rows_exactly_once_in_validation():
    y = np.arange(50)
    folds = make_folds(y, "regression", n_splits=5, seed=42)
    assert len(folds) == 5
    seen = np.concatenate([va for _, va in folds])
    assert sorted(seen.tolist()) == list(range(50))


def test_folds_are_deterministic():
    y = np.arange(50)
    a = make_folds(y, "regression", n_splits=5, seed=42)
    b = make_folds(y, "regression", n_splits=5, seed=42)
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        assert np.array_equal(tr_a, tr_b)
        assert np.array_equal(va_a, va_b)


def test_stratified_preserves_class_ratio():
    y = np.array([0] * 40 + [1] * 10)
    folds = make_folds(y, "binary", n_splits=5, seed=42)
    for _, va in folds:
        ratio = y[va].mean()
        assert 0.1 <= ratio <= 0.3  # ~0.2 in every fold


# ------------------------------------------------------- forward chaining


def test_no_fold_ever_trains_on_the_future():
    """The property the whole strategy exists for. Random folds let a model
    predict Wednesday having trained on Tuesday and Thursday of the same store —
    it interpolates instead of forecasting, and scores far too well for it."""
    order = np.repeat(np.arange(10), 3)  # 10 days, 3 stores each
    y = np.arange(30, dtype=float)

    folds = make_folds(y, "regression", n_splits=3, order=order)
    assert len(folds) == 3
    for tr, va in folds:
        assert order[tr].max() < order[va].min()


def test_ordered_folds_never_split_a_value_across_the_boundary():
    """Rows sharing a day share the promo calendar, the weather, the holiday. A
    cut inside a day lets the model learn that day's level from one group of
    stores and apply it to the other."""
    order = np.repeat(np.arange(10), 3)
    y = np.arange(30, dtype=float)

    for tr, va in make_folds(y, "regression", n_splits=3, order=order):
        assert not set(order[tr]) & set(order[va])


def test_the_earliest_block_is_never_validated():
    """Nothing precedes it to train on, so it has no out-of-fold prediction —
    which downstream code has to carry as NaN rather than as a zero."""
    order = np.repeat(np.arange(12), 2)
    y = np.arange(24, dtype=float)

    folds = make_folds(y, "regression", n_splits=3, order=order)
    validated = np.concatenate([va for _, va in folds])
    assert len(validated) == len(set(validated.tolist())), "no row validated twice"

    never = sorted(set(range(24)) - set(validated.tolist()))
    assert never, "forward chaining must leave the first block unvalidated"
    assert order[never].max() < order[validated].min()


def test_ordered_folds_are_deterministic_and_ignore_the_seed():
    """There is nothing random to seed: the order decides everything."""
    order = np.repeat(np.arange(10), 3)
    y = np.arange(30, dtype=float)

    a = make_folds(y, "regression", n_splits=3, seed=1, order=order)
    b = make_folds(y, "regression", n_splits=3, seed=999, order=order)
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        assert np.array_equal(tr_a, tr_b) and np.array_equal(va_a, va_b)


def test_unsorted_input_is_handled_rather_than_assumed():
    """The rows arrive in whatever order the table has, not in time order."""
    rng = np.random.default_rng(0)
    order = rng.permutation(np.repeat(np.arange(10), 3))
    y = np.arange(30, dtype=float)

    for tr, va in make_folds(y, "regression", n_splits=3, order=order):
        assert order[tr].max() < order[va].min()


def test_too_few_distinct_values_is_an_error_not_a_silent_bad_split():
    """Splitting a three-year column into six blocks cannot be done honestly, and
    quietly returning something else would produce a confident wrong number."""
    import pytest

    order = np.repeat(np.arange(3), 10)
    y = np.arange(30, dtype=float)
    with pytest.raises(ValueError, match="distinct values"):
        make_folds(y, "regression", n_splits=5, order=order)


def test_order_of_the_wrong_length_is_refused():
    import pytest

    with pytest.raises(ValueError, match="ordering column"):
        make_folds(np.arange(30, dtype=float), "regression", order=np.arange(10))


def test_random_folds_are_unchanged_by_any_of_this():
    """Both strategies are kept: exchangeable rows are still best served by
    random folds, where every row is validated and every fold trains on as much."""
    y = np.arange(50)
    folds = make_folds(y, "regression", n_splits=5, seed=42)
    assert sorted(np.concatenate([va for _, va in folds]).tolist()) == list(range(50))
