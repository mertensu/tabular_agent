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
