import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.models import train_cv, predict_model, TrainResult


def _regression_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.1 * rng.normal(size=n)
    return pd.DataFrame(x, columns=["a", "b", "c"]), pd.Series(y)


def test_baseline_regression_oof_is_constant_mean():
    X, y = _regression_data()
    folds = make_folds(y, "regression", seed=42)
    res = train_cv(X, y, folds, "regression", "rmse", model_type="baseline")
    assert isinstance(res, TrainResult)
    assert res.oof.shape == (len(y),)
    # Baseline predicts a (near-)constant; std across predictions is tiny.
    assert res.oof.std() < 0.2


def test_lgbm_regression_beats_baseline():
    X, y = _regression_data()
    folds = make_folds(y, "regression", seed=42)
    base = train_cv(X, y, folds, "regression", "rmse", model_type="baseline")
    lgbm = train_cv(X, y, folds, "regression", "rmse", model_type="lgbm")
    assert np.mean(lgbm.fold_metrics) < np.mean(base.fold_metrics)
    assert lgbm.feature_names == ["a", "b", "c"]
    assert len(lgbm.models) == 5
    assert lgbm.final_model is not None


def test_binary_oof_is_probability_vector():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] + rng.normal(scale=0.3, size=200) > 0).astype(int))
    folds = make_folds(y, "binary", seed=42)
    res = train_cv(X, y, folds, "binary", "auc", model_type="lgbm")
    assert res.oof.shape == (200,)
    assert res.oof.min() >= 0.0 and res.oof.max() <= 1.0
    assert np.mean(res.fold_metrics) > 0.8  # AUC


def test_predict_model_binary_returns_positive_class_prob():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] > 0).astype(int))
    folds = make_folds(y, "binary", seed=42)
    res = train_cv(X, y, folds, "binary", "auc", model_type="lgbm")
    preds = predict_model(res.final_model, X, "binary")
    assert preds.shape == (100,)
