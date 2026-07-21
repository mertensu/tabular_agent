import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.models import train_cv, predict_model, TrainResult
from tabular_agent.pipeline import Pipeline, save_bundle, load_bundle, predict_bundle
from tabular_agent.runctx import RunContext


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


def _multiclass_imbalanced_data(seed=0):
    # Imbalanced 3-class data (70/46/4) sized so that with 5-fold
    # StratifiedKFold (seed=42) the rarest class (4 members) is not spread
    # across every fold: at least one fold's validation split ends up with
    # only 2 of the 3 classes present, even though the model (trained on
    # all classes globally) always emits 3 proba columns.
    rng = np.random.default_rng(seed)
    n0, n1, n2 = 70, 46, 4
    y = np.array([0] * n0 + [1] * n1 + [2] * n2)
    n = len(y)
    x0 = rng.normal(size=n) + y * 1.5
    x1 = rng.normal(size=n) - y * 0.8
    X = pd.DataFrame({"f0": x0, "f1": x1})
    return X, pd.Series(y)


def test_multiclass_cv_survives_fold_missing_a_class():
    X, y = _multiclass_imbalanced_data(seed=0)
    folds = make_folds(y, "multiclass", seed=42)
    res = train_cv(X, y, folds, "multiclass", "logloss", model_type="lgbm")
    assert res.oof.shape == (len(y), 3)
    assert all(np.isfinite(m) for m in res.fold_metrics)
    assert len(res.models) == 5


def test_multiclass_bundle_roundtrip_no_target_needed(tmp_path):
    X, y = _multiclass_imbalanced_data(seed=0)
    folds = make_folds(y, "multiclass", seed=42)
    res = train_cv(X, y, folds, "multiclass", "logloss", model_type="lgbm")

    pipeline = Pipeline("multiclass")
    ctx = RunContext(str(tmp_path / "run"))
    bundle_dir = save_bundle(
        ctx,
        pipeline,
        models=[res.final_model],
        weights=[1.0],
        feature_names=list(X.columns),
        task_type="multiclass",
    )
    bundle = load_bundle(bundle_dir)

    rng = np.random.default_rng(1)
    new_raw = pd.DataFrame({"f0": rng.normal(size=5), "f1": rng.normal(size=5)})
    preds = predict_bundle(bundle, new_raw)
    assert preds.shape == (5, 3)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-6)
