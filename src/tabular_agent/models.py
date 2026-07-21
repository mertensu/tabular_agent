"""Model wrappers over LightGBM/XGBoost/CatBoost plus a trivial baseline,
and cross-validated training that emits OOF predictions and fold metrics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import compute_metric


class _Baseline:
    """Predicts the training mean (regression) or class priors (classification)."""

    def __init__(self, task_type: str):
        self.task_type = task_type

    def fit(self, X, y):
        y = np.asarray(y)
        if self.task_type == "regression":
            self.value_ = float(np.mean(y))
        else:
            self.classes_ = np.unique(y)
            self.proba_ = np.array([float((y == c).mean()) for c in self.classes_])
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)

    def predict_proba(self, X):
        return np.tile(self.proba_, (len(X), 1))


def _make_model(model_type: str, task_type: str, params: dict | None, seed: int):
    p = dict(params or {})
    if model_type == "baseline":
        return _Baseline(task_type)
    if model_type == "lgbm":
        import lightgbm as lgb

        if task_type == "regression":
            return lgb.LGBMRegressor(random_state=seed, n_estimators=200, verbose=-1, **p)
        return lgb.LGBMClassifier(random_state=seed, n_estimators=200, verbose=-1, **p)
    if model_type == "xgb":
        import xgboost as xgb

        if task_type == "regression":
            return xgb.XGBRegressor(random_state=seed, n_estimators=200, verbosity=0, **p)
        return xgb.XGBClassifier(random_state=seed, n_estimators=200, verbosity=0, **p)
    if model_type == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        if task_type == "regression":
            return CatBoostRegressor(random_state=seed, n_estimators=200, verbose=False, **p)
        return CatBoostClassifier(random_state=seed, n_estimators=200, verbose=False, **p)
    raise ValueError(f"unknown model_type: {model_type}")


def predict_model(model, X: pd.DataFrame, task_type: str, classes=None) -> np.ndarray:
    if task_type == "regression":
        return np.asarray(model.predict(X), dtype=float)
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if classes is None:
        if task_type == "binary":
            return proba[:, 1]
        return proba
    # Map the model's own (possibly class-incomplete) columns into the
    # global, sorted `classes` index space so callers always get a
    # consistently shaped matrix regardless of which classes this
    # particular model happened to see during fitting.
    classes = np.asarray(classes)
    model_classes = np.asarray(model.classes_)
    col_idx = np.searchsorted(classes, model_classes)
    full = np.zeros((len(X), len(classes)))
    full[:, col_idx] = proba
    if task_type == "binary":
        return full[:, 1]
    return full


@dataclass
class TrainResult:
    oof: np.ndarray
    fold_metrics: list[float]
    models: list
    final_model: object
    feature_names: list[str]
    model_type: str


def train_cv(X, y, folds, task_type, metric, model_type="lgbm", params=None, seed=42) -> TrainResult:
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    n = len(X)
    classes = None
    if task_type in ("binary", "multiclass"):
        classes = np.unique(y.to_numpy())
    if task_type == "multiclass":
        k = len(classes)
        oof = np.zeros((n, k))
    else:
        oof = np.zeros(n)

    models = []
    fold_metrics = []
    for tr, va in folds:
        model = _make_model(model_type, task_type, params, seed)
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = predict_model(model, X.iloc[va], task_type, classes=classes)
        oof[va] = preds
        # `labels` is only consumed by the logloss branch of compute_metric; it's a harmless no-op for other metrics.
        metric_kwargs = {"labels": list(classes)} if classes is not None else {}
        fold_metrics.append(compute_metric(metric, y.iloc[va].to_numpy(), preds, **metric_kwargs))
        models.append(model)

    final = _make_model(model_type, task_type, params, seed)
    final.fit(X, y)
    return TrainResult(oof, fold_metrics, models, final, list(X.columns), model_type)
