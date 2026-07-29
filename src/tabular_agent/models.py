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


#: CatBoost accepts several spellings of the same setting and **rejects** being given
#: two of them, even with identical values. Our defaults use one spelling; a caller may
#: reasonably use another (`iterations` for `n_estimators`, `depth` for `max_depth`), so
#: a supplied alias has to *replace* our default rather than sit next to it.
_CATBOOST_ALIASES: tuple[tuple[str, ...], ...] = (
    ("n_estimators", "iterations", "num_boost_round", "num_trees"),
    ("random_state", "random_seed"),
    ("verbose", "logging_level", "silent", "verbose_eval"),
    ("learning_rate", "eta"),
    ("max_depth", "depth"),
    ("reg_lambda", "l2_leaf_reg"),
)


def _resolve_params(defaults: dict, params: dict | None, aliases=()) -> dict:
    """Defaults with caller params layered on top, one key per setting.

    Splatting `**params` next to keyword defaults raises `TypeError: got multiple
    values` the moment a caller re-specifies a default — and these params come from
    an LLM, which will. Merging into a single dict makes overriding the normal case
    instead of a crash.
    """
    out = dict(defaults)
    p = dict(params or {})
    for group in aliases:
        if any(name in p for name in group):
            for name in group:
                out.pop(name, None)
    out.update(p)
    return out


def _make_model(model_type: str, task_type: str, params: dict | None, seed: int):
    if model_type == "baseline":
        return _Baseline(task_type)
    if model_type == "lgbm":
        import lightgbm as lgb

        kw = _resolve_params({"random_state": seed, "n_estimators": 200, "verbose": -1}, params)
        cls = lgb.LGBMRegressor if task_type == "regression" else lgb.LGBMClassifier
        return cls(**kw)
    if model_type == "xgb":
        import xgboost as xgb

        kw = _resolve_params({"random_state": seed, "n_estimators": 200, "verbosity": 0}, params)
        cls = xgb.XGBRegressor if task_type == "regression" else xgb.XGBClassifier
        return cls(**kw)
    if model_type == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        kw = _resolve_params(
            {"random_state": seed, "n_estimators": 200, "verbose": False},
            params,
            _CATBOOST_ALIASES,
        )
        cls = CatBoostRegressor if task_type == "regression" else CatBoostClassifier
        return cls(**kw)
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
