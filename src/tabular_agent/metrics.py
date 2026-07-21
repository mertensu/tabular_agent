"""Metric registry. See prediction-format contract in the module test."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

SUPPORTED_METRICS = {"rmse", "rmsle", "mae", "logloss", "auc", "accuracy"}

_GREATER_IS_BETTER = {
    "rmse": False,
    "rmsle": False,
    "mae": False,
    "logloss": False,
    "auc": True,
    "accuracy": True,
}


def greater_is_better(name: str) -> bool:
    if name not in _GREATER_IS_BETTER:
        raise ValueError(f"unknown metric: {name}")
    return _GREATER_IS_BETTER[name]


def compute_metric(name: str, y_true, y_pred, labels: list | None = None) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if name == "rmsle":
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError("rmsle requires non-negative targets and predictions")
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "logloss":
        return float(log_loss(y_true, y_pred, labels=labels))
    if name == "auc":
        return float(roc_auc_score(y_true, y_pred))
    if name == "accuracy":
        labels = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else (y_pred >= 0.5).astype(int)
        return float(accuracy_score(y_true, labels))
    raise ValueError(f"unknown metric: {name}")
