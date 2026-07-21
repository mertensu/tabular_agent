"""Non-negative blend weights over out-of-fold predictions."""

import numpy as np
from scipy.optimize import minimize

from .metrics import compute_metric, greater_is_better


def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    total = w.sum()
    return w / total if total > 0 else np.ones(len(w)) / len(w)


def optimize_blend(oofs: list[np.ndarray], y, metric: str, task_type: str) -> np.ndarray:
    y = np.asarray(y)
    m = len(oofs)
    sign = -1.0 if greater_is_better(metric) else 1.0

    def objective(raw: np.ndarray) -> float:
        w = _normalize(raw)
        pred = sum(wi * o for wi, o in zip(w, oofs))
        return sign * compute_metric(metric, y, pred)

    result = minimize(
        objective,
        np.ones(m) / m,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-6},
    )
    return _normalize(result.x)


def apply_blend(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if len(weights) != len(preds):
        raise ValueError(
            f"weights length ({len(weights)}) does not match preds length ({len(preds)})"
        )
    return sum(wi * p for wi, p in zip(weights, preds))
