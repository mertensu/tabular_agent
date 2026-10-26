"""Metric registry. See prediction-format contract in the module test."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    roc_auc_score,
)

SUPPORTED_METRICS = {
    "rmse", "rmsle", "rmspe", "mae", "poisson_deviance", "logloss", "auc", "accuracy",
}

#: The smallest prediction `poisson_deviance` scores. The deviance takes
#: ``log(y_pred)``, so a zero or negative prediction is not a large error but an
#: undefined one. A Poisson fit never produces one; a squared-error member of a
#: blend can, and flooring it charges that row a steep but finite price instead
#: of crashing the comparison.
POISSON_PRED_FLOOR = 1e-9

_GREATER_IS_BETTER = {
    "rmse": False,
    "rmsle": False,
    "rmspe": False,
    "mae": False,
    "poisson_deviance": False,
    "logloss": False,
    "auc": True,
    "accuracy": True,
}

#: What each metric actually measures, for the agent.
#:
#: Not decoration. A metric name plus a direction says which way to push a number
#: and nothing about *what it rewards*, and the two come apart in ways that change
#: what a good model looks like: `rmsle` and `rmspe` both punish relative error, so
#: being 100 out on a value of 200 is a catastrophe and being 100 out on 20,000 is
#: noise — the opposite of `rmse`, where only the absolute size matters.
#:
#: `rmspe` earns its entry twice over, because *which rows it scores* is part of
#: the definition. Rows whose true value is zero are excluded, since the relative
#: error of a zero is undefined. An agent told this can work out for itself what
#: to do about the 17% of Rossmann rows where a closed store sold nothing; an
#: agent not told will optimise hard against rows nobody is scoring.
METRIC_NOTES = {
    "rmse": "Root mean squared error, in the target's own units. Large absolute "
            "errors dominate, so it cares most about the biggest values.",
    "rmsle": "Root mean squared error of log1p(y). Punishes *relative* error, so "
             "being wrong by 100 on a value of 200 counts far more than being "
             "wrong by 100 on a value of 20,000. Under-prediction is penalised "
             "more than over-prediction.",
    "rmspe": "Root mean square percentage error: the error on each row divided by "
             "that row's true value, squared and averaged. Punishes *relative* "
             "error, so a small target with a small absolute error can dominate "
             "the score. **Rows whose true value is zero are not scored at all** — "
             "the relative error of zero is undefined — so predictions on those "
             "rows cannot help or hurt.",
    "mae": "Mean absolute error, in the target's own units. Every row counts the "
           "same regardless of how wrong it is, so outliers matter less than under "
           "rmse.",
    "poisson_deviance": "Mean Poisson deviance: how well the predictions work as the "
                        "expected value of a count. Zeros are ordinary observations, "
                        "not a special case, and the penalty for a miss grows with "
                        "its size relative to the predicted count, so a slow seller "
                        "predicted at 3 when it sold 0 costs more than a fast seller "
                        "predicted at 103 when it sold 100. Predicting zero for a "
                        "row that sold something is punished severely. **Rows whose "
                        "true value is negative are not scored** — a count cannot be "
                        "negative, so its deviance is undefined — and a Poisson fit "
                        "does not train on them either.",
    "logloss": "Penalises confident wrong probabilities sharply. A confident "
               "mistake costs far more than an uncertain one, so calibration "
               "matters as much as ranking.",
    "auc": "Ranking quality: the chance a random positive scores above a random "
           "negative. Indifferent to calibration — only the order matters.",
    "accuracy": "The fraction of rows classified correctly at a 0.5 threshold. "
                "Blind to class imbalance and to how confident the model was.",
}


def greater_is_better(name: str) -> bool:
    if name not in _GREATER_IS_BETTER:
        raise ValueError(f"unknown metric: {name}")
    return _GREATER_IS_BETTER[name]


def describe_metric(name: str) -> str:
    """One sentence on what ``name`` rewards, for whoever has to optimise it."""
    if name not in METRIC_NOTES:
        raise ValueError(f"unknown metric: {name}")
    return METRIC_NOTES[name]


def compute_metric(name: str, y_true, y_pred, labels: list | None = None) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if name == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if name == "rmsle":
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError("rmsle requires non-negative targets and predictions")
        return float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred))))
    if name == "rmspe":
        # Rows with a true value of zero are excluded rather than guarded against:
        # the relative error of zero is not a large number, it is undefined, and
        # the competitions that use this metric say so explicitly. This is part of
        # the metric's definition, not a cleaning step — which is why it lives
        # here and not in a prep script that would hide the rows from the model.
        scored = y_true != 0
        if not np.any(scored):
            raise ValueError(
                "rmspe scores only rows whose true value is non-zero, and this set "
                "has none; every row's relative error is undefined"
            )
        rel = (y_true[scored] - y_pred[scored]) / y_true[scored]
        return float(np.sqrt(np.mean(np.square(rel))))
    if name == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if name == "poisson_deviance":
        # Negative true values are outside a count's domain (a return booked as
        # a sale, a correction), so they are excluded as `rmspe` excludes zeros:
        # part of the metric's definition, stated to the agent in METRIC_NOTES.
        scored = y_true >= 0
        if not np.any(scored):
            raise ValueError(
                "poisson_deviance scores only rows whose true value is non-negative, "
                "and this set has none"
            )
        return float(mean_poisson_deviance(
            y_true[scored], np.maximum(y_pred[scored], POISSON_PRED_FLOOR)
        ))
    if name == "logloss":
        return float(log_loss(y_true, y_pred, labels=labels))
    if name == "auc":
        return float(roc_auc_score(y_true, y_pred))
    if name == "accuracy":
        class_labels = None if labels is None else np.asarray(labels)
        if y_pred.ndim == 2:
            positions = np.argmax(y_pred, axis=1)
            predicted = positions if class_labels is None else class_labels[positions]
        else:
            positive = y_pred >= 0.5
            predicted = positive.astype(int) if class_labels is None else np.where(
                positive, class_labels[-1], class_labels[0]
            )
        return float(accuracy_score(y_true, predicted))
    raise ValueError(f"unknown metric: {name}")
