"""Deterministic fold generation shared by encoding and training."""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def make_folds(y, task_type: str, n_splits: int = 5, seed: int = 42):
    y = np.asarray(y)
    if task_type in ("binary", "multiclass"):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, va) for tr, va in splitter.split(np.zeros(len(y)), y)]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in splitter.split(np.zeros(len(y)))]
