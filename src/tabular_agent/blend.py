"""Apply already-declared weights to prediction vectors."""

import numpy as np


def apply_blend(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if len(weights) != len(preds):
        raise ValueError(
            f"weights length ({len(weights)}) does not match preds length ({len(preds)})"
        )
    return sum(wi * p for wi, p in zip(weights, preds))
