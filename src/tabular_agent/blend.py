"""Apply already-declared weights to prediction vectors."""

import numpy as np


def apply_blend(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if len(weights) != len(preds):
        raise ValueError(
            f"weights length ({len(weights)}) does not match preds length ({len(preds)})"
        )
    return sum(wi * p for wi, p in zip(weights, preds))


def blend_weights(fold_spreads, eps: float | None = None) -> np.ndarray:
    """Weight blend members inversely to their out-of-sample instability (0080).

    ``fold_spreads[i]`` is member ``i``'s cross-fitting fold-spread — the
    fold-to-fold dispersion of its OOF score, computed by the harness and derived
    from neither ``val`` (0022) nor the sealed set. A member that is unstable
    across the ordered folds is the in-sample signature of one that will be
    optimistic out of sample, and it is down-weighted here so it cannot inflate
    the shipped average by contributing ``1/k`` of every prediction regardless of
    how wrong it is (the failure 0045 registered).

    The weight is ``w_i ∝ 1 / (s_i + eps)``, normalised to sum to one. This is a
    *relative* use of the spread, not an absolute bar: 0022 found ``fold_spread``
    unreliable as an absolute threshold (it ran ~4x too small), and only the
    ratios between members are read here, never a level. ``eps`` defaults to the
    median spread so that a member reporting a zero or unknown spread — a constant
    predictor's folds can be identical — cannot take all the weight; the safety
    net at the call site is what catches that member instead. Not a category rule,
    no per-dataset constant (0034): the same formula runs on every run.
    """
    s = np.asarray(list(fold_spreads), dtype="float64")
    s = np.where(np.isfinite(s) & (s >= 0.0), s, np.nan)
    med = np.nanmedian(s) if np.isfinite(np.nanmedian(s)) else 0.0
    if eps is None:
        eps = med if med > 0.0 else 1.0
    s = np.where(np.isnan(s), med if med > 0.0 else eps, s)
    inv = 1.0 / (s + eps)
    total = inv.sum()
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(s), 1.0 / len(s))
    return inv / total
