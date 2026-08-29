"""Target transforms: what the model is fitted on, chosen per experiment.

A regression model fitted with squared loss optimises squared error *on the
scale it is given*. When the metric is `rmsle` — squared error on the log scale —
those are different objectives, and the model spends its capacity on the wrong
one. Fitting on ``log1p(y)`` aligns them.

Three rules shape this module, and each closes a way the idea could go wrong.

**A registry of names, not agent-written code.** The transform must be
*invertible* and *stateless*, and neither property is checkable on arbitrary
code. Free-form code operating on the target would also be exactly what
:mod:`~tabular_agent.transform` exists to prevent (0009): the target is hidden
from agent code because a column that is a function of the target is a leak, and
because there is no target at predict time.

**Stateless only.** Everything here is a fixed function — no parameter is
estimated from the data. That is what makes a target transform leakage-free
without any per-fold machinery: applying ``log1p`` to a training fold cannot
carry information about the rows held out. Box-Cox and Yeo-Johnson are
deliberately absent; their ``lambda`` is fitted, which makes them a member of the
*learned* transform family and a different decision entirely.

**Scores are always reported on the original scale.** The inverse is applied to
predictions before any metric is computed, so `cv_score` and `val_score` mean the
same thing whether an experiment transformed its target or not. Without that rule
the leaderboard would be ranking numbers measured in different units, which is a
worse failure than not having the feature at all.
"""

from collections.abc import Mapping

import numpy as np

__all__ = [
    "TARGET_TRANSFORMS", "apply_target", "invert_target", "check_target_domain",
    "normalize_reference_target", "reference_target_columns",
    "apply_reference_target", "invert_reference_target",
]


REFERENCE_TARGET_KEYS = {"baseline_column", "scale_column"}


def normalize_reference_target(spec, columns=None) -> dict | None:
    """Validate and canonicalise the fixed, per-experiment recipe (0081)."""
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ValueError(
            "reference_target must be an object with baseline_column and/or "
            "scale_column"
        )
    unknown = sorted(set(spec) - REFERENCE_TARGET_KEYS)
    if unknown:
        raise ValueError(
            f"unknown reference_target fields {unknown}; use only "
            "baseline_column and/or scale_column"
        )
    out = {
        key: spec.get(key)
        for key in ("baseline_column", "scale_column")
        if spec.get(key) is not None
    }
    if not out:
        raise ValueError(
            "reference_target must name baseline_column, scale_column, or both; "
            "use None for the direct target"
        )
    for key, value in out.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"reference_target {key} must be a non-empty column name")
    if columns is not None:
        missing = [col for col in out.values() if col not in columns]
        if missing:
            raise ValueError(
                f"reference_target columns are absent from the feature table: {missing}"
            )
    return out


def reference_target_columns(spec) -> list[str]:
    canonical = normalize_reference_target(spec)
    return [] if canonical is None else list(dict.fromkeys(canonical.values()))


def _reference_values(frame, column: str, *, scale: bool) -> np.ndarray:
    try:
        raw = frame[column]
    except (KeyError, TypeError):
        raise ValueError(f"reference_target column {column!r} is unavailable") from None
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"reference_target column {column!r} must be numeric") from None
    if values.ndim != 1:
        raise ValueError(f"reference_target column {column!r} must be one-dimensional")
    invalid = ~np.isfinite(values)
    if invalid.any():
        raise ValueError(
            f"reference_target column {column!r} has {int(invalid.sum())} missing or "
            "non-finite values; reference targets must invert every prediction"
        )
    if scale and (values == 0).any():
        raise ValueError(
            f"reference_target scale_column {column!r} has "
            f"{int((values == 0).sum())} zero values; division would not be invertible"
        )
    return values


def apply_reference_target(y, frame, spec):
    """Return ``(y - baseline) / scale`` under the fixed 0081 contract."""
    canonical = normalize_reference_target(spec, getattr(frame, "columns", None))
    out = np.asarray(y, dtype=float)
    if canonical is None:
        return out
    if len(out) != len(frame):
        raise ValueError("reference_target rows must align one-for-one with the target")
    baseline = canonical.get("baseline_column")
    scale = canonical.get("scale_column")
    if baseline is not None:
        out = out - _reference_values(frame, baseline, scale=False)
    if scale is not None:
        out = out / _reference_values(frame, scale, scale=True)
    if not np.isfinite(out).all():
        raise ValueError("reference_target produced missing or non-finite fit values")
    return out


def invert_reference_target(preds, frame, spec):
    """Invert 0081 back to original target units for scoring and serving."""
    canonical = normalize_reference_target(spec, getattr(frame, "columns", None))
    out = np.asarray(preds, dtype=float)
    if canonical is None:
        return out
    if len(out) != len(frame):
        raise ValueError("reference_target rows must align one-for-one with predictions")
    scale = canonical.get("scale_column")
    baseline = canonical.get("baseline_column")
    if scale is not None:
        out = out * _reference_values(frame, scale, scale=True)
    if baseline is not None:
        out = out + _reference_values(frame, baseline, scale=False)
    return out


def _identity(y):
    return y


#: ``name -> (forward, inverse, requirement)``. ``requirement`` is ``None`` when
#: the transform is defined everywhere, otherwise a short phrase naming the
#: domain, used verbatim in the error the agent gets back.
TARGET_TRANSFORMS = {
    "none": (_identity, _identity, None),
    "log1p": (np.log1p, np.expm1, "values >= 0"),
    "sqrt": (np.sqrt, np.square, "values >= 0"),
}


def _resolve(name: str):
    if name not in TARGET_TRANSFORMS:
        raise ValueError(
            f"unknown target_transform {name!r}. Available: "
            + ", ".join(repr(k) for k in TARGET_TRANSFORMS)
        )
    return TARGET_TRANSFORMS[name]


def check_target_domain(name: str, y, task_type: str) -> None:
    """Raise a message the agent can act on when the transform does not apply.

    Two separate refusals, because they call for different next moves: a
    classification target is never transformed at all, while a regression target
    with negatives rules out *this* transform but not the others.
    """
    forward, _, requirement = _resolve(name)
    if name == "none":
        return
    if task_type != "regression":
        raise ValueError(
            f"target_transform={name!r} applies to regression only; this is a "
            f"{task_type} task, where the target is a class label and has no scale "
            f"to transform. Use target_transform='none'."
        )
    if requirement is not None:
        arr = np.asarray(y, dtype=float)
        if requirement == "values >= 0" and np.nanmin(arr) < 0:
            raise ValueError(
                f"target_transform={name!r} requires {requirement}; the target's "
                f"minimum is {np.nanmin(arr):.4g}. Use 'none', or a transform "
                f"defined on negatives."
            )


def apply_target(name: str, y):
    """Forward transform, for fitting."""
    return _resolve(name)[0](np.asarray(y, dtype=float))


def invert_target(name: str, preds):
    """Back to the original scale, for scoring, storage and prediction.

    Clipped at the transform's own floor rather than left to produce nonsense: a
    model fitted on ``sqrt(y)`` can predict a negative, and squaring that returns
    a *positive* number, which is silently wrong rather than loudly wrong. The
    clip makes the inverse a genuine left inverse on the domain the forward
    transform can reach.
    """
    _, inverse, requirement = _resolve(name)
    preds = np.asarray(preds, dtype=float)
    if name == "sqrt":
        preds = np.clip(preds, 0.0, None)
    elif name == "log1p":
        # expm1 is fine for any real input, but a very negative prediction maps
        # to -1, below the domain the forward transform accepts.
        preds = np.clip(preds, -1.0 + 1e-9, None)
    out = inverse(preds)
    if requirement == "values >= 0":
        out = np.clip(out, 0.0, None)
    return out
