"""Per-prediction feature attributions for a deployed bundle.

``bundle.explain(rows)`` answers "why *this* number for *this* row", feature by
feature, for the prediction the bundle actually ships. It is the companion to the
run report's *global* story (which features earned their place across the table):
here the unit is one served row.

Three properties make the answer trustworthy rather than decorative, and each is
the reason a piece of this module looks the way it does.

**It explains the shipped blend, not one member.** The bundle's prediction is the
0080-weighted combination of its members (0044/0080), so the attribution is the
same weighted combination of each member's attribution. Explaining the search
winner, or an unweighted average, would be explaining a number the bundle does
not emit.

**Every contribution is model-native and exact — no surrogate, no sampling.** Each
member reports its own contributions: TreeSHAP for the gradient-boosted members
(``pred_contrib`` / ``ShapValues``), and the closed-form linear attribution
``coefᵢ·(zᵢ)`` for the elasticnet, read off the *fitted, standardised* features so
the centring is the model's own. There is no background dataset to choose and no
Monte-Carlo variance to explain away.

**It is additive in the space the models were fitted on, and says which space
that is.** SHAP decomposes the model *output*: ``base + Σ contributions`` equals
the raw model output, exactly. For a plain regression (``target_transform="none"``)
that raw output *is* the prediction, so the reconstruction equals
``bundle.predict`` to numerical precision (double for every member but XGBoost,
whose ``pred_contribs`` are computed in float32). Under a non-linear target transform
(``log1p``), under a Poisson objective (log of the expected count) or for a
classifier (log-odds), the raw output is one link function away from the shipped
number — the attribution stays exactly additive on the
transformed scale, and :attr:`Explanation.space` names it rather than pretending
the sum lands on the euros or the probability. That honesty is the whole point:
0044 measured how much the invert-then-blend order matters, and a decomposition
that quietly blended on the wrong scale to look additive would be the same error
one level down.

v1 covers regression and binary classification — the two shapes the facade builds
today. Multiclass raises rather than guessing a per-class story nobody has scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Explanation:
    """Per-row, per-feature attributions of a bundle's shipped prediction.

    ``base_value + contributions.sum(axis=1)`` reconstructs :attr:`predicted`, the
    model output on the scale named by :attr:`space`. For ``space == "target"``
    that reconstruction is the shipped prediction itself.
    """

    #: One row per input row, one column per model feature (in the bundle's
    #: feature order). Cell ``(i, j)`` is feature ``j``'s signed push on row ``i``'s
    #: model output, in :attr:`space` units.
    contributions: pd.DataFrame
    #: The blended expected model output — the value a row is explained *away
    #: from*. Scalar, on the :attr:`space` scale.
    base_value: float
    #: ``base_value + contributions.sum(axis=1)``: the reconstructed model output.
    predicted: np.ndarray
    #: The scale the attribution is additive on. ``"target"`` when it equals the
    #: shipped prediction (regression, no transform); otherwise the transformed
    #: regression scale (e.g. ``"log1p(target)"``), ``"log(target)"`` under a
    #: Poisson objective, or ``"log-odds"`` for a classifier — one monotone link
    #: away from what ``predict`` returns.
    space: str

    def for_row(self, i: int) -> pd.Series:
        """Row ``i``'s contributions, largest absolute push first — the natural
        reading order for "what moved this prediction"."""
        row = self.contributions.iloc[i]
        return row.reindex(row.abs().sort_values(ascending=False).index)


def _linear_pipeline_contributions(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """``coefⱼ·zⱼ`` for the elasticnet member, in model-output space.

    The member is an sklearn ``Pipeline`` (impute → scale → penalised linear fit),
    so the exact attribution needs the features *as the linear model saw them*:
    push ``X`` through every step but the last to get the standardised matrix
    ``z``, then read ``coef`` and ``intercept`` off the fitted estimator. Because
    the scaler centred ``z`` on the training mean, ``intercept`` already is the
    model's expected output and ``coefⱼ·zⱼ`` is the deviation feature ``j``
    explains — the linear case of the same SHAP decomposition the trees report,
    with no background set to pick.
    """
    from sklearn.pipeline import Pipeline as SkPipeline

    if not isinstance(model, SkPipeline):
        raise TypeError(f"expected an sklearn Pipeline, got {type(model)!r}")
    z = X
    for _, step in model.steps[:-1]:
        z = step.transform(z)
    z = np.asarray(z, dtype=float)
    final = model.steps[-1][1]
    # `_PenaltySearchThenRefit` wraps the actual ElasticNet/LogisticRegression.
    estimator = getattr(final, "estimator_", final)
    coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)
    intercept = float(np.ravel(estimator.intercept_)[0])
    contribs = z * coef  # (n, f)
    return contribs, intercept


def _baseline_contributions(model, X: pd.DataFrame, task_type: str,
                            objective: str | None = None) -> tuple[np.ndarray, float]:
    """A constant predictor pushes nothing: zeros, with the constant as the base.

    In log-odds space for a classifier, and in log space under a Poisson
    objective, so it combines with the other members on the same scale the trees
    and the linear model report.
    """
    n, f = len(X), X.shape[1]
    zeros = np.zeros((n, f), dtype=float)
    if task_type == "regression":
        if objective == "poisson":
            return zeros, float(np.log(max(float(model.value_), 1e-12)))
        return zeros, float(model.value_)
    p = float(np.asarray(model.proba_)[1])
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return zeros, float(np.log(p / (1.0 - p)))


def _tree_contributions(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Native TreeSHAP for an lgbm / xgb / catboost member.

    Each library returns an ``(n, n_features + 1)`` block whose final column is the
    member's expected output and whose sum across the row equals its raw model
    output. The three are the same object under different spellings; the base
    column is peeled off here so the caller sees a uniform ``(contribs, base)``.
    """
    module = type(model).__module__.split(".")[0]
    if module == "lightgbm":
        raw = np.asarray(model.predict(X, pred_contrib=True), dtype=float)
    elif module == "xgboost":
        import xgboost as xgb

        booster = model.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        raw = np.asarray(booster.predict(dmatrix, pred_contribs=True), dtype=float)
    elif module == "catboost":
        from catboost import Pool

        raw = np.asarray(
            model.get_feature_importance(Pool(X), type="ShapValues"), dtype=float
        )
    else:
        raise NotImplementedError(
            f"no per-prediction attribution for model type {type(model)!r}"
        )
    if raw.ndim != 2 or raw.shape[1] != X.shape[1] + 1:
        # A multiclass tree member returns a per-class block instead; v1 does not
        # ship a multiclass story, and the facade does not build one.
        raise NotImplementedError(
            "per-prediction attribution supports regression and binary "
            f"classification only; this member returned a {raw.shape} block for "
            f"{X.shape[1]} features"
        )
    return raw[:, :-1], float(np.mean(raw[:, -1]))


def _member_contributions(model, X: pd.DataFrame, task_type: str,
                          objective: str | None = None) -> tuple[np.ndarray, float]:
    """Dispatch one member to its native attribution, in model-output space.

    Regression members report on the (possibly transformed) target scale, or on
    the log of the expected count under a Poisson objective; binary classifiers
    report in log-odds — the space each family is additive in, which is
    why the blend is combined there and the link is applied, if at all, only once
    the caller reads :attr:`Explanation.space`.
    """
    module = type(model).__module__.split(".")[0]
    if module in ("lightgbm", "xgboost", "catboost"):
        return _tree_contributions(model, X)
    if type(model).__name__ == "_Baseline":
        return _baseline_contributions(model, X, task_type, objective)
    return _linear_pipeline_contributions(model, X)


def _space(task_type: str, target_transform: str, objective: str | None = None) -> str:
    if task_type != "regression":
        return "log-odds"
    if objective == "poisson":
        return "log(target)"
    if target_transform == "none":
        return "target"
    return f"{target_transform}(target)"


def explain_bundle(bundle: dict, raw_df: pd.DataFrame, streams: dict | None = None) -> Explanation:
    """Attribute the bundle's shipped prediction to features, one row at a time.

    Replays the exact lineage :func:`~tabular_agent.pipeline.predict_bundle` does —
    same steps, same ``streams`` contract, same feature slice — then asks each
    member for its native contributions and combines them with the shipped 0080
    weights. What comes back reconstructs the model output on
    :attr:`Explanation.space`, which is the prediction itself for a plain
    regression and one link away otherwise.
    """
    task_type = bundle["task_type"]
    if task_type == "multiclass":
        raise NotImplementedError(
            "explain() supports regression and binary classification; multiclass "
            "attribution is not built"
        )
    features = bundle["pipeline"].transform(raw_df, streams=streams or {})
    X = features[bundle["feature_names"]]
    weights = np.asarray(bundle["weights"], dtype=float)
    if len(weights) != len(bundle["models"]):
        raise ValueError(
            f"weights length ({len(weights)}) does not match models "
            f"({len(bundle['models'])})"
        )

    combined = np.zeros((len(X), X.shape[1]), dtype=float)
    base = 0.0
    for weight, model in zip(weights, bundle["models"]):
        contribs, member_base = _member_contributions(
            model, X, task_type, bundle.get("objective")
        )
        combined += weight * contribs
        base += weight * member_base

    contributions = pd.DataFrame(
        combined, columns=list(bundle["feature_names"]), index=X.index
    )
    predicted = base + combined.sum(axis=1)
    return Explanation(
        contributions=contributions,
        base_value=float(base),
        predicted=predicted,
        space=_space(task_type, bundle.get("target_transform", "none"), bundle.get("objective")),
    )
