"""Per-prediction attribution: additive in the space it names, and it names the
shipped blend, not one member."""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.cv import make_folds
from tabular_agent.explain import explain_bundle
from tabular_agent.models import train_cv
from tabular_agent.pipeline import Pipeline, load_bundle, predict_bundle, save_bundle
from tabular_agent.runctx import RunContext


def _features(seed=0, n=200):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)}
    )


def _bundle(tmp_path, X, y, model_types, task_type, metric, weights, transform="none",
            objective=None):
    folds = make_folds(y, task_type, n_splits=4, seed=42)
    results = [
        train_cv(X, y, folds, task_type, metric, model_type=mt, target_transform=transform,
                 objective=objective)
        for mt in model_types
    ]
    ctx = RunContext(str(tmp_path / f"run_{'_'.join(model_types)}_{task_type}"))
    bundle_dir = save_bundle(
        ctx,
        Pipeline(task_type),
        [r.final_model for r in results],
        np.asarray(weights, dtype=float),
        list(X.columns),
        task_type,
        target_transform=transform,
        objective=objective,
    )
    return load_bundle(bundle_dir)


@pytest.mark.parametrize(
    "model_types",
    [["lgbm"], ["catboost"], ["elasticnet"], ["lgbm", "catboost", "elasticnet"]],
)
def test_regression_reconstructs_the_shipped_prediction_exactly(tmp_path, model_types):
    """For target_transform='none' the model output *is* the prediction, so
    base + Σ contributions equals predict_bundle to numerical precision."""
    X = _features()
    y = pd.Series(2 * X["a"] - 3 * X["b"] + 10.0)
    weights = np.full(len(model_types), 1.0 / len(model_types))
    bundle = _bundle(tmp_path, X, y, model_types, "regression", "rmse", weights)

    explanation = explain_bundle(bundle, X)
    prediction = predict_bundle(bundle, X)

    assert explanation.space == "target"
    np.testing.assert_allclose(explanation.predicted, prediction, atol=1e-9)


def test_xgb_reconstructs_within_its_float32_shap_precision(tmp_path):
    """XGBoost's pred_contribs run in float32, so the reconstruction matches the
    prediction to that precision rather than to double — still exact for reading."""
    X = _features()
    y = pd.Series(2 * X["a"] - 3 * X["b"] + 10.0)
    bundle = _bundle(tmp_path, X, y, ["xgb"], "regression", "rmse", [1.0])

    explanation = explain_bundle(bundle, X)
    np.testing.assert_allclose(
        explanation.predicted, predict_bundle(bundle, X), rtol=1e-4, atol=1e-3
    )


def test_contributions_shape_columns_and_index(tmp_path):
    X = _features(n=30)
    X.index = [f"row{i}" for i in range(len(X))]
    y = pd.Series(X["a"].to_numpy() + 5.0, index=X.index)
    bundle = _bundle(tmp_path, X, y, ["lgbm"], "regression", "rmse", [1.0])

    explanation = explain_bundle(bundle, X)

    assert list(explanation.contributions.columns) == ["a", "b", "c"]
    assert list(explanation.contributions.index) == list(X.index)
    assert explanation.contributions.shape == (30, 3)


def test_for_row_orders_by_absolute_push(tmp_path):
    X = _features(n=60)
    y = pd.Series(4 * X["a"] - 0.1 * X["b"] + 2.0)
    bundle = _bundle(tmp_path, X, y, ["elasticnet"], "regression", "rmse", [1.0])

    row = explain_bundle(bundle, X).for_row(0)
    magnitudes = row.abs().to_numpy()
    assert (magnitudes[:-1] >= magnitudes[1:]).all()
    # `a` drives the target an order of magnitude harder than `b`, so it leads.
    assert row.index[0] == "a"


def test_log1p_is_additive_on_the_log_scale_and_says_so(tmp_path):
    """Under a non-linear transform the attribution stays exact on the transformed
    scale, and `space` names it instead of pretending it lands on the target."""
    X = _features(seed=1, n=300)
    y = pd.Series(np.clip(np.expm1(1.5 + 0.7 * X["a"] - 0.4 * X["b"]), 0.0, None))
    bundle = _bundle(
        tmp_path, X, y, ["lgbm", "elasticnet"], "regression", "rmse",
        [0.5, 0.5], transform="log1p",
    )

    explanation = explain_bundle(bundle, X)
    from tabular_agent.models import predict_model

    blended_log = sum(
        0.5 * predict_model(m, X[bundle["feature_names"]], "regression")
        for m in bundle["models"]
    )
    assert explanation.space == "log1p(target)"
    np.testing.assert_allclose(explanation.predicted, blended_log, atol=1e-9)


def test_binary_is_additive_in_log_odds(tmp_path):
    """A classifier is additive in log-odds; each member's sigmoid(base+Σ) is its
    own probability, and the average of those is the shipped probability."""
    from scipy.special import expit

    X = _features(seed=2, n=300)
    y = pd.Series((1.2 * X["a"] - 0.8 * X["b"] > 0).astype(int))
    bundle = _bundle(tmp_path, X, y, ["lgbm", "elasticnet"], "binary", "logloss", [0.5, 0.5])

    explanation = explain_bundle(bundle, X)
    from tabular_agent.explain import _member_contributions

    member_probas = []
    for model in bundle["models"]:
        contribs, base = _member_contributions(model, X[bundle["feature_names"]], "binary")
        member_probas.append(expit(base + contribs.sum(axis=1)))
    shipped = predict_bundle(bundle, X)

    assert explanation.space == "log-odds"
    np.testing.assert_allclose(0.5 * member_probas[0] + 0.5 * member_probas[1], shipped, atol=1e-9)


def test_weighted_blend_is_the_weighted_sum_of_member_attributions(tmp_path):
    """The combined attribution is the shipped 0080-weighted combination, so an
    unequal weight moves the base and the contributions by that weight."""
    X = _features(n=120)
    y = pd.Series(X["a"].to_numpy() * 3 - X["c"].to_numpy() + 7.0)
    from tabular_agent.explain import _member_contributions

    weights = [0.75, 0.25]
    bundle = _bundle(tmp_path, X, y, ["lgbm", "elasticnet"], "regression", "rmse", weights)
    combined = explain_bundle(bundle, X)

    Xf = X[bundle["feature_names"]]
    expected_base = 0.0
    expected = np.zeros((len(X), X.shape[1]))
    for w, model in zip(weights, bundle["models"]):
        contribs, base = _member_contributions(model, Xf, "regression")
        expected += w * contribs
        expected_base += w * base
    np.testing.assert_allclose(combined.contributions.to_numpy(), expected, atol=1e-9)
    assert combined.base_value == pytest.approx(expected_base)


def test_multiclass_is_refused_not_guessed(tmp_path):
    X = _features(n=150)
    y = pd.Series(np.repeat([0, 1, 2], 50))
    bundle = _bundle(tmp_path, X, y, ["lgbm"], "multiclass", "logloss", [1.0])
    with pytest.raises(NotImplementedError, match="multiclass"):
        explain_bundle(bundle, X)


def test_facade_explain_drops_identifiers_like_predict(tmp_path):
    """Through the public Bundle, explain prepares rows exactly as predict does:
    the identifier is transport, never a feature that gets a contribution."""
    import json

    from tabular_agent import Bundle

    X = _features(n=80)
    y = pd.Series(2 * X["a"] + 1.0)
    bundle = _bundle(tmp_path, X, y, ["lgbm", "elasticnet"], "regression", "rmse", [0.5, 0.5])
    bundle_dir = tmp_path / "run_lgbm_elasticnet_regression" / "bundle"
    (bundle_dir / "api_plan.json").write_text(
        json.dumps({"identifier_columns": ["RowId"], "joins": [], "split_on": False})
    )

    served = X.copy()
    served.insert(0, "RowId", range(len(served)))
    facade = Bundle(bundle_dir)
    explanation = facade.explain(served)

    assert "RowId" not in explanation.contributions.columns
    np.testing.assert_allclose(explanation.predicted, facade.predict(served), atol=1e-9)


@pytest.mark.parametrize("model_types", [["lgbm"], ["xgb"], ["catboost"], ["elasticnet"]])
def test_a_poisson_member_is_additive_in_log_space(tmp_path, model_types):
    """Under a Poisson objective every member's raw output is log(expected count),
    so the attribution is additive there and exp() of it is the shipped count."""
    X = _features(n=400)
    rng = np.random.default_rng(1)
    y = pd.Series(rng.poisson(np.exp(0.5 + 0.7 * X["a"] - 0.4 * X["b"])).astype(float))
    bundle = _bundle(tmp_path, X, y, model_types, "regression", "poisson_deviance",
                     [1.0], objective="poisson")

    explanation = explain_bundle(bundle, X)

    assert explanation.space == "log(target)"
    np.testing.assert_allclose(
        np.exp(explanation.predicted), predict_bundle(bundle, X), rtol=1e-4
    )
