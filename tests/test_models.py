import numpy as np
import pandas as pd
import pytest

from tabular_agent.cv import make_folds
from tabular_agent.models import TrainingTooSlow, train_cv, predict_model, TrainResult
from tabular_agent.pipeline import Pipeline, save_bundle, load_bundle, predict_bundle
from tabular_agent.runctx import RunContext


def _regression_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 2.0 * x[:, 0] - x[:, 1] + 0.1 * rng.normal(size=n)
    return pd.DataFrame(x, columns=["a", "b", "c"]), pd.Series(y)


def test_baseline_regression_oof_is_constant_mean():
    X, y = _regression_data()
    folds = make_folds(y, "regression", seed=42)
    res = train_cv(X, y, folds, "regression", "rmse", model_type="baseline")
    assert isinstance(res, TrainResult)
    assert res.oof.shape == (len(y),)
    # Baseline predicts a (near-)constant; std across predictions is tiny.
    assert res.oof.std() < 0.2


def test_lgbm_regression_beats_baseline():
    X, y = _regression_data()
    folds = make_folds(y, "regression", seed=42)
    base = train_cv(X, y, folds, "regression", "rmse", model_type="baseline")
    lgbm = train_cv(X, y, folds, "regression", "rmse", model_type="lgbm")
    assert np.mean(lgbm.fold_metrics) < np.mean(base.fold_metrics)
    assert lgbm.feature_names == ["a", "b", "c"]
    assert len(lgbm.models) == 5
    assert lgbm.final_model is not None


def test_binary_oof_is_probability_vector():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] + rng.normal(scale=0.3, size=200) > 0).astype(int))
    folds = make_folds(y, "binary", seed=42)
    res = train_cv(X, y, folds, "binary", "auc", model_type="lgbm")
    assert res.oof.shape == (200,)
    assert res.oof.min() >= 0.0 and res.oof.max() <= 1.0
    assert np.mean(res.fold_metrics) > 0.8  # AUC


def test_predict_model_binary_returns_positive_class_prob():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    y = pd.Series((X["a"] > 0).astype(int))
    folds = make_folds(y, "binary", seed=42)
    res = train_cv(X, y, folds, "binary", "auc", model_type="lgbm")
    preds = predict_model(res.final_model, X, "binary")
    assert preds.shape == (100,)


def _multiclass_imbalanced_data(seed=0):
    # Imbalanced 3-class data (70/46/4) sized so that with 5-fold
    # StratifiedKFold (seed=42) the rarest class (4 members) is not spread
    # across every fold: at least one fold's validation split ends up with
    # only 2 of the 3 classes present, even though the model (trained on
    # all classes globally) always emits 3 proba columns.
    rng = np.random.default_rng(seed)
    n0, n1, n2 = 70, 46, 4
    y = np.array([0] * n0 + [1] * n1 + [2] * n2)
    n = len(y)
    x0 = rng.normal(size=n) + y * 1.5
    x1 = rng.normal(size=n) - y * 0.8
    X = pd.DataFrame({"f0": x0, "f1": x1})
    return X, pd.Series(y)


def test_multiclass_cv_survives_fold_missing_a_class():
    X, y = _multiclass_imbalanced_data(seed=0)
    folds = make_folds(y, "multiclass", seed=42)
    res = train_cv(X, y, folds, "multiclass", "logloss", model_type="lgbm")
    assert res.oof.shape == (len(y), 3)
    assert all(np.isfinite(m) for m in res.fold_metrics)
    assert len(res.models) == 5


def test_multiclass_oof_zero_pads_class_absent_from_a_fold_train_split():
    # Class 2 has a single member, so StratifiedKFold(n_splits=5) is forced
    # to put it entirely in one fold's validation split, leaving that fold's
    # training split with zero instances of class 2. This is exactly the
    # branch in predict_model() that scatters a fold model's (class-
    # incomplete) predict_proba columns into the full global-classes index
    # space via np.searchsorted, leaving the unseen class's column at 0.
    # StratifiedKFold emits a "least populated class" UserWarning here; that
    # is expected and inherent to the scenario, not a bug to fix.
    rng = np.random.default_rng(7)
    n0, n1, n2 = 60, 40, 1
    rare_class = 2
    y = np.array([0] * n0 + [1] * n1 + [rare_class] * n2)
    n = len(y)
    x0 = rng.normal(size=n) + y * 1.5
    x1 = rng.normal(size=n) - y * 0.8
    x2 = rng.normal(size=n) + (y == 1) * 2.0
    X = pd.DataFrame({"f0": x0, "f1": x1, "f2": x2})
    y = pd.Series(y)

    folds = make_folds(y, "multiclass", n_splits=5, seed=42)
    res = train_cv(X, y, folds, task_type="multiclass", metric="logloss", model_type="lgbm")

    assert res.oof.shape == (len(y), 3)
    assert all(np.isfinite(m) for m in res.fold_metrics)

    # Locate the fold whose training split lacks the rare class entirely
    # while its validation split contains it.
    zero_padded_va = None
    for tr, va in folds:
        ytr = y.iloc[tr]
        yva = y.iloc[va]
        if (ytr == rare_class).sum() == 0 and (yva == rare_class).sum() > 0:
            zero_padded_va = va
            break
    assert zero_padded_va is not None, "fixture must produce a fold missing the rare class from training"

    # Direct proof the zero-padding branch ran and is correct: the OOF
    # probability column for the rare class is exactly 0 on that fold's
    # validation rows.
    assert np.allclose(res.oof[zero_padded_va, rare_class], 0.0)

    # For all other (non-zero-padded) folds, OOF rows still sum to ~1.
    other_va = np.setdiff1d(np.arange(len(y)), zero_padded_va)
    assert np.allclose(res.oof[other_va].sum(axis=1), 1.0, atol=1e-6)


def test_multiclass_bundle_roundtrip_no_target_needed(tmp_path):
    X, y = _multiclass_imbalanced_data(seed=0)
    folds = make_folds(y, "multiclass", seed=42)
    res = train_cv(X, y, folds, "multiclass", "logloss", model_type="lgbm")

    pipeline = Pipeline("multiclass")
    ctx = RunContext(str(tmp_path / "run"))
    bundle_dir = save_bundle(
        ctx,
        pipeline,
        models=[res.final_model],
        weights=[1.0],
        feature_names=list(X.columns),
        task_type="multiclass",
    )
    bundle = load_bundle(bundle_dir)

    rng = np.random.default_rng(1)
    new_raw = pd.DataFrame({"f0": rng.normal(size=5), "f1": rng.normal(size=5)})
    preds = predict_bundle(bundle, new_raw)
    assert preds.shape == (5, 3)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-6)


def test_caller_params_override_defaults_instead_of_colliding():
    """These params come from an LLM: re-specifying a default must not be a crash."""
    from tabular_agent.models import _make_model

    m = _make_model("lgbm", "binary", {"n_estimators": 7}, seed=0)
    assert m.n_estimators == 7
    m = _make_model("xgb", "regression", {"n_estimators": 7, "random_state": 1}, seed=0)
    assert m.n_estimators == 7 and m.random_state == 1


def test_catboost_aliases_replace_the_default_spelling():
    """CatBoost rejects two spellings of one setting, even with equal values."""
    from tabular_agent.models import _make_model

    m = _make_model("catboost", "binary", {"iterations": 5, "depth": 3}, seed=0)
    p = m.get_params()
    assert p["iterations"] == 5 and "n_estimators" not in p
    assert p["depth"] == 3 and "max_depth" not in p
    m.fit(pd.DataFrame({"a": [0.0, 1.0] * 10}), pd.Series([0, 1] * 10))  # actually builds


# ------------------------------------------------------------ target transforms


def _skewed_regression(n=200, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, 10, n)
    # Multiplicative noise on an exponential response: the shape where fitting on
    # the log scale is the right move and squared error on the raw scale is not.
    y = np.exp(0.5 * x) * rng.lognormal(0, 0.3, n)
    return pd.DataFrame({"x": x}), pd.Series(y)


def test_scores_stay_on_the_original_scale():
    """The rule the whole feature rests on. If a transformed experiment reported
    its metric on the transformed scale, the leaderboard would rank numbers in
    different units and every comparison in the run would be void."""
    from tabular_agent.cv import make_folds

    X, y = _skewed_regression()
    folds = make_folds(y, "regression", n_splits=3)
    res = train_cv(X, y, folds, "regression", "rmsle", model_type="lgbm",
                   target_transform="log1p")

    # An rmsle computed on log1p(y) would be a small number near 0.3; on the
    # original scale it is comparable to the raw-fit score below.
    raw = train_cv(X, y, folds, "regression", "rmsle", model_type="lgbm")
    assert res.oof.min() > 0                      # dollars, not log-dollars
    assert abs(res.oof.mean() - y.mean()) < 0.5 * y.mean()
    assert abs(raw.oof.mean() - y.mean()) < 0.5 * y.mean()
    # Both metrics are on the same scale, so the comparison is meaningful at all.
    assert 0 < np.mean(res.fold_metrics) < 2
    assert 0 < np.mean(raw.fold_metrics) < 2


def test_fitting_on_the_log_scale_helps_when_the_metric_is_rmsle():
    from tabular_agent.cv import make_folds

    X, y = _skewed_regression()
    folds = make_folds(y, "regression", n_splits=3)
    raw = np.mean(train_cv(X, y, folds, "regression", "rmsle", model_type="lgbm").fold_metrics)
    log = np.mean(train_cv(X, y, folds, "regression", "rmsle", model_type="lgbm",
                           target_transform="log1p").fold_metrics)
    assert log < raw


def test_oof_is_stored_on_the_original_scale():
    """`explore_experiment` computes `residual = y - oof_pred`. Out-of-fold values
    in log-dollars would silently redefine what a residual means."""
    from tabular_agent.cv import make_folds

    X, y = _skewed_regression()
    folds = make_folds(y, "regression", n_splits=3)
    res = train_cv(X, y, folds, "regression", "rmsle", model_type="lgbm",
                   target_transform="log1p")
    assert np.corrcoef(res.oof, y.to_numpy())[0, 1] > 0.8
    assert res.oof.max() > 10  # log1p of this target maxes out around 5


def test_transform_round_trips_through_the_bundle():
    """The failure this guards against is silent: a bundle that forgets the
    inverse predicts log-dollars and every number still looks like a number."""
    import numpy as _np

    from tabular_agent.pipeline import Pipeline, load_bundle, predict_bundle, save_bundle
    from tabular_agent.runctx import RunContext
    import tempfile

    from tabular_agent.cv import make_folds

    X, y = _skewed_regression()
    folds = make_folds(y, "regression", n_splits=3)
    res = train_cv(X, y, folds, "regression", "rmsle", model_type="lgbm",
                   target_transform="log1p")

    with tempfile.TemporaryDirectory() as d:
        ctx = RunContext(d)
        save_bundle(ctx, Pipeline("regression"), [res.final_model], _np.array([1.0]),
                    list(X.columns), "regression", "log1p")
        preds = predict_bundle(load_bundle(ctx.path("bundle", "_").rsplit("/", 1)[0]), X)

    assert preds.mean() > 0.3 * y.mean()  # dollars
    assert _np.corrcoef(preds, y.to_numpy())[0, 1] > 0.9


def test_a_transform_outside_its_domain_is_refused_with_a_fix():
    from tabular_agent.cv import make_folds

    X, y = _skewed_regression()
    y = y - y.mean()  # now has negatives
    folds = make_folds(y, "regression", n_splits=3)
    with pytest.raises(ValueError, match="requires values >= 0"):
        train_cv(X, y, folds, "regression", "rmse", target_transform="log1p")


def test_classification_targets_are_never_transformed():
    from tabular_agent.cv import make_folds

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=80)})
    y = pd.Series((X["x"] > 0).astype(int))
    folds = make_folds(y, "binary", n_splits=2)
    with pytest.raises(ValueError, match="regression only"):
        train_cv(X, y, folds, "binary", "logloss", target_transform="log1p")


def test_unknown_transform_lists_the_real_ones():
    from tabular_agent.cv import make_folds

    X, y = _skewed_regression()
    folds = make_folds(y, "regression", n_splits=2)
    with pytest.raises(ValueError, match="unknown target_transform"):
        train_cv(X, y, folds, "regression", "rmse", target_transform="boxcox")


def test_a_fit_reports_what_it_cost():
    """Wall clock is recorded because nothing else in a run measures it.

    Rossmann v6 produced two experiments with byte-identical parameters that took
    3.8 and 113.2 minutes, and the only way to find that out afterwards was the
    mtimes of the files they wrote.
    """
    X = pd.DataFrame({"a": np.arange(60.0), "b": np.arange(60.0) % 7})
    y = pd.Series(np.arange(60.0))
    folds = make_folds(y, "regression", n_splits=3)

    result = train_cv(X, y, folds, "regression", "rmse", model_type="baseline")

    assert result.fit_seconds >= 0.0
    assert isinstance(result.fit_seconds, float)


def test_a_fit_over_its_time_budget_is_abandoned_with_a_measurement():
    """A wall-clock ceiling, checked between folds.

    Deliberately not a cap on parameters: v6's 113-minute fit had the same
    parameters as one that took under four minutes, so cost is not a function of
    what was asked for. A budget of zero stands in for "however slow the machine
    turned out to be" — the first completed fold always exceeds it.
    """
    X = pd.DataFrame({"a": np.arange(60.0), "b": np.arange(60.0) % 7})
    y = pd.Series(np.arange(60.0))
    folds = make_folds(y, "regression", n_splits=3)

    with pytest.raises(TrainingTooSlow, match="1 of 3 folds done"):
        train_cv(X, y, folds, "regression", "rmse", model_type="baseline",
                 time_budget_s=0.0)


def test_no_budget_means_no_ceiling():
    """The default, and the only safe default for a library that cannot know the
    size of the data or the speed of the machine."""
    X = pd.DataFrame({"a": np.arange(60.0), "b": np.arange(60.0) % 7})
    y = pd.Series(np.arange(60.0))
    folds = make_folds(y, "regression", n_splits=3)

    result = train_cv(X, y, folds, "regression", "rmse", model_type="baseline",
                      time_budget_s=None)

    assert len(result.fold_metrics) == 3
