"""A column declared categorical: fitted levels, one rule per model family (0074)."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.encoders import CategoricalEncoder, fit_encoding
from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    _feature_frame, finalize_run, fit_encoding as fit_encoding_tool, inspect_data,
)
from tabular_agent.models import _make_model, predict_model
from tabular_agent.taskspec import TaskSpec


def _frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    shop = rng.choice([f"s{i}" for i in range(12)], size=n)
    effect = {f"s{i}": float(i % 4) * 2.0 for i in range(12)}
    x = rng.normal(size=n)
    y = np.array([effect[s] for s in shop]) + x + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"shop": shop, "x": x, "y": y})


def test_the_levels_are_fitted_and_a_batch_cannot_change_a_code():
    df = _frame()
    enc = CategoricalEncoder(["shop"]).fit(df)
    full = enc.transform(df)["shop_cat"]
    one = enc.transform(df.iloc[[5]])["shop_cat"]

    assert list(full.cat.categories) == sorted(df["shop"].unique())
    assert list(one.cat.categories) == list(full.cat.categories)
    assert one.cat.codes.iloc[0] == full.cat.codes.iloc[5]

    unseen = enc.transform(pd.DataFrame({"shop": ["new", None]}))["shop_cat"]
    assert unseen.isna().all()

    replayed = CategoricalEncoder.from_dict(json.loads(json.dumps(enc.to_dict())))
    assert replayed.transform(df)["shop_cat"].equals(full)


def test_a_numeric_code_can_be_declared_categorical():
    df = pd.DataFrame({"code": [7, 7, 3, np.nan]})
    out, enc = fit_encoding(df, ["code"], "categorical")

    assert list(out["code_cat"].cat.categories) == ["3.0", "7.0"]
    assert out["code_cat"].isna().tolist() == [False, False, False, True]


@pytest.mark.parametrize("model_type", ["lgbm", "xgb", "catboost", "elasticnet"])
def test_every_family_takes_the_category_and_answers_a_single_row_alike(model_type):
    """One rule for every family: a category column is fitted on, as a category."""
    from tabular_agent.explain import _member_contributions

    df = _frame()
    feats, _ = fit_encoding(df, ["shop"], "categorical")
    X = pd.concat([feats, df[["x"]]], axis=1)
    model = _make_model(model_type, "regression", None, 0)
    model.fit(X, df["y"])

    preds = predict_model(model, X, "regression")
    assert np.corrcoef(preds, df["y"])[0, 1] > 0.9
    single = predict_model(model, X.iloc[[3]], "regression")
    assert single[0] == pytest.approx(preds[3], rel=1e-6)

    contribs, base = _member_contributions(model, X, "regression")
    assert contribs.shape == X.shape
    assert np.allclose(contribs.sum(axis=1) + base, preds, atol=1e-3)


def test_the_feature_frame_keeps_categories_and_drops_strings():
    df = _frame()
    feats, _ = fit_encoding(df, ["shop"], "categorical")
    X, dropped = _feature_frame(df.join(feats), "y")

    assert list(X.columns) == ["x", "shop_cat"]
    assert dropped == ["shop"]


@pytest.fixture
def session(tmp_path):
    path = str(tmp_path / "data.csv")
    _frame(n=1500).to_csv(path, index=False)
    return RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3))


def test_a_categorical_table_ships_and_serves_any_batch(session):
    """The test 0074 named: a batch of a few levels, and a level never seen."""
    from tabular_agent.pipeline import load_bundle, predict_bundle

    inspect_data(session)
    out = fit_encoding_tool(session, ["shop"], method="categorical")
    assert out["new_columns"] == ["shop_cat"]
    assert out["probe_score"] is not None
    assert "shop" in out["non_numeric_columns"]

    done = finalize_run(session)
    bundle = load_bundle(os.path.join(session.run_dir, done["bundle_dir"]))
    assert "shop_cat" in bundle["feature_names"]

    rows = _frame(n=200, seed=1).drop(columns=["y"])
    full = predict_bundle(bundle, rows)
    few = predict_bundle(bundle, rows.iloc[:3])
    assert np.allclose(few, full[:3])

    odd = rows.iloc[:2].copy()
    odd["shop"] = ["never-seen", None]
    assert np.isfinite(predict_bundle(bundle, odd)).all()
    assert journal.load(session.run_dir)["tables"][0]["method"] == "categorical"
