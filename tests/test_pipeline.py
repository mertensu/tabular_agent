import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.encoders import OrdinalEncoder, oof_target_encode
from tabular_agent.models import train_cv
from tabular_agent.pipeline import (
    Pipeline,
    save_bundle,
    load_bundle,
    predict_bundle,
)
from tabular_agent.runctx import RunContext


def test_pipeline_transform_roundtrip_dict():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    y = np.array([1.0, 3.0, 9.0])
    folds = make_folds(y, "regression", n_splits=3, seed=42)
    _, final_enc = oof_target_encode(df, y, ["c"], folds)
    p = Pipeline("regression")
    p.add_encoder(final_enc)
    restored = Pipeline.from_dict(p.to_dict())
    pd.testing.assert_frame_equal(p.transform(df), restored.transform(df))


def test_ordinal_encoding_survives_the_bundle_and_ignores_row_order(tmp_path):
    """The train/serve-skew guarantee: codes come from the stored order, not the data."""
    levels = ["small", "medium", "large"]
    train = pd.DataFrame({"size": ["small", "medium", "large", "small"]})
    y = pd.Series([1.0, 5.0, 9.0, 1.0])
    enc = OrdinalEncoder({"size": levels}).fit(train)

    folds = make_folds(y, "regression", n_splits=4, seed=42)
    X = enc.transform(train)[["size_ord"]]
    res = train_cv(X, y, folds, "regression", "rmse", model_type="baseline")

    pipeline = Pipeline("regression")
    pipeline.add_encoder(enc)
    ctx = RunContext(str(tmp_path / "run"))
    bundle_dir = save_bundle(
        ctx, pipeline, [res.final_model], np.array([1.0]), ["size_ord"], "regression"
    )
    bundle = load_bundle(bundle_dir)  # goes through JSON, like a real predict

    # Reversed rows, and a level absent from this frame entirely: codes are unchanged.
    new_raw = pd.DataFrame({"size": ["large", "small"]})
    coded = bundle["pipeline"].transform(new_raw)["size_ord"]
    assert list(coded) == [2.0, 0.0]
    assert predict_bundle(bundle, new_raw).shape == (2,)


def test_predict_bundle_no_target_needed_and_consistent(tmp_path):
    # Build a tiny regression problem with one categorical driver.
    rng = np.random.default_rng(0)
    cats = rng.integers(0, 4, size=120)
    base = np.array([0.0, 10.0, 20.0, 30.0])[cats]
    df = pd.DataFrame({"c": [f"k{c}" for c in cats]})
    y = pd.Series(base + rng.normal(scale=0.5, size=120))

    folds = make_folds(y, "regression", seed=42)
    oof_enc, final_enc = oof_target_encode(df, y, ["c"], folds)

    # Train on the encoded feature.
    X = oof_enc  # single column c_te (OOF-encoded for honest training)
    res = train_cv(X, y, folds, "regression", "rmse", model_type="lgbm")

    pipeline = Pipeline("regression")
    pipeline.add_encoder(final_enc)
    ctx = RunContext(str(tmp_path / "run"))
    bundle_dir = save_bundle(
        ctx,
        pipeline,
        models=[res.final_model],
        weights=np.array([1.0]),
        feature_names=["c_te"],
        task_type="regression",
    )

    bundle = load_bundle(bundle_dir)
    # New raw data: only the raw category column, NO target.
    new_raw = pd.DataFrame({"c": ["k0", "k1", "k2", "k3"]})
    preds = predict_bundle(bundle, new_raw)
    assert preds.shape == (4,)
    # Predictions must track the category means monotonically.
    assert preds[0] < preds[1] < preds[2] < preds[3]


def test_predict_is_deterministic(tmp_path):
    df = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    y = pd.Series([1.0, 2.0, 8.0, 9.0])
    folds = make_folds(y, "regression", n_splits=4, seed=42)
    oof_enc, final_enc = oof_target_encode(df, y, ["c"], folds)
    res = train_cv(oof_enc, y, folds, "regression", "rmse", model_type="baseline")
    pipeline = Pipeline("regression")
    pipeline.add_encoder(final_enc)
    ctx = RunContext(str(tmp_path / "run"))
    bundle_dir = save_bundle(ctx, pipeline, [res.final_model], np.array([1.0]), ["c_te"], "regression")
    bundle = load_bundle(bundle_dir)
    raw = pd.DataFrame({"c": ["a", "b"]})
    p1 = predict_bundle(bundle, raw)
    p2 = predict_bundle(bundle, raw)
    assert np.array_equal(p1, p2)
