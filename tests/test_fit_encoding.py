import numpy as np
import pandas as pd
import pytest

from tabular_agent.cv import make_folds
from tabular_agent.encoders import (
    FrequencyEncoder,
    ENCODER_CLASSES,
    fit_encoding,
    oof_target_encode,
)


def test_frequency_encoder_maps_and_normalizes():
    df = pd.DataFrame({"c": ["a", "a", "a", "b"]})
    enc = FrequencyEncoder(["c"], normalize=True).fit(df)
    out = enc.transform(df)["c_freq"].to_numpy()
    assert np.allclose(out, [0.75, 0.75, 0.75, 0.25])


def test_frequency_encoder_counts_when_not_normalized():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    enc = FrequencyEncoder(["c"], normalize=False).fit(df)
    assert list(enc.transform(df)["c_freq"]) == [2.0, 2.0, 1.0]


def test_frequency_unknown_category_maps_to_zero():
    train = pd.DataFrame({"c": ["a", "b"]})
    enc = FrequencyEncoder(["c"]).fit(train)
    unseen = pd.DataFrame({"c": ["a", "z"]})
    assert enc.transform(unseen)["c_freq"].to_list() == [0.5, 0.0]


def test_frequency_encoder_roundtrips_through_registry():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    enc = FrequencyEncoder(["c"]).fit(df)
    restored = ENCODER_CLASSES["frequency"].from_dict(enc.to_dict())
    pd.testing.assert_frame_equal(enc.transform(df), restored.transform(df))


def test_fit_encoding_target_returns_oof_and_final():
    df = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    y = np.array([1.0, 3.0, 8.0, 9.0])
    folds = make_folds(y, "regression", n_splits=4, seed=42)
    train_feats, final = fit_encoding(df, ["c"], "target", y=y, folds=folds)
    # matches the underlying oof helper exactly
    oof_ref, final_ref = oof_target_encode(df, y, ["c"], folds)
    pd.testing.assert_frame_equal(train_feats, oof_ref)
    assert final.kind == "target"
    pd.testing.assert_frame_equal(final.transform(df), final_ref.transform(df))


def test_fit_encoding_target_requires_y_and_folds():
    df = pd.DataFrame({"c": ["a", "b"]})
    with pytest.raises(ValueError, match="requires both y and folds"):
        fit_encoding(df, ["c"], "target")


def test_fit_encoding_frequency_needs_no_target():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    train_feats, final = fit_encoding(df, ["c"], "frequency")
    assert list(train_feats.columns) == ["c_freq"]
    assert final.kind == "frequency"
    assert list(train_feats["c_freq"]) == pytest.approx([2 / 3, 2 / 3, 1 / 3])


def test_fit_encoding_rejects_unknown_method():
    df = pd.DataFrame({"c": ["a"]})
    with pytest.raises(ValueError, match="unknown encoding method"):
        fit_encoding(df, ["c"], "onehot")
