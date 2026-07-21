import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.encoders import TargetEncoder, ENCODER_CLASSES, oof_target_encode


def test_target_encoder_maps_stored_means_not_recomputed():
    df = pd.DataFrame({"c": ["a", "a", "b", "b"]})
    y = np.array([10.0, 20.0, 100.0, 200.0])
    enc = TargetEncoder(["c"], smoothing=0.0).fit(df, y)
    # New data has NO target column; transform must still work from stored means.
    out = enc.transform(pd.DataFrame({"c": ["a", "b"]}))
    assert out["c_te"].tolist() == [15.0, 150.0]


def test_target_encoder_unseen_category_falls_back_to_global_mean():
    df = pd.DataFrame({"c": ["a", "b"]})
    y = np.array([10.0, 30.0])
    enc = TargetEncoder(["c"], smoothing=0.0).fit(df, y)
    out = enc.transform(pd.DataFrame({"c": ["zzz"]}))
    assert out["c_te"].iloc[0] == 20.0  # global mean


def test_target_encoder_serialization_roundtrip():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    y = np.array([1.0, 3.0, 9.0])
    enc = TargetEncoder(["c"], smoothing=1.0).fit(df, y)
    restored = ENCODER_CLASSES["target"].from_dict(enc.to_dict())
    a = enc.transform(df)["c_te"].to_numpy()
    b = restored.transform(df)["c_te"].to_numpy()
    assert np.allclose(a, b)


def test_oof_target_encoding_is_less_leaky_than_naive():
    # Each category appears a few times; target is pure noise w.r.t. category.
    rng = np.random.default_rng(0)
    n = 300
    cats = rng.integers(0, 60, size=n)  # ~5 rows per category
    df = pd.DataFrame({"c": [f"k{c}" for c in cats]})
    y = rng.normal(size=n)  # no real signal

    # Naive: fit on ALL rows, then read back -> each row sees its own target.
    naive = TargetEncoder(["c"], smoothing=0.3).fit(df, y).transform(df)["c_te"].to_numpy()

    # Out-of-fold: each row encoded from folds where it was NOT present.
    folds = make_folds(y, "regression", n_splits=5, seed=42)
    oof_df, _ = oof_target_encode(df, y, ["c"], folds, smoothing=0.3)
    oof = oof_df["c_te"].to_numpy()

    corr_naive = abs(np.corrcoef(naive, y)[0, 1])
    corr_oof = abs(np.corrcoef(oof, y)[0, 1])
    # Signal is noise: the naive encoding fabricates correlation; OOF stays near zero.
    assert corr_naive > 0.25
    assert corr_oof < 0.15
    assert corr_naive > corr_oof + 0.1


def test_oof_returns_final_encoder_fit_on_all():
    df = pd.DataFrame({"c": ["a", "a", "b", "b", "b"]})
    y = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    folds = make_folds(y, "regression", n_splits=5, seed=42)
    _, final = oof_target_encode(df, y, ["c"], folds, smoothing=0.0)
    out = final.transform(pd.DataFrame({"c": ["a", "b"]}))
    assert out["c_te"].tolist() == [2.0, 20.0]
