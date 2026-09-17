import numpy as np
import pandas as pd

from tabular_agent.cv import make_folds
from tabular_agent.encoders import (
    ENCODER_CLASSES,
    GroupStatsEncoder,
    TargetEncoder,
    oof_target_encode,
)


def test_target_aggregate_serving_value_is_batch_independent():
    """The serving-time contract (capability-review finding, 2026-09-16): a
    target-derived aggregate's value for a row must come only from the stored
    training lookup and its own key — never from the target (absent at serving)
    or from which other rows happen to share the batch. Verified here so it
    cannot silently regress into batch-dependent (leaky) behaviour."""
    rng = np.random.default_rng(0)
    n = 400
    train = pd.DataFrame({
        "store": rng.integers(0, 10, n).astype(str),
        "sales": rng.normal(100, 20, n),
    })
    serve = pd.DataFrame({"store": ["3", "7", "1", "3", "999"]})  # 999 unseen; no target col

    # TargetEncoder: aggregate of the target itself.
    te = TargetEncoder(["store"], smoothing=10.0).fit(train[["store"]], train["sales"])
    full = te.transform(serve)["store_te"].to_numpy()
    reversed_ = te.transform(serve.iloc[::-1])["store_te"].to_numpy()[::-1]
    per_row = np.array([te.transform(serve.iloc[[i]])["store_te"].iloc[0] for i in range(len(serve))])
    assert np.allclose(full, reversed_)      # order of the batch cannot change a value
    assert np.allclose(full, per_row)        # nor which rows share the batch
    assert np.isclose(full[-1], te.defaults_["mean"])  # unseen key -> stored default

    # GroupStatsEncoder: aggregate of a value column that may correlate with the target.
    gs = GroupStatsEncoder(["store"], "sales", "mean").fit(train)
    g_full = gs.transform(serve)[gs.output_column].to_numpy()  # 'sales' absent at serving
    g_rev = gs.transform(serve.iloc[::-1])[gs.output_column].to_numpy()[::-1]
    g_row = np.array([gs.transform(serve.iloc[[i]])[gs.output_column].iloc[0] for i in range(len(serve))])
    assert np.allclose(g_full, g_rev)
    assert np.allclose(g_full, g_row)
    assert np.isclose(g_full[0], gs.mapping_["3"])  # serving value is the stored training mapping


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
