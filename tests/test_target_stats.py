"""Per-group statistics of the target beyond the smoothed mean (Gap A).

The mean was the only thing a category could be summarised by. These cover the
rest of the group's target distribution — spread, middle, size — and the two
properties that make it safe: the extra columns are as out-of-fold as the mean,
and an encoder written before they existed keeps predicting what it predicted.
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.cv import make_folds
from tabular_agent.encoders import (
    ENCODER_CLASSES,
    TARGET_STATS,
    TargetEncoder,
    fit_encoding,
    oof_target_encode,
)

ALL_STATS = ["mean", "std", "median", "count"]


def _frame():
    # "a" is symmetric, "b" is skewed by one large value, "c" has a single row.
    df = pd.DataFrame({"k": ["a", "a", "a", "b", "b", "b", "c"]})
    y = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 300.0, 7.0])
    return df, y


# --- the columns themselves ------------------------------------------------


def test_default_is_mean_only_and_keeps_the_bare_column_name():
    df, y = _frame()
    enc = TargetEncoder(["k"]).fit(df, y)
    out = enc.transform(df)
    assert enc.output_columns == ["k_te"]
    assert [c for c in out.columns if c.startswith("k_te")] == ["k_te"]


def test_asking_for_more_stats_adds_suffixed_columns():
    df, y = _frame()
    enc = TargetEncoder(["k"], smoothing=0.0, stats=ALL_STATS).fit(df, y)
    out = enc.transform(df)
    assert enc.output_columns == ["k_te", "k_te_std", "k_te_median", "k_te_count"]
    for name in enc.output_columns:
        assert name in out.columns


def test_the_mean_is_unchanged_by_asking_for_the_others():
    """Adding a statistic must not move the one that was already there."""
    df, y = _frame()
    alone = TargetEncoder(["k"], smoothing=3.0).fit(df, y).transform(df)["k_te"]
    together = (
        TargetEncoder(["k"], smoothing=3.0, stats=ALL_STATS)
        .fit(df, y)
        .transform(df)["k_te"]
    )
    assert np.allclose(alone.to_numpy(), together.to_numpy())


def test_unknown_stat_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown target statistic"):
        TargetEncoder(["k"], stats=["mean", "variance"])


def test_stats_menu_excludes_extremes_and_sum():
    # min/max of a group are essentially one row's target; sum is count * mean.
    assert set(TARGET_STATS) == {"mean", "std", "median", "count"}


# --- what each statistic actually computes ---------------------------------


def test_count_is_the_group_size_and_unseen_means_zero():
    df, y = _frame()
    enc = TargetEncoder(["k"], stats=["count"]).fit(df, y)
    out = enc.transform(pd.DataFrame({"k": ["a", "b", "c", "never-seen"]}))
    assert out["k_te_count"].tolist() == [3.0, 3.0, 1.0, 0.0]


def test_median_resists_the_tail_the_mean_chases():
    df, y = _frame()
    enc = TargetEncoder(["k"], smoothing=0.0, stats=["mean", "median"]).fit(df, y)
    out = enc.transform(pd.DataFrame({"k": ["b"]}))
    assert out["k_te"].iloc[0] == pytest.approx(101.0)  # (1 + 2 + 300) / 3
    assert out["k_te_median"].iloc[0] == pytest.approx(2.0)


def test_std_separates_two_groups_with_the_same_mean():
    df = pd.DataFrame({"k": ["tight", "tight", "wide", "wide"]})
    y = np.array([9.0, 11.0, -90.0, 110.0])
    enc = TargetEncoder(["k"], smoothing=0.0, stats=["mean", "std"]).fit(df, y)
    out = enc.transform(pd.DataFrame({"k": ["tight", "wide"]}))
    assert out["k_te"].tolist() == [10.0, 10.0]  # indistinguishable by mean
    assert out["k_te_std"].iloc[0] < out["k_te_std"].iloc[1]


def test_a_one_row_group_has_no_std_and_falls_back_to_the_population():
    """Undefined and unseen mean the same thing: no usable estimate here."""
    df, y = _frame()
    enc = TargetEncoder(["k"], stats=["std"]).fit(df, y)
    out = enc.transform(pd.DataFrame({"k": ["c", "never-seen"]}))
    population = float(pd.Series(y).std())
    assert out["k_te_std"].tolist() == pytest.approx([population, population])
    assert "c" not in enc.mapping_["k"]["std"]


def test_only_the_mean_is_smoothed():
    """Smoothing is a statement about means; the others must ignore it."""
    df, y = _frame()
    loose = TargetEncoder(["k"], smoothing=0.0, stats=ALL_STATS).fit(df, y).transform(df)
    heavy = TargetEncoder(["k"], smoothing=50.0, stats=ALL_STATS).fit(df, y).transform(df)
    assert not np.allclose(loose["k_te"], heavy["k_te"])
    for name in ["k_te_std", "k_te_median", "k_te_count"]:
        assert np.allclose(loose[name], heavy[name])


# --- leakage ---------------------------------------------------------------


def test_every_statistic_is_out_of_fold_not_just_the_mean():
    """A row must not be able to see itself through std/median/count either.

    Tested by moving one row's target and watching its own feature, rather than
    by correlating the feature with ``y`` the way the mean's leak test does.
    Correlation is the wrong probe for a spread: a row inflates its group's
    ``std`` whether its target is unusually high *or* unusually low, so the leak
    shows up against ``|y - mean|`` and can sit at zero against ``y`` itself
    while being every bit as real. Perturbation catches all four statistics with
    one mechanism, and it is the property that actually matters — whether the
    number a row is handed depends on that row's own answer.
    """
    rng = np.random.default_rng(0)
    n = 300
    cats = rng.integers(0, 60, size=n)  # ~5 rows per category
    df = pd.DataFrame({"k": [f"k{c}" for c in cats]})
    y = rng.normal(size=n)
    moved = y.copy()
    moved[0] += 50.0  # one row, far away

    folds = make_folds(y, "regression", n_splits=5, seed=42)
    names = TargetEncoder(["k"], stats=ALL_STATS).output_columns

    def naive(target):
        enc = TargetEncoder(["k"], smoothing=0.3, stats=ALL_STATS).fit(df, target)
        return enc.transform(df)

    def out_of_fold(target):
        frame, _ = oof_target_encode(
            df, target, ["k"], folds, smoothing=0.3, stats=ALL_STATS
        )
        return frame

    before_naive, after_naive = naive(y), naive(moved)
    before_oof, after_oof = out_of_fold(y), out_of_fold(moved)

    for name in ["k_te", "k_te_std", "k_te_median"]:
        # Fitting on all rows, row 0 moves its own feature: that is the leak.
        assert before_naive[name].iloc[0] != after_naive[name].iloc[0], name
        # Out-of-fold, row 0 is encoded by folds it was not in, so its own
        # target cannot reach it.
        assert before_oof[name].iloc[0] == after_oof[name].iloc[0], name

    # `count` does not move under either, and that is correct rather than a gap:
    # it does not read the target's value at all, only that a row was there.
    for frame_before, frame_after in [(before_naive, after_naive), (before_oof, after_oof)]:
        assert frame_before["k_te_count"].iloc[0] == frame_after["k_te_count"].iloc[0]
    assert set(names) == set(before_oof.columns)


def test_out_of_fold_counts_exclude_the_rows_own_fold():
    df = pd.DataFrame({"k": ["a"] * 10})
    y = np.arange(10, dtype=float)
    folds = make_folds(y, "regression", n_splits=5, seed=42)
    oof, final = oof_target_encode(df, y, ["k"], folds, stats=["count"])
    # Each fold fits on 8 of 10 rows, so the held-out rows see a count of 8 —
    # never the 10 the final encoder stores.
    assert set(oof["k_te_count"]) == {8.0}
    assert final.transform(df)["k_te_count"].iloc[0] == 10.0


# --- serialization ---------------------------------------------------------


def test_roundtrip_preserves_every_statistic():
    df, y = _frame()
    enc = TargetEncoder(["k"], smoothing=1.0, stats=ALL_STATS).fit(df, y)
    restored = ENCODER_CLASSES["target"].from_dict(enc.to_dict())
    before = enc.transform(df)[enc.output_columns]
    after = restored.transform(df)[enc.output_columns]
    pd.testing.assert_frame_equal(before, after)


def test_an_encoder_serialized_before_stats_existed_still_predicts_the_same():
    """A bundle on disk has to keep predicting what it predicted when written."""
    df, y = _frame()
    expected = TargetEncoder(["k"], smoothing=2.0).fit(df, y).transform(df)["k_te"]

    legacy = {
        "kind": "target",
        "columns": ["k"],
        "smoothing": 2.0,
        # Flat {category: value}, and no "stats" key at all.
        "mapping": {
            "k": TargetEncoder(["k"], smoothing=2.0).fit(df, y).mapping_["k"]["mean"]
        },
        "global_mean": float(np.mean(y)),
    }
    restored = ENCODER_CLASSES["target"].from_dict(legacy)
    assert restored.stats == ("mean",)
    assert np.allclose(restored.transform(df)["k_te"].to_numpy(), expected.to_numpy())
    # And an unseen category still lands on the global mean, not on 0.0.
    unseen = restored.transform(pd.DataFrame({"k": ["never-seen"]}))
    assert unseen["k_te"].iloc[0] == pytest.approx(float(np.mean(y)))


# --- the fit_encoding entry point ------------------------------------------


def test_fit_encoding_passes_stats_through():
    df, y = _frame()
    folds = make_folds(y, "regression", n_splits=3, seed=0)
    train_features, encoder = fit_encoding(
        df, ["k"], "target", y=y, folds=folds, stats=ALL_STATS
    )
    assert list(train_features.columns) == encoder.output_columns
    assert len(train_features) == len(df)
