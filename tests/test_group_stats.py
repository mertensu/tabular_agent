"""GroupStatsEncoder: a per-group statistic that is looked up, not recomputed.

The property under test throughout is the one that separates this from the
`groupby(...).transform("mean")` the stability check rejects: the number a row
gets must not depend on which other rows arrived with it.
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.cv import make_folds
from tabular_agent.encoders import (
    ENCODER_CLASSES,
    GroupStatsEncoder,
    fit_group_stats,
)


@pytest.fixture
def panel():
    """Three groups, four rows each, with a group-level signal in `value`."""
    return pd.DataFrame(
        {
            "store": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0, 100.0, 100.0, 100.0, 100.0],
            "y": np.arange(12.0),
        }
    )


def test_the_fitted_value_does_not_depend_on_the_batch(panel):
    """The whole justification for this being a tool rather than free code."""
    enc = GroupStatsEncoder(["store"], "value", "mean").fit(panel)
    col = enc.output_column

    whole = enc.transform(panel)[col]
    # One row on its own — the predict-time worst case, and exactly what
    # `groupby("store")["value"].transform("mean")` gets wrong (it would return
    # the row's own value).
    alone = enc.transform(panel.iloc[[1]])[col]

    assert whole.iloc[1] == pytest.approx(2.5)
    assert alone.iloc[0] == pytest.approx(2.5)


def test_the_value_column_need_not_exist_at_predict_time(panel):
    """Why this makes an unavailable column usable instead of merely refused."""
    enc = GroupStatsEncoder(["store"], "value", "mean").fit(panel)
    serving = panel.drop(columns=["value", "y"])

    out = enc.transform(serving)

    assert out[enc.output_column].tolist()[:4] == [2.5] * 4


def test_a_group_never_seen_falls_back_to_the_population_value(panel):
    enc = GroupStatsEncoder(["store"], "value", "mean").fit(panel)
    unseen = pd.DataFrame({"store": ["zzz"]})

    value = enc.transform(unseen)[enc.output_column].iloc[0]

    assert value == pytest.approx(panel["value"].mean())


def test_an_extensive_stat_falls_back_to_zero_instead(panel):
    """`count` of a group that was never seen is zero, not the population count."""
    enc = GroupStatsEncoder(["store"], "value", "count").fit(panel)

    unseen = enc.transform(pd.DataFrame({"store": ["zzz"]}))[enc.output_column].iloc[0]
    seen = enc.transform(pd.DataFrame({"store": ["a"]}))[enc.output_column].iloc[0]

    assert unseen == 0.0
    assert seen == 4.0


def test_a_group_with_no_usable_estimate_also_takes_the_default():
    """`std` of a one-row group is NaN; that is a fallback, not a NaN feature."""
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 3.0, 5.0]})
    enc = GroupStatsEncoder(["g"], "v", "std").fit(df)

    out = enc.transform(df)[enc.output_column]

    assert not out.isna().any()
    assert out.iloc[2] == pytest.approx(df["v"].std())


def test_a_multi_column_key_cannot_collide():
    """`("a_b", "c")` and `("a", "b_c")` are different groups, not one."""
    df = pd.DataFrame(
        {"p": ["a_b", "a"], "q": ["c", "b_c"], "v": [1.0, 100.0]}
    )
    enc = GroupStatsEncoder(["p", "q"], "v", "mean").fit(df)

    out = enc.transform(df)[enc.output_column]

    assert out.tolist() == [1.0, 100.0]


def test_the_column_name_states_everything_that_defines_it(panel):
    assert GroupStatsEncoder(["store"], "value", "mean").output_column == "store_value_mean"
    assert (
        GroupStatsEncoder(["store", "dow"], "value", "max").output_column
        == "store_dow_value_max"
    )


def test_it_round_trips_through_json(panel):
    import json

    enc = GroupStatsEncoder(["store"], "value", "median").fit(panel)
    revived = ENCODER_CLASSES["group_stats"].from_dict(json.loads(json.dumps(enc.to_dict())))

    pd.testing.assert_series_equal(
        enc.transform(panel)[enc.output_column],
        revived.transform(panel)[enc.output_column],
    )


def test_it_refuses_a_stat_it_cannot_serialize(panel):
    with pytest.raises(ValueError, match="unknown stat"):
        GroupStatsEncoder(["store"], "value", "kurtosis")


def test_it_refuses_to_group_a_column_by_itself():
    with pytest.raises(ValueError, match="grouping a column by itself"):
        GroupStatsEncoder(["v"], "v", "mean")


def test_a_missing_group_key_says_it_is_needed_at_predict_time(panel):
    enc = GroupStatsEncoder(["store"], "value", "mean").fit(panel)
    with pytest.raises(ValueError, match="predict time"):
        enc.transform(panel.drop(columns=["store"]))


# --------------------------------------------------------------- out-of-fold


def test_a_row_does_not_contribute_to_the_statistic_it_receives():
    """The leak this would be without folds, made visible.

    Each group has two rows with wildly different values. Fitted in-fold, a row
    would see the group mean it helped make; out of fold it sees only its
    partner — which is the whole difference between a feature and a smuggled
    copy of the row.
    """
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "v": [0.0, 100.0, 0.0, 100.0],
            "y": [0.0, 1.0, 0.0, 1.0],
        }
    )
    folds = [(np.array([1, 3]), np.array([0, 2])), (np.array([0, 2]), np.array([1, 3]))]

    oof, final = fit_group_stats(df, ["g"], "v", "mean", folds=folds)
    col = final.output_column

    # Rows 0 and 2 hold 0.0 and were scored by folds trained on rows 1 and 3.
    assert oof[col].tolist() == [100.0, 0.0, 100.0, 0.0]
    # The encoder that ships is fitted on everything — correct at predict, where
    # no row of the request is in the training data at all.
    assert final.transform(df)[col].tolist() == [50.0] * 4


def test_forward_chaining_leaves_the_first_block_without_a_value():
    """The same gap out-of-fold target encoding leaves, for the same reason."""
    df = pd.DataFrame(
        {
            "g": list("abab") * 3,
            "v": np.arange(12.0),
            "t": np.repeat(np.arange(6), 2),
        }
    )
    folds = make_folds(df["v"], "regression", n_splits=2, order=df["t"].to_numpy())

    oof, final = fit_group_stats(df, ["g"], "v", "mean", folds=folds)

    assert oof[final.output_column].isna().sum() > 0


@pytest.mark.parametrize("stat", ["min", "mean"])
def test_a_non_numeric_value_is_refused_with_a_correction(stat):
    # A TypeError here ended a whole run; a ValueError reaches the agent as a retry.
    df = pd.DataFrame({
        "g": ["a", "a", "b"],
        "day": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
    })
    with pytest.raises(ValueError, match="needs numbers"):
        GroupStatsEncoder(["g"], "day", stat).fit(df)


def test_count_of_a_non_numeric_value_is_still_allowed():
    df = pd.DataFrame({
        "g": ["a", "a", "b"],
        "day": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
    })
    enc = GroupStatsEncoder(["g"], "day", "count").fit(df)
    assert enc.mapping_ == {"a": 2.0, "b": 1.0}
