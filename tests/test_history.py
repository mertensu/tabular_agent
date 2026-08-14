"""HistoryEncoder: a group's own past, frozen at fit and replayed with its age.

Two properties carry every test here, and they are different properties:

* **Batch independence** (0011) — what a row reads comes from stored state, so a
  row alone and the same row in a batch get identical answers. This is what
  `groupby().shift()` inside `transform_features` cannot promise.
* **Serving symmetry** (0048) — a validation row reads history frozen at the end
  of its training prefix, exactly as a served row reads history frozen at `T`.
  The fold loop that produces this looks like `fit_group_stats`'s out-of-fold
  loop and exists for an unrelated reason; the tests that pin it down are the
  ones about *which* rows may be in the fitting set.
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.cv import make_folds
from tabular_agent.encoders import (
    ENCODER_CLASSES,
    HistoryEncoder,
    fit_history,
    to_periods,
)


@pytest.fixture
def panel():
    """Two stores, ten days each, with a per-store level in `sales`."""
    days = pd.date_range("2015-01-01", periods=10)
    return pd.DataFrame(
        {
            "store": np.tile(["a", "b"], 10),
            "date": np.repeat(days, 2),
            "sales": np.arange(20.0),
        }
    )


@pytest.fixture
def ordered_folds(panel):
    return make_folds(
        panel["sales"], "regression", n_splits=3, order=panel["date"].to_numpy()
    )


# --- what a stored history answers ------------------------------------------


def test_the_pair_is_the_last_value_and_its_distance_from_the_row(panel):
    enc = HistoryEncoder(["store"], "sales", "date").fit(panel)
    # Store "a" last appears on the 10th day with sales 18.
    served = pd.DataFrame({"store": ["a"], "date": pd.to_datetime(["2015-01-14"])})

    out = enc.transform(served)

    assert out["store_sales_latest"].iloc[0] == 18.0
    assert out["store_sales_latest_ago"].iloc[0] == 4.0


def test_the_answer_does_not_depend_on_the_batch(panel):
    """0011's property, and the reason this is a tool rather than free code."""
    enc = HistoryEncoder(["store"], "sales", "date").fit(panel)
    served = pd.DataFrame(
        {
            "store": ["a", "b", "a"],
            "date": pd.to_datetime(["2015-01-11", "2015-01-12", "2015-01-20"]),
        }
    )

    batch = enc.transform(served)
    alone = enc.transform(served.iloc[[2]])

    assert alone["store_sales_latest"].iloc[0] == batch["store_sales_latest"].iloc[2]
    assert alone["store_sales_latest_ago"].iloc[0] == batch["store_sales_latest_ago"].iloc[2]


def test_the_value_column_need_not_exist_at_predict_time(panel):
    """The point of stored history: the target has no column in a served frame."""
    enc = HistoryEncoder(["store"], "sales", "date").fit(panel)
    serving = pd.DataFrame({"store": ["a"], "date": pd.to_datetime(["2015-01-11"])})

    out = enc.transform(serving)

    assert "sales" not in serving.columns
    assert out["store_sales_latest"].iloc[0] == 18.0


def test_a_group_never_seen_gets_nothing_rather_than_a_population_value(panel):
    """Unlike a group *statistic*, there is no sensible fallback for "most recent"."""
    enc = HistoryEncoder(["store"], "sales", "date").fit(panel)

    out = enc.transform(
        pd.DataFrame({"store": ["zzz"], "date": pd.to_datetime(["2015-01-11"])})
    )

    assert np.isnan(out["store_sales_latest"].iloc[0])
    assert np.isnan(out["store_sales_latest_ago"].iloc[0])


def test_a_row_with_no_ordering_value_reads_nothing(panel):
    """Without a position there is no distance, and `searchsorted` would happily
    hand back the newest observation instead."""
    enc = HistoryEncoder(["store"], "sales", "date").fit(panel)

    out = enc.transform(pd.DataFrame({"store": ["a"], "date": [pd.NaT]}))

    assert np.isnan(out["store_sales_latest"].iloc[0])


def test_age_grows_across_the_horizon_while_the_value_stands_still(panel):
    """The whole argument for the pair: the number stops being right, and says so."""
    enc = HistoryEncoder(["store"], "sales", "date").fit(panel)
    served = pd.DataFrame(
        {
            "store": ["a"] * 3,
            "date": pd.to_datetime(["2015-01-11", "2015-01-15", "2015-02-01"]),
        }
    )

    out = enc.transform(served)

    assert out["store_sales_latest"].tolist() == [18.0, 18.0, 18.0]
    assert out["store_sales_latest_ago"].tolist() == [1.0, 5.0, 22.0]


# --- storage is bounded ------------------------------------------------------


def test_storage_is_bounded_by_keep_not_by_the_length_of_the_table():
    """The property that lets this live in a lineage at all."""
    days = pd.date_range("2015-01-01", periods=500)
    long = pd.DataFrame({"g": ["a"] * 500, "t": days, "v": np.arange(500.0)})

    enc = HistoryEncoder(["g"], "v", "t", keep=5).fit(long)

    assert len(enc.history_["a"][0]) == 5
    # The tail, not the head: it kept the most recent five.
    assert enc.history_["a"][1] == [495.0, 496.0, 497.0, 498.0, 499.0]


def test_keep_defaults_to_covering_the_largest_offset():
    enc = HistoryEncoder(["g"], "v", "t", offsets=[7])
    assert enc.keep == 8


# --- lags, and how they degrade ---------------------------------------------


def test_an_offset_reads_the_observation_at_or_before_the_anchor(panel, ordered_folds):
    enc = HistoryEncoder(["store"], "sales", "date", offsets=[3], keep=10).fit(panel)
    served = pd.DataFrame({"store": ["a"], "date": pd.to_datetime(["2015-01-11"])})

    out = enc.transform(served)

    # Anchor is 2015-01-08, store "a"'s value that day is 14.
    assert out["store_sales_3_ago"].iloc[0] == 14.0
    assert out["store_sales_3_ago_exact"].iloc[0] == 1.0


def test_an_unanswerable_offset_degrades_into_last_and_flags_itself(panel):
    """It does not lie about which observation it found."""
    enc = HistoryEncoder(["store"], "sales", "date", offsets=[3], keep=10).fit(panel)
    # Twenty days past the end of history: "three days ago" is not in history.
    served = pd.DataFrame({"store": ["a"], "date": pd.to_datetime(["2015-01-30"])})

    out = enc.transform(served)

    assert out["store_sales_3_ago"].iloc[0] == 18.0
    # It found something 20 back, not 3, and says so.
    assert out["store_sales_3_ago_exact"].iloc[0] == 0.0


def test_offsets_are_refused_without_a_declared_horizon(panel, ordered_folds):
    with pytest.raises(ValueError, match="forecast_horizon"):
        fit_history(
            panel, ["store"], "sales", "date", folds=ordered_folds, offsets=[1]
        )


def test_offsets_shorter_than_the_horizon_are_refused(panel, ordered_folds):
    """A `_lag1` column on a block starting seven days out holds a seven-day-old
    number under a name that promises one day."""
    with pytest.raises(ValueError, match="shorter than the declared forecast horizon"):
        fit_history(
            panel, ["store"], "sales", "date", folds=ordered_folds,
            offsets=[1], forecast_horizon=7,
        )


def test_an_offset_at_the_horizon_is_allowed(panel, ordered_folds):
    oof, enc = fit_history(
        panel, ["store"], "sales", "date", folds=ordered_folds,
        offsets=[7], forecast_horizon=7, keep=10,
    )

    assert "store_sales_7_ago" in oof.columns
    assert "store_sales_7_ago_exact" in oof.columns


# --- the per-fold freeze (0048) ---------------------------------------------


def test_random_folds_are_refused(panel):
    """0048: history frozen at a scattered training fold reports an age that is an
    artifact of the seed and matches no serving contract."""
    random_folds = make_folds(panel["sales"], "regression", n_splits=3)

    with pytest.raises(ValueError, match="forward-chaining folds"):
        fit_history(panel, ["store"], "sales", "date", folds=random_folds)


def test_the_first_block_is_never_validated_and_keeps_nan(panel, ordered_folds):
    oof, _ = fit_history(panel, ["store"], "sales", "date", folds=ordered_folds)

    validated = np.unique(np.concatenate([va for _, va in ordered_folds]))
    unvalidated = sorted(set(range(len(panel))) - set(validated.tolist()))

    assert unvalidated, "forward chaining should leave the first block unvalidated"
    assert oof["store_sales_latest"].iloc[unvalidated].isna().all()


def test_a_validation_row_never_reads_its_own_or_a_later_period(panel, ordered_folds):
    """The leak this would have if history were fitted on "everything outside the
    validation fold" — the rule `fit_group_stats` correctly follows and this one
    must not."""
    oof, _ = fit_history(panel, ["store"], "sales", "date", folds=ordered_folds)

    values = oof["store_sales_latest"]
    ages = oof["store_sales_latest_ago"]
    seen = ~values.isna()

    # Every value read is strictly older than the row that read it.
    assert (ages[seen] > 0).all()
    # And it is a value that genuinely occurred earlier in the same store.
    for i in np.flatnonzero(seen.to_numpy()):
        row = panel.iloc[i]
        earlier = panel[(panel["store"] == row["store"]) & (panel["date"] < row["date"])]
        assert values.iloc[i] in set(earlier["sales"].tolist())


def test_the_freeze_makes_age_grow_across_a_validation_block(panel, ordered_folds):
    """Serving symmetry: within one block the feature goes stale exactly as it
    would across a real forecast horizon, instead of being refreshed daily."""
    oof, _ = fit_history(panel, ["store"], "sales", "date", folds=ordered_folds)

    _, va = ordered_folds[-1]
    block = panel.iloc[va]
    a_rows = va[(block["store"] == "a").to_numpy()]
    ages = oof["store_sales_latest_ago"].iloc[a_rows].to_numpy()

    assert len(ages) > 1
    assert (np.diff(ages) > 0).all(), "age must grow across the block, not reset"


def test_a_naive_backward_lag_would_look_fresher_than_the_frozen_one(panel, ordered_folds):
    """0048's stated cost, pinned so it is not later mistaken for a bug: the honest
    feature is staler than the one that reads neighbouring validation rows."""
    oof, _ = fit_history(panel, ["store"], "sales", "date", folds=ordered_folds)

    frozen_age = oof["store_sales_latest_ago"].dropna()
    # A naive within-store shift would give every row an age of exactly one day.
    assert frozen_age.max() > 1.0


# --- serialisation and replay ------------------------------------------------


def test_the_encoder_survives_a_round_trip(panel):
    enc = HistoryEncoder(["store"], "sales", "date", offsets=[2], keep=6).fit(panel)
    served = pd.DataFrame({"store": ["a", "b"], "date": pd.to_datetime(["2015-01-12"] * 2)})

    revived = ENCODER_CLASSES[enc.kind].from_dict(enc.to_dict())

    pd.testing.assert_frame_equal(enc.transform(served), revived.transform(served))


def test_it_replays_through_a_pipeline(panel, ordered_folds):
    from tabular_agent.pipeline import Pipeline

    _, enc = fit_history(panel, ["store"], "sales", "date", folds=ordered_folds)
    pipe = Pipeline("regression")
    pipe.add_encoder(enc)
    served = pd.DataFrame({"store": ["a"], "date": pd.to_datetime(["2015-01-11"])})

    out = Pipeline.from_dict(pipe.to_dict()).transform(served)

    assert out["store_sales_latest"].iloc[0] == 18.0
    assert out["store_sales_latest_ago"].iloc[0] == 1.0


# --- the ordering column -----------------------------------------------------


def test_iso_date_strings_are_accepted():
    """Every table loaded from CSV has its dates as text; refusing them would
    refuse the datasets history exists for."""
    periods = to_periods(pd.Series(["2015-01-01", "2015-01-02", "2015-01-09"]))

    assert np.diff(periods).tolist() == [1.0, 7.0]


def test_an_ambiguous_date_format_is_refused_rather_than_guessed():
    """`pd.to_datetime` left to itself reads '01/02/2015' as either the 1st of
    February or the 2nd of January depending on the rest of the column."""
    with pytest.raises(ValueError, match="not ISO 8601"):
        to_periods(pd.Series(["01/02/2015", "02/02/2015"]))


def test_a_non_subtractable_column_is_refused():
    with pytest.raises(ValueError, match="subtract"):
        to_periods(pd.Series(pd.Categorical(["a", "b"])))


def test_a_numeric_ordering_column_keeps_its_own_units():
    df = pd.DataFrame({"g": ["a"] * 4, "t": [1, 2, 3, 4], "v": [10.0, 20.0, 30.0, 40.0]})
    enc = HistoryEncoder(["g"], "v", "t").fit(df)

    out = enc.transform(pd.DataFrame({"g": ["a"], "t": [9]}))

    assert out["g_v_latest"].iloc[0] == 40.0
    assert out["g_v_latest_ago"].iloc[0] == 5.0


def test_dates_are_measured_in_days_whatever_the_datetime_resolution():
    """pandas hands back datetime64 in several units; assuming one puts every age
    out by a factor of a thousand."""
    days = pd.date_range("2015-01-01", periods=3)
    for unit in ("s", "ms", "us", "ns"):
        periods = to_periods(pd.Series(days.astype(f"datetime64[{unit}]")))
        assert np.diff(periods).tolist() == [1.0, 1.0]


# --- construction refusals ---------------------------------------------------


def test_grouping_a_series_by_its_own_timestamp_is_refused():
    with pytest.raises(ValueError, match="no history at all"):
        HistoryEncoder(["date"], "sales", "date")


def test_a_zero_or_negative_offset_is_refused():
    with pytest.raises(ValueError, match="offsets must be positive"):
        HistoryEncoder(["store"], "sales", "date", offsets=[0])
    with pytest.raises(ValueError, match="offsets must be positive"):
        HistoryEncoder(["store"], "sales", "date", offsets=[-1])


# --- trailing windows (slice 3) ----------------------------------------------
#
# The panel gives store "a" one observation a day for ten days, sales 0, 2, 4 …
# 18, so a window's contents are checkable by hand throughout.


def test_a_window_averages_the_rows_own_stretch_of_the_past(panel):
    enc = HistoryEncoder(["store"], "sales", "date", windows=[(3, "mean")], keep=10)
    served = pd.DataFrame({"store": ["a"], "date": pd.to_datetime(["2015-01-11"])})

    out = enc.fit(panel).transform(served)

    # [01-08, 01-11) holds the 8th, 9th and 10th: 14, 16, 18.
    assert out["store_sales_3_mean"].iloc[0] == 16.0
    assert out["store_sales_3_count"].iloc[0] == 3.0


def test_the_window_moves_with_the_row_rather_than_with_the_freeze(panel):
    """0050: the property the whole slice exists for. A boundary-anchored value is
    one draw shared by every served row; this must be a different stretch per row."""
    enc = HistoryEncoder(["store"], "sales", "date", windows=[(3, "mean")], keep=10)
    served = pd.DataFrame(
        {"store": ["a"] * 3, "date": pd.to_datetime(["2015-01-11", "2015-01-12", "2015-01-13"])}
    )

    out = enc.fit(panel).transform(served)

    # 14,16,18 -> 16,18 -> 18: each row draws its own window, and the count falls
    # away as the row moves past the end of stored history.
    assert out["store_sales_3_mean"].tolist() == [16.0, 17.0, 18.0]
    assert out["store_sales_3_count"].tolist() == [3.0, 2.0, 1.0]
    # `_latest` is the same number for all three, which is the contrast.
    assert out["store_sales_latest"].tolist() == [18.0, 18.0, 18.0]


def test_a_row_is_never_inside_its_own_window(panel):
    """The right endpoint is strict. A self-included point in a mean over many is
    diluted to invisibility, so this cannot be left to the per-fold freeze."""
    enc = HistoryEncoder(["store"], "sales", "date", windows=[(3, "mean")], keep=10)

    out = enc.fit(panel).transform(panel)
    a = out[out["store"] == "a"]

    # The 4th day's own sales are 6, and its window is the 1st-3rd: 0, 2, 4.
    assert a["store_sales_3_mean"].iloc[3] == 2.0
    # The first row has no past at all, and gets an empty window rather than itself.
    assert a["store_sales_3_count"].iloc[0] == 0.0
    assert np.isnan(a["store_sales_3_mean"].iloc[0])


def test_an_empty_window_is_a_zero_count_and_a_missing_mean(panel):
    """Distinct from a group that was never seen, which knows nothing at all."""
    enc = HistoryEncoder(["store"], "sales", "date", windows=[(3, "mean")], keep=10)
    enc.fit(panel)

    past_the_end = enc.transform(
        pd.DataFrame({"store": ["a"], "date": pd.to_datetime(["2015-02-20"])})
    )
    unseen = enc.transform(
        pd.DataFrame({"store": ["zzz"], "date": pd.to_datetime(["2015-01-11"])})
    )

    assert past_the_end["store_sales_3_count"].iloc[0] == 0.0
    assert np.isnan(past_the_end["store_sales_3_mean"].iloc[0])
    # Nothing was stored for this group, so even the count is unknown.
    assert np.isnan(unseen["store_sales_3_count"].iloc[0])


def test_the_count_is_emitted_once_per_length_however_many_stats_are_asked(panel):
    enc = HistoryEncoder(
        ["store"], "sales", "date",
        windows=[(3, "mean"), (3, "std"), (5, "median")], keep=10,
    )

    assert enc.output_columns[-3:] == [
        "store_sales_5_median", "store_sales_3_count", "store_sales_5_count",
    ]
    assert enc.output_columns.count("store_sales_3_count") == 1


def test_the_spread_of_a_single_observation_is_missing_not_zero(panel):
    """A confident zero off one point says "this group never moves", which is the
    same class of lie as a mean over one."""
    enc = HistoryEncoder(["store"], "sales", "date", windows=[(3, "std")], keep=10)
    served = pd.DataFrame(
        {"store": ["a", "a"], "date": pd.to_datetime(["2015-01-11", "2015-01-13"])}
    )

    out = enc.fit(panel).transform(served)

    # 14, 16, 18 -> sample sd 2; then a single observation -> nothing.
    assert out["store_sales_3_std"].iloc[0] == pytest.approx(2.0)
    assert np.isnan(out["store_sales_3_std"].iloc[1])


def test_a_window_of_one_repeated_value_has_a_spread_of_zero():
    """Zero is the honest answer when there really were several observations, and
    float error must not turn it into a NaN that reads as "no data"."""
    df = pd.DataFrame({"g": ["a"] * 4, "t": [1, 2, 3, 4], "v": [7.0, 7.0, 7.0, 7.0]})
    enc = HistoryEncoder(["g"], "v", "t", windows=[(3, "std")], keep=10).fit(df)

    out = enc.transform(pd.DataFrame({"g": ["a"], "t": [5]}))

    assert out["g_v_3_std"].iloc[0] == 0.0


def test_the_median_ignores_a_spike_the_mean_carries():
    """0050's failure in miniature: the last open day before a holiday sold 53%
    above normal, and it is the newest observation that a mean weights fully."""
    df = pd.DataFrame(
        {"g": ["a"] * 5, "t": [1, 2, 3, 4, 5], "v": [10.0, 10.0, 10.0, 10.0, 100.0]}
    )
    enc = HistoryEncoder(
        ["g"], "v", "t", windows=[(5, "mean"), (5, "median")], keep=10
    ).fit(df)

    out = enc.transform(pd.DataFrame({"g": ["a"], "t": [6]}))

    assert out["g_v_5_mean"].iloc[0] == 28.0
    assert out["g_v_5_median"].iloc[0] == 10.0


def test_a_window_answer_does_not_depend_on_the_batch(panel):
    """0011, for the new columns: stored state, so one row alone and the same row
    in company get identical answers."""
    enc = HistoryEncoder(
        ["store"], "sales", "date", windows=[(4, "mean"), (4, "std")], keep=10
    ).fit(panel)
    served = pd.DataFrame(
        {"store": ["a", "b", "a"], "date": pd.to_datetime(["2015-01-11"] * 3)}
    )

    batch = enc.transform(served)
    alone = enc.transform(served.iloc[[2]])

    for col in ("store_sales_4_mean", "store_sales_4_std", "store_sales_4_count"):
        assert alone[col].iloc[0] == batch[col].iloc[2]


# --- windows: what is refused ------------------------------------------------


def test_windows_need_a_declared_horizon(panel, ordered_folds):
    with pytest.raises(ValueError, match="forecast_horizon"):
        fit_history(
            panel, ["store"], "sales", "date", folds=ordered_folds,
            windows=[(3, "mean")],
        )


def test_a_window_shorter_than_the_horizon_is_refused(panel, ordered_folds):
    """Stored history stops one horizon before the earliest row predicted, so such
    a window covers only the gap and is empty for every served row."""
    with pytest.raises(ValueError, match="shorter than the declared forecast horizon"):
        fit_history(
            panel, ["store"], "sales", "date", folds=ordered_folds,
            windows=[(2, "mean")], forecast_horizon=5,
        )


def test_an_unknown_window_statistic_is_refused():
    with pytest.raises(ValueError, match="unknown window statistic"):
        HistoryEncoder(["store"], "sales", "date", windows=[(3, "mode")])


def test_min_and_max_are_off_the_menu():
    """0046's reasoning at a shorter range: an extreme over a trailing window is
    close to a single observation, which is what 0049 was written about."""
    for stat in ("min", "max", "sum"):
        with pytest.raises(ValueError, match="deliberately absent"):
            HistoryEncoder(["store"], "sales", "date", windows=[(3, stat)])


def test_a_bare_length_is_refused_rather_than_assumed_to_mean_the_mean():
    with pytest.raises(ValueError, match=r"\(length, statistic\) pairs"):
        HistoryEncoder(["store"], "sales", "date", windows=[90])


def test_a_zero_or_negative_window_is_refused():
    with pytest.raises(ValueError, match="window lengths must be positive"):
        HistoryEncoder(["store"], "sales", "date", windows=[(0, "mean")])


def test_keep_defaults_to_covering_the_longest_window():
    enc = HistoryEncoder(["g"], "v", "t", offsets=[7], windows=[(90, "mean")])
    assert enc.keep == 91


def test_a_window_too_long_to_store_is_refused_not_quietly_shortened():
    """A window that averages the most recent 1000 observations of a longer series
    is a wrong number under a confident name."""
    with pytest.raises(ValueError, match="more history than an encoder may carry"):
        HistoryEncoder(["g"], "v", "t", windows=[(5000, "mean")])


def test_a_sparse_series_may_say_so_with_an_explicit_keep():
    """Weekly readings over a year-long window need 52 slots, not 365, and the
    caller is the only one who knows that."""
    enc = HistoryEncoder(["g"], "v", "t", windows=[(5000, "mean")], keep=60)
    assert enc.keep == 60


# --- windows: serialisation --------------------------------------------------


def test_a_windowed_encoder_survives_a_round_trip(panel):
    enc = HistoryEncoder(
        ["store"], "sales", "date", windows=[(4, "mean"), (4, "median")], keep=10
    ).fit(panel)
    served = pd.DataFrame({"store": ["a", "b"], "date": pd.to_datetime(["2015-01-12"] * 2)})

    import json

    revived = ENCODER_CLASSES[enc.kind].from_dict(json.loads(json.dumps(enc.to_dict())))

    pd.testing.assert_frame_equal(enc.transform(served), revived.transform(served))


def test_a_bundle_written_before_windows_existed_still_loads(panel):
    enc = HistoryEncoder(["store"], "sales", "date", offsets=[2], keep=6).fit(panel)
    old = {k: v for k, v in enc.to_dict().items() if k != "windows"}

    revived = ENCODER_CLASSES[enc.kind].from_dict(old)

    assert revived.windows == []
    assert "store_sales_2_ago" in revived.output_columns


def test_windows_are_frozen_per_fold_like_everything_else(panel, ordered_folds):
    """The freeze is what makes a validation row's window match a served row's, so
    the count has to fall away across a block rather than staying full."""
    oof, _ = fit_history(
        panel, ["store"], "sales", "date", folds=ordered_folds,
        windows=[(4, "mean")], forecast_horizon=1, keep=10,
    )

    _, va = ordered_folds[-1]
    block = oof.iloc[va]
    counts = block["store_sales_4_count"].dropna()

    assert len(counts) > 0
    # Later rows in the block are further past the freeze, so their windows hold
    # less — the same degradation `_latest_ago` reports as a growing age.
    assert counts.iloc[-1] < counts.iloc[0]
