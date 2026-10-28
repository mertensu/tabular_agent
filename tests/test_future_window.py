"""Invariants for FutureWindowEncoder and fit_future_covariate_window (0091).

The forward, exogenous counterpart to fit_history. These check the three
properties the block invariant rests on — order is restored, no aggregate crosses
a series or block boundary, and a delivered serving block reproduces exactly what
training computed for the same rows — plus the leakage rule (only a future-known
covariate is eligible; the target and undeclared columns are refused) and a clean
round trip through the lineage.
"""
import numpy as np
import pandas as pd
import pytest

from tabular_agent.encoders import (
    FutureWindowEncoder,
    fit_future_covariate_window,
)

BLOCK_ROWS = 672  # 7 days at 15-minute resolution, one delivered block
HORIZON = 7.0  # block length in days, the forecast horizon


def _panel(series=2, blocks=3):
    """A clean panel: `series` ids, each `blocks` whole 7-day blocks of 15-min steps."""
    rows = []
    for sid in range(1, series + 1):
        n = blocks * BLOCK_ROWS
        t = pd.date_range("2018-01-01", periods=n, freq="15min")
        rng = np.random.default_rng(sid)
        rows.append(pd.DataFrame({
            "time": t,
            "T2m": rng.normal(12, 6, n),
            "Gb(i)": np.abs(rng.normal(200, 100, n)),
            "dataset_id": sid,
            # a training-only target that must never enter the features:
            "residual_load": rng.normal(52, 144, n),
        }))
    return pd.concat(rows, ignore_index=True)


AGGS = [
    {"family": "period", "unit": "day", "stat": "sum"},
    {"family": "period", "unit": "day", "stat": "max"},
    {"family": "period", "unit": "block", "stat": "mean"},
    {"family": "trailing", "window": 4, "stat": "mean"},
    {"family": "trailing", "window": 16, "stat": "std"},
    {"family": "forward", "window": 12, "stat": "mean"},
]

#: Same families without `std`. A std over fewer than two observations is NaN by
#: design (a single point has no spread — the choice HistoryEncoder makes too), so
#: the "every value is finite" invariants use this set and the std behaviour is
#: checked on its own below.
AGGS_FINITE = [s for s in AGGS if s["stat"] != "std"]


def _encoder(value="T2m", by=("dataset_id",), aggregates=AGGS):
    return FutureWindowEncoder(list(by), value, "time", HORIZON, aggregates)


def test_index_and_order_are_restored_even_when_shuffled():
    df = _panel().sample(frac=1.0, random_state=7)
    enc = _encoder(aggregates=AGGS_FINITE).fit(df)
    x = enc.transform(df)[enc.output_columns]
    assert x.index.equals(df.index)
    assert len(x) == len(df)
    assert np.isfinite(x.to_numpy()).all()


def test_std_is_nan_only_where_a_window_holds_one_observation():
    df = _panel(series=1, blocks=1)
    enc = FutureWindowEncoder([], "T2m", "time", HORIZON, [
        {"family": "trailing", "window": 16, "stat": "std"},
        {"family": "trailing", "window": 16, "stat": "count"},
    ])
    x = enc.transform(df)
    nan_std = x["T2m_roll16_std"].isna()
    one_obs = x["T2m_roll16_count"] == 1
    # a std is missing exactly when the window held a single point, never otherwise
    assert (nan_std == one_obs).all()


def test_shuffle_does_not_change_a_rows_features():
    df = _panel()
    enc = _encoder().fit(df)
    ordered = enc.transform(df)[enc.output_columns]
    shuffled = enc.transform(df.sample(frac=1.0, random_state=3))[enc.output_columns]
    pd.testing.assert_frame_equal(
        ordered.sort_index(), shuffled.sort_index(), check_like=True
    )


def test_no_target_or_extra_column_enters_the_features():
    df = _panel()
    _, enc = fit_future_covariate_window(df, ["dataset_id"], "T2m", "time", HORIZON, AGGS)
    assert "residual_load" not in enc.output_columns
    # the covariate value never has to survive into the output either — only its
    # aggregates do
    assert all(c.startswith("dataset_id_T2m_") for c in enc.output_columns)


def test_aggregates_never_cross_a_series_boundary():
    df = _panel(series=2, blocks=2)
    enc = FutureWindowEncoder(["dataset_id"], "T2m", "time", HORIZON,
                              [{"family": "period", "unit": "block", "stat": "mean"}])
    x = enc.transform(df)
    col = "dataset_id_T2m_block_mean"
    # the whole-block mean is a per-(series, block) constant; the two series' most
    # recent blocks must not share a value, i.e. no group spanned both series
    b0 = x.loc[df.dataset_id == 1, col].iloc[-1]
    b0_other = x.loc[df.dataset_id == 2, col].iloc[-1]
    assert b0 != b0_other


def test_a_delivered_block_reproduces_training_exactly():
    """Serving-legality: the next block served alone == what training computes for it.

    The encoder is fitted on the past only, as a shipped one is; the block that
    follows is then delivered on its own, with nothing before it in the frame.
    Every family must give it the values a frame holding the whole series gives.
    """
    df = _panel(series=1, blocks=3)
    past, block = df.iloc[:-BLOCK_ROWS], df.iloc[-BLOCK_ROWS:]
    enc = _encoder(by=()).fit(past)  # single series
    trained = enc.transform(df)[enc.output_columns]
    served = block[["time", "T2m", "Gb(i)"]].copy()  # serving cols only
    replayed = enc.transform(served)[enc.output_columns]
    pd.testing.assert_frame_equal(
        trained.iloc[-BLOCK_ROWS:].reset_index(drop=True),
        replayed.reset_index(drop=True),
        check_like=True,
    )


def test_trailing_reaches_back_and_forward_is_bounded_at_the_block_edge():
    df = _panel(series=1, blocks=2)
    enc = FutureWindowEncoder([], "T2m", "time", HORIZON, [
        {"family": "trailing", "window": 4, "stat": "count"},
        {"family": "forward", "window": 4, "stat": "count"},
    ]).fit(df)
    x = enc.transform(df)
    # Newest block is block 0 = the final BLOCK_ROWS rows. Its first row reads the
    # three rows before the block: the covariate's past is known (0093). Its last
    # row looks ahead at nothing: nothing past the block end is delivered.
    block0 = x.iloc[-BLOCK_ROWS:]
    assert block0["T2m_roll4_count"].iloc[0] == 4
    assert block0["T2m_fwd4_count"].iloc[-1] == 1
    # Only the series' very first row has no past at all.
    assert x["T2m_roll4_count"].iloc[0] == 1
    assert block0["T2m_fwd4_count"].iloc[10] == 4


def test_a_served_block_alone_reads_the_stored_past():
    df = _panel(series=2, blocks=2)
    past = df.groupby("dataset_id").head(BLOCK_ROWS)
    block = df.groupby("dataset_id").tail(BLOCK_ROWS)
    enc = FutureWindowEncoder(["dataset_id"], "T2m", "time", HORIZON, [
        {"family": "trailing", "window": 4, "stat": "mean"},
    ]).fit(past)
    served = enc.transform(block[["dataset_id", "time", "T2m"]])
    for sid in (1, 2):
        rows = df[df.dataset_id == sid]
        first = rows.index[BLOCK_ROWS]
        expected = rows.loc[first - 3:first, "T2m"].mean()  # three stored + itself
        assert served.loc[first, "dataset_id_T2m_roll4_mean"] == pytest.approx(expected)


def test_a_gap_before_the_delivered_rows_is_refused():
    """0095: a window over fewer periods than training saw is not computed."""
    df = _panel(series=1, blocks=3)
    enc = FutureWindowEncoder([], "T2m", "time", HORIZON, [
        {"family": "trailing", "window": 4, "stat": "count"},
    ]).fit(df.iloc[:BLOCK_ROWS])
    with pytest.raises(ValueError, match="known_past") as err:
        enc.transform(df.iloc[-BLOCK_ROWS:][["time", "T2m"]])
    # It names the column, how much past is needed and where the record ends.
    assert "'T2m'" in str(err.value) and "3 rows" in str(err.value)
    assert "last recorded 2018-01-07 23:45:00" in str(err.value)
    assert "first delivered 2018-01-15" in str(err.value)


def test_known_past_closes_the_gap_and_reproduces_the_whole_series():
    df = _panel(series=2, blocks=3)
    aggs = [{"family": "trailing", "window": 4, "stat": "sum"},
            {"family": "forward", "window": 4, "stat": "sum"}]
    enc = FutureWindowEncoder(["dataset_id"], "T2m", "time", HORIZON, aggs)
    enc.fit(df.groupby("dataset_id").head(BLOCK_ROWS))
    block = df.groupby("dataset_id").tail(BLOCK_ROWS)
    # The block in the middle is the gap: known, but not in the forecast rows.
    middle = df.drop(df.groupby("dataset_id").head(BLOCK_ROWS).index).drop(block.index)
    served = enc.transform(block[["dataset_id", "time", "T2m"]],
                           known_past=middle[["dataset_id", "time", "T2m"]])
    whole = enc.transform(df)
    pd.testing.assert_frame_equal(served[enc.output_columns],
                                  whole.loc[block.index, enc.output_columns])


def test_known_past_alone_is_enough_without_a_stored_tail():
    df = _panel(series=1, blocks=2)
    enc = FutureWindowEncoder([], "T2m", "time", HORIZON, [
        {"family": "trailing", "window": 4, "stat": "mean"},
    ]).fit(df.iloc[:BLOCK_ROWS])
    enc.tail_ = {}
    block = df.iloc[-BLOCK_ROWS:][["time", "T2m"]]
    served = enc.transform(block, known_past=df.iloc[BLOCK_ROWS - 3:BLOCK_ROWS])
    assert served["T2m_roll4_mean"].iloc[0] == pytest.approx(
        df["T2m"].iloc[BLOCK_ROWS - 3:BLOCK_ROWS + 1].mean())


def test_known_past_must_carry_the_columns_the_windows_read():
    df = _panel(series=1, blocks=2)
    enc = FutureWindowEncoder([], "T2m", "time", HORIZON, [
        {"family": "trailing", "window": 4, "stat": "mean"},
    ]).fit(df.iloc[:BLOCK_ROWS])
    with pytest.raises(ValueError, match="known_past lacks"):
        enc.transform(df.iloc[-BLOCK_ROWS:], known_past=df[["time"]])


def test_a_replay_of_the_training_rows_needs_no_past():
    """Rows that start where training started are at the series start, as in fit."""
    df = _panel(series=2, blocks=2)
    enc = FutureWindowEncoder(["dataset_id"], "T2m", "time", HORIZON, [
        {"family": "trailing", "window": 16, "stat": "mean"},
    ]).fit(df)
    assert enc.transform(df)["dataset_id_T2m_roll16_mean"].notna().all()


def test_an_encoder_written_before_lookback_still_resets():
    """A bundle on disk keeps predicting what its model was trained against."""
    df = _panel(series=1, blocks=2)
    enc = FutureWindowEncoder([], "T2m", "time", HORIZON, [
        {"family": "trailing", "window": 4, "stat": "count"},
    ]).fit(df)
    legacy = enc.to_dict()
    for key in ("lookback", "tail", "spacing"):
        legacy.pop(key)
    x = FutureWindowEncoder.from_dict(legacy).transform(df)
    assert x["T2m_roll4_count"].iloc[-BLOCK_ROWS] == 1


def test_a_text_covariate_is_refused_rather_than_read_as_missing():
    df = _panel(series=1, blocks=1)
    df["holiday"] = np.where(np.arange(len(df)) % 50 == 0, "a", "0")
    with pytest.raises(ValueError, match="need numbers"):
        fit_future_covariate_window(df, [], "holiday", "time", HORIZON,
                                    [{"family": "period", "unit": "block", "stat": "sum"}])


def test_dst_duplicate_timestamps_keep_source_order():
    df = _panel(series=1, blocks=1).reset_index(drop=True)
    df.loc[11, "time"] = df.loc[10, "time"]  # duplicate a timestamp
    enc = _encoder(by=(), aggregates=AGGS_FINITE).fit(df)
    x = enc.transform(df)
    assert x.index.equals(df.index)
    assert np.isfinite(x[enc.output_columns].to_numpy()).all()


def test_roundtrip_through_dict_is_identity():
    enc = _encoder()
    restored = FutureWindowEncoder.from_dict(enc.to_dict())
    assert restored.output_columns == enc.output_columns
    df = _panel()
    pd.testing.assert_frame_equal(
        enc.transform(df)[enc.output_columns],
        restored.transform(df)[restored.output_columns],
    )


def test_bad_specs_are_refused():
    with pytest.raises(ValueError):
        FutureWindowEncoder([], "T2m", "time", HORIZON, [])  # nothing to build
    with pytest.raises(ValueError):
        FutureWindowEncoder([], "T2m", "time", 0.0, AGGS)  # non-positive block
    with pytest.raises(ValueError):
        FutureWindowEncoder([], "T2m", "time", HORIZON,
                            [{"family": "nope", "stat": "mean"}])
    with pytest.raises(ValueError):
        FutureWindowEncoder([], "T2m", "time", HORIZON,
                            [{"family": "period", "unit": "fortnight", "stat": "mean"}])
    with pytest.raises(ValueError):
        FutureWindowEncoder([], "T2m", "time", HORIZON,
                            [{"family": "trailing", "window": 0, "stat": "mean"}])
    with pytest.raises(ValueError):
        FutureWindowEncoder([], "T2m", "time", HORIZON,
                            [{"family": "period", "unit": "day", "stat": "quantile"}])
