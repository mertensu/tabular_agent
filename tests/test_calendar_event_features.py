"""The Easter computus and known-event distances, validated against Rossmann.

Rossmann labels ``StateHoliday`` ``'b' = Easter`` and ``'c' = Christmas``, so the
shipped dataset is ground truth for the computed calendar the ForecastingAgent
recipe (``docs/examples/calendar_features.py``) relies on. The point of computing
the movable feasts rather than joining a table is that they stay correct for the
future dates a forecast scores; ``test_easter_sunday_known_years`` locks that in
on a year no training data ever saw.
"""

import datetime
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "calendar_features", ROOT / "docs" / "examples" / "calendar_features.py"
)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)

ROSSMANN = ROOT / "rossmann_store_sales" / "train.csv"
_rossmann = pytest.mark.skipif(not ROSSMANN.exists(), reason="Rossmann data not present")


def test_easter_sunday_known_years():
    assert cf.easter_sunday(2013) == datetime.date(2013, 3, 31)
    assert cf.easter_sunday(2014) == datetime.date(2014, 4, 20)
    assert cf.easter_sunday(2015) == datetime.date(2015, 4, 5)
    # A far-future year the training data never saw.
    assert cf.easter_sunday(2030) == datetime.date(2030, 4, 21)


@_rossmann
def test_easter_family_matches_rossmann_b():
    d = pd.read_csv(ROSSMANN, dtype={"StateHoliday": str})
    b_dates = pd.to_datetime(sorted(d.loc[d.StateHoliday == "b", "Date"].unique()))
    assert len(b_dates) > 0
    for dt in b_dates:
        es = cf.easter_sunday(dt.year)
        offset = (dt.date() - es).days
        # 'b' rows are Good Friday (-2) or Easter Monday (+1).
        assert offset in (-2, 1), f"{dt.date()} is {offset}d from Easter {es}"


@_rossmann
def test_christmas_matches_rossmann_c():
    d = pd.read_csv(ROSSMANN, dtype={"StateHoliday": str})
    c_dates = pd.to_datetime(sorted(d.loc[d.StateHoliday == "c", "Date"].unique()))
    assert len(c_dates) > 0
    for dt in c_dates:
        assert dt.month == 12 and dt.day in (25, 26)


@_rossmann
def test_on_day_flag_and_distances():
    d = pd.read_csv(ROSSMANN, dtype={"StateHoliday": str})
    out = cf.add_event_distances(d, "Date", events=("christmas",))
    dd = pd.to_datetime(d["Date"])
    is_dec25 = ((dd.dt.month == 12) & (dd.dt.day == 25)).astype(int)
    assert (out["cal_is_christmas"].to_numpy() == is_dec25.to_numpy()).all()
    assert (out["cal_days_until_christmas"] >= 0).all()
    assert (out["cal_days_since_christmas"] >= 0).all()


@_rossmann
def test_calendar_parts_present_and_typed():
    d = pd.read_csv(ROSSMANN, dtype={"StateHoliday": str}).head(200)
    out = cf.add_calendar_parts(d, "Date")
    for col in (
        "cal_month",
        "cal_dayofweek",
        "cal_dayofyear",
        "cal_is_weekend",
        "cal_days_to_month_end",
    ):
        assert col in out.columns
    assert set(out["cal_is_weekend"].unique()) <= {0, 1}
    assert (out["cal_days_to_month_end"] >= 0).all()
