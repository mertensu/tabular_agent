"""Stateless calendar and known-event features for a ForecastingAgent run.

Copy the function bodies into ``transform_features(code=...)``. Every column here
is a pure function of each row's date, so it is row-local: it survives the escape
hatch's stability check and replays identically at predict time — future dates
included, because the Easter family is *computed*, not looked up in a table that
ends where the training data ends.

No sin/cos on purpose. A gradient-boosted tree splits raw integer cycle parts
(month, day-of-week, day-of-year) natively; the wrap-around smoothing that sin/cos
buys mainly helps linear/NN models (see ``docs/time_aware_features.md`` and
decision 0089). Add cyclical encodings only if you later fit a non-tree model.

What is deliberately NOT here: a distance to a *data-observed* event — "days since
this store was last closed", "days since the last sale". That reads other rows'
outcomes, so it is not row-local; it belongs to ``fit_history`` and the temporal
seam, not to this escape hatch.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


def add_calendar_parts(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Integer calendar parts a tree can split on directly."""
    d = pd.to_datetime(df[date_col])
    out = df.copy()
    out["cal_year"] = d.dt.year
    out["cal_month"] = d.dt.month
    out["cal_day"] = d.dt.day
    out["cal_dayofweek"] = d.dt.dayofweek
    out["cal_dayofyear"] = d.dt.dayofyear
    out["cal_week"] = d.dt.isocalendar().week.astype("int64")
    out["cal_quarter"] = d.dt.quarter
    out["cal_is_weekend"] = (d.dt.dayofweek >= 5).astype("int64")
    out["cal_is_month_start"] = d.dt.is_month_start.astype("int64")
    out["cal_is_month_end"] = d.dt.is_month_end.astype("int64")
    out["cal_days_to_month_end"] = (d.dt.days_in_month - d.dt.day).astype("int64")
    return out


def easter_sunday(year: int) -> datetime.date:
    """Gregorian Easter Sunday (Anonymous Gregorian algorithm / Meeus-Jones-Butcher).

    Exact for every year in the Gregorian calendar, so it works for the future
    dates a forecast must score — the reason we *compute* the movable feasts
    rather than join a holiday table that stops at the end of training.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def known_events(years) -> dict[str, list[datetime.date]]:
    """Fixed-date and Easter-derived events, per year — no external data needed."""
    out: dict[str, list[datetime.date]] = {
        "new_year": [],
        "christmas": [],
        "boxing_day": [],
        "easter_sunday": [],
        "good_friday": [],
        "easter_monday": [],
        "ascension": [],
        "whit_monday": [],
    }
    for y in years:
        es = easter_sunday(y)
        out["new_year"].append(datetime.date(y, 1, 1))
        out["christmas"].append(datetime.date(y, 12, 25))
        out["boxing_day"].append(datetime.date(y, 12, 26))
        out["easter_sunday"].append(es)
        out["good_friday"].append(es - datetime.timedelta(days=2))
        out["easter_monday"].append(es + datetime.timedelta(days=1))
        out["ascension"].append(es + datetime.timedelta(days=39))
        out["whit_monday"].append(es + datetime.timedelta(days=50))
    return out


def add_event_distances(
    df: pd.DataFrame,
    date_col: str,
    events: tuple[str, ...] = ("christmas", "easter_sunday", "good_friday", "new_year"),
) -> pd.DataFrame:
    """``days_until`` / ``days_since`` each named known event, plus an on-day flag.

    The two distances are separate non-negative columns — a form a tree reads
    cleanly. Both are pure functions of the row's date against a *computed*
    calendar, so predict-time rows past the end of training get correct values.
    """
    d = pd.to_datetime(df[date_col]).dt.normalize()
    out = df.copy()
    day = (d - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    years = range(int(d.dt.year.min()) - 1, int(d.dt.year.max()) + 2)
    cal = known_events(years)
    epoch = datetime.date(1970, 1, 1)
    for name in events:
        ev = np.array(sorted((dt - epoch).days for dt in cal[name]))
        nxt = np.searchsorted(ev, day, side="left").clip(max=len(ev) - 1)
        prv = (np.searchsorted(ev, day, side="right") - 1).clip(min=0)
        out[f"cal_days_until_{name}"] = (ev[nxt] - day).clip(min=0)
        out[f"cal_days_since_{name}"] = (day - ev[prv]).clip(min=0)
        out[f"cal_is_{name}"] = (out[f"cal_days_until_{name}"] == 0).astype("int64")
    return out
