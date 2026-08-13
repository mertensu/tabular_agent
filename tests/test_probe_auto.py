"""Every table is scored the moment it is created (0035/0036).

The point of automatic scoring is coverage: across the runs in `_runs/`, 22 of
the 24 tables that recombined an abandoned branch had no number at all, because
the agent trains only the endpoint of a chain. These tests pin the two things
that make the coverage trustworthy — that the score arrives without being asked
for, and that failing to score never costs the agent the table.
"""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal, tools
from tabular_agent.harness.session import RunSession
from tabular_agent.taskspec import TaskSpec


@pytest.fixture
def session(tmp_path):
    rng = np.random.default_rng(0)
    n = 240
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "city": rng.choice(["berlin", "hamburg", "koeln"], size=n),
        "y": 3.0 * x1 - x2 + rng.normal(scale=0.2, size=n) + 20.0,
    })
    path = str(tmp_path / "raw.csv")
    df.to_csv(path, index=False)
    return RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3))


def _table_row(session, table_path):
    j = journal.load(session.run_dir)
    return next(t for t in j["tables"] if t["path"] == table_path)


def test_a_new_table_is_scored_without_being_asked(session):
    out = tools.transform_features(session, "def transform_features(df):\n"
                                            "    df = df.copy()\n"
                                            "    df['ratio'] = df['x1'] / (df['x2'].abs() + 1)\n"
                                            "    return df")
    assert out["probe_score"] is not None
    assert out["probe_target_transform"] in ("none", "log1p")
    # and it is on the journal row, not only in the tool's reply
    assert _table_row(session, out["table"])["probe_score"] == out["probe_score"]


def test_the_score_is_the_same_ruler_every_time(session):
    """Two identical tables must score identically, or the ruler is not a ruler."""
    code = ("def transform_features(df):\n"
            "    df = df.copy()\n"
            "    df['z'] = df['x1'] * 2\n"
            "    return df")
    a = tools.transform_features(session, code)
    b = tools.transform_features(session, code)
    assert a["probe_score"] == b["probe_score"]


def test_a_derived_table_is_compared_to_the_table_it_came_from(session):
    first = tools.transform_features(session, "def transform_features(df):\n"
                                              "    df = df.copy()\n"
                                              "    df['z'] = df['x1'] * 2\n"
                                              "    return df")
    second = tools.transform_features(
        session,
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['w'] = df['x2'] ** 2\n"
        "    return df",
        table=first["table"],
    )
    assert "probe_vs_source_table" in second
    # positive is always better, whatever the metric's direction (_gain)
    assert isinstance(second["probe_vs_source_table"], float)


def test_encoding_and_filtering_are_scored_too(session):
    enc = tools.fit_encoding(session, columns=["city"], method="target")
    assert enc["probe_score"] is not None

    filt = tools.filter_rows(
        session,
        "def keep_rows(df):\n    return df['x1'] < 2.5",
        table=enc["table"],
    )
    assert filt["probe_score"] is not None
    assert _table_row(session, filt["table"])["probe_score"] is not None


def test_a_table_that_cannot_be_scored_is_still_a_table(session, monkeypatch):
    """Scoring is a measurement of the artifact, not part of producing it. If the
    ruler explodes, the agent keeps the table and hears why — 0007's shape."""
    def boom(*a, **k):
        raise RuntimeError("no ruler today")

    monkeypatch.setattr(tools, "fit_probe", boom)
    out = tools.transform_features(session, "def transform_features(df):\n"
                                            "    df = df.copy()\n"
                                            "    df['z'] = df['x1'] * 2\n"
                                            "    return df")
    assert out["table"]
    assert out["probe_score"] is None
    assert "no ruler today" in out["probe_unavailable"]


def test_the_tables_frame_carries_the_scores(session):
    tools.transform_features(session, "def transform_features(df):\n"
                                      "    df = df.copy()\n"
                                      "    df['z'] = df['x1'] * 2\n"
                                      "    return df")
    frames = tools._journal_frames(session)
    assert "probe_score" in frames["tables"].columns
    assert frames["tables"]["probe_score"].notna().all()


def test_an_empty_run_still_has_the_probe_columns(session):
    """Declared, not inferred: `tables.sort_values("probe_score")` on a fresh run
    must return nothing rather than raise."""
    frames = tools._journal_frames(session)
    assert list(frames["tables"].columns) == tools._TABLE_COLUMNS
