"""Every table is scored the moment it is created (0035/0036).

The point of automatic scoring is coverage: across the runs in `_runs/`, 22 of
the 24 tables that recombined an abandoned branch had no number at all, because
the agent trains only the endpoint of a chain. These tests pin the two things
that make the coverage trustworthy — that the score arrives without being asked
for, and that failing to score never costs the agent the table.
"""

import json

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


def test_the_ruler_is_on_the_terminal_ballot(session):
    """The invariant: a run never ships a model worse than the number it selected by.

    It holds by construction rather than by an assertion — the probe's own model
    type and parameters are a terminal candidate, so the score the search was
    ranked on is already demonstrated achievable. Porto Seguro v3 is what the
    other arrangement costs: a table measured at gini 0.29066 by the probe,
    packaged at 0.25389 because the ruler was not on the ballot.
    """
    from tabular_agent.probe import PROBE_MODEL_TYPE

    assert PROBE_MODEL_TYPE in tools.TERMINAL_MODEL_TYPES


def test_the_shipped_model_is_not_worse_than_the_winning_table_probe(session):
    tools.inspect_data(session)
    t = tools.transform_features(session, "def transform_features(df):\n"
                                          "    df = df.copy()\n"
                                          "    df['z'] = df['x1'] * 2\n"
                                          "    return df")
    out = tools.finalize_run(session)

    winner = _table_row(session, t["table"])
    # rmsle here, so lower is better; the shipped score must not be worse than
    # the ruler's on the table it was chosen from.
    assert out["val_score"] <= winner["probe_score"] + 1e-9


def test_an_unservable_table_is_scored_as_unscorable(session):
    """0030 at its new home.

    A table whose lineage needs a column that will not exist at predict time
    cannot be packaged into a bundle. `train_tabular_model` used to refuse it and
    tell the agent; since 0035 the ruler is the first thing to touch a table, so
    it is the only place left to say so. Left *unscored* rather than scored: an
    unservable table must not be able to win `best_table`.
    """
    s = RunSession(
        session.run_dir + "_srv", session.source_path,
        TaskSpec(target="y", n_splits=3),
        serving_columns=["x1", "city"],   # x2 will not exist at predict time
    )
    tools.inspect_data(s)
    out = tools.transform_features(s, "def transform_features(df):\n"
                                      "    df = df.copy()\n"
                                      "    df['from_x2'] = df['x2'] * 2\n"
                                      "    return df")
    assert out["probe_score"] is None
    assert out["unservable"] is True
    assert "x2" in json.dumps(out["columns_not_available_at_predict_time"])

    # and it cannot win the terminal fit
    from tabular_agent.harness.stop import best_table
    j = journal.load(s.run_dir)
    assert best_table(j["tables"], "rmsle") is None


def test_the_terminal_phase_reproduces_the_probe_exactly(session):
    """The invariant, and the reason it is about the *fit* and not the model type.

    Porto Seguro v4 put `PROBE_MODEL_TYPE` on the terminal ballot but fitted it
    with the run's seed rather than the probe's, and the result came in 0.00279 of
    gini under the probe on the table the probe had chosen — lgbm moves 0.003 of
    auc between seeds at 476k rows. So this asserts equality, not membership.
    """
    from tabular_agent.probe import PROBE_MODEL_TYPE

    tools.inspect_data(session)
    tools.transform_features(session, "def transform_features(df):\n"
                                      "    df = df.copy()\n"
                                      "    df['z'] = df['x1'] * 2\n"
                                      "    return df")
    tools.finalize_run(session)

    j = journal.load(session.run_dir)
    row = next(r for r in j["leaderboard"] if r["model_type"] == PROBE_MODEL_TYPE)
    if row["source_table"] == session.rel(session.data_path):
        winner_probe = j["work_probe"]
    else:
        winner_probe = _table_row(session, row["source_table"])
    # `probe_score` is rounded to 6 dp on the journal row; the fit itself is
    # bit-identical, so this is the storage precision and not a tolerance.
    assert row["val_score"] == pytest.approx(winner_probe["probe_score"], abs=5e-7)
