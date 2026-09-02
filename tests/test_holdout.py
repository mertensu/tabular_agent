"""The sealed third split (0023).

Every test here is about one of three claims: the sealed rows are *unreachable*
while the run is going, they are read *once* and only after the winner is picked,
and turning them on does not silently re-cut the rows every earlier run was
measured against.
"""

import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    fit_encoding,
    finalize_run,
    inspect_data,
    transform_features,
)
from tabular_agent.io import load_table
from tabular_agent.taskspec import TaskSpec


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "day": np.arange(n),
            "x1": x1,
            "x2": x2,
            "city": rng.choice(["berlin", "hamburg", "koeln"], size=n),
            "y": (x1 + 0.5 * x2 + rng.normal(scale=0.3, size=n) > 0).astype(int),
        }
    )


def _source(tmp_path, df=None):
    path = str(tmp_path / "data.csv")
    (df if df is not None else _frame()).to_csv(path, index=False)
    return path


def _session(tmp_path, name="run", **kwargs):
    return RunSession(
        str(tmp_path / name), _source(tmp_path), TaskSpec(target="y", n_splits=3), **kwargs
    )


@pytest.fixture
def sealed(tmp_path):
    return _session(tmp_path, holdout_fraction=0.2)


# ------------------------------------------------------------------- the split


def test_off_by_default(tmp_path):
    """A library default that re-cuts everyone's data is not a default anyone asked
    for. Turning the third split on takes rows out of the work half, which makes
    the run incomparable to every run before it — a decision for the driver."""
    s = _session(tmp_path)

    assert s.holdout_path is None
    assert not os.path.exists(os.path.join(s.run_dir, "data", "holdout.parquet"))
    assert s.split_record()["n_holdout"] == 0


def test_the_holdout_comes_out_of_work_and_leaves_validation_alone(tmp_path):
    """The comparability claim, and the reason `holdout_fraction` is a fraction of
    the whole input rather than a slice carved out of the validation half.

    Under a random split the validation rows must be *the same rows* with the
    holdout on as off, or every `val_se` and every val-denominated comparison
    quietly changes meaning the day this lands."""
    plain = _session(tmp_path, name="plain")
    with_holdout = _session(tmp_path, name="sealed", holdout_fraction=0.2)

    pd.testing.assert_frame_equal(
        load_table(plain.val_path), load_table(with_holdout.val_path)
    )
    assert len(load_table(plain.data_path)) == 240
    assert len(load_table(with_holdout.data_path)) == 180
    assert len(load_table(with_holdout.holdout_path)) == 60


def test_the_three_parts_are_disjoint_and_cover_the_input(sealed):
    """A row in two parts is a leak; a row in none is data thrown away silently."""
    parts = [
        load_table(p)["day"] for p in (sealed.data_path, sealed.val_path, sealed.holdout_path)
    ]
    days = pd.concat(parts)

    assert len(days) == 300
    assert sorted(days) == list(range(300))


def test_an_ordered_split_seals_the_latest_period(tmp_path):
    """work | val | holdout, in time order. The sealed slice stands in for unseen
    data, and unseen data is always the future — so it gets the last rows, even
    though that means the validation rows move when the holdout is turned on."""
    df = _frame().sample(frac=1, random_state=7)  # arrives out of order
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3),
        split_on="day", holdout_fraction=0.2,
    )

    work = load_table(s.data_path)["day"]
    val = load_table(s.val_path)["day"]
    held = load_table(s.holdout_path)["day"]

    assert work.max() < val.min() < val.max() < held.min()


def test_an_ordered_split_without_a_holdout_keeps_every_row(tmp_path):
    """Regression. The second boundary is computed only when it is wanted: the
    honest-boundary search never returns `n`, so asking it for the end of the
    validation slice would push the last rows into a holdout that is never
    written — losing them out of both halves, silently."""
    df = _frame().sample(frac=1, random_state=7)
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3), split_on="day"
    )

    total = len(load_table(s.data_path)) + len(load_table(s.val_path))
    assert total == 300
    assert s.holdout_path is None


def test_too_few_rows_names_all_three_parts(tmp_path):
    """Three splits are not viable on a tiny table, and the error has to say which
    part came up short — otherwise the fix ('drop holdout_fraction') is guesswork."""
    path = str(tmp_path / "d.csv")
    _frame(n=20).to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"), path, TaskSpec(target="y"), val_fraction=0.4,
        holdout_fraction=0.5,
    )

    with pytest.raises(ValueError, match=r"work=2, val=8, holdout=10"):
        s.data_path


def test_the_split_record_carries_the_sealed_rows(sealed):
    """The run dir may be deleted; the journal is where 'was this run honest' has
    to remain answerable."""
    inspect_data(sealed)
    split = journal.load(sealed.run_dir)["split"]

    assert (split["n_work"], split["n_val"], split["n_holdout"]) == (180, 60, 60)
    assert split["holdout_fraction"] == 0.2
    assert split["holdout_is_thin"] is False


# ------------------------------------------------------------------- the seal


def test_no_tool_can_reach_the_sealed_rows(sealed):
    """The guarantee is 'the agent cannot', not 'the agent has no reason to' —
    the same allowlist argument the validation half already leans on. Naming the
    file directly is the attack, and an agent stumbles into it after seeing
    `data/work.parquet` in a result."""
    inspect_data(sealed)
    assert os.path.exists(sealed.holdout_path)

    for attempt in ("data/holdout.parquet", sealed.holdout_path, "../run/data/holdout.parquet"):
        with pytest.raises(ValueError, match="no such table"):
            sealed.resolve_table(attempt)

    assert "data/holdout.parquet" not in sealed.available_tables()


def test_nothing_is_scored_on_the_seal_before_finalize(sealed):
    """Rule 1 of 0023. A number consulted repeatedly becomes a number to optimise
    against, which is exactly how the *second* split stopped being independent."""
    inspect_data(sealed)
    fit_encoding(sealed, ["city"], "frequency")
    transform_features(
        sealed,
        "def transform_features(df):\n    df = df.copy()\n    df['z'] = df['x1'] * 2\n    return df",
    )

    j = journal.load(sealed.run_dir)
    assert j.get("holdout") is None
    # And nothing that ranks anything carries a sealed number.
    assert all("holdout_score" not in t for t in j["tables"])
    assert all("holdout_score" not in e for e in j["leaderboard"])


def test_set_holdout_is_write_once(tmp_path):
    """'Scored exactly once' has to be auditable from the journal alone, not a
    property of whoever happens to be calling."""
    run_dir = str(tmp_path / "run")
    os.makedirs(run_dir)
    journal.set_holdout(run_dir, {"n_rows": 10, "scores": [{"experiment_id": "e1"}]})
    journal.set_holdout(run_dir, {"n_rows": 999, "scores": []})

    assert journal.load(run_dir)["holdout"]["n_rows"] == 10


# ---------------------------------------------------------------- the verdict


def test_finalize_scores_every_terminal_candidate_once(sealed):
    """The winner *and* the models it was picked over, on the same sealed rows.

    The fits already exist, so the extra numbers are free — and a winner beside
    its runners-up is the selected-vs-unselected pair 0039 and 0040 could only
    get from a Kaggle submission."""
    inspect_data(sealed)
    fit_encoding(sealed, ["city"], "frequency")
    out = finalize_run(sealed)

    record = journal.load(sealed.run_dir)["holdout"]
    assert record["n_rows"] == 60
    assert len(record["scores"]) >= 2
    # Exactly one of them is the candidate the *search* crowned. Since 0044 that
    # is no longer the same row as the one that ships, and both are kept: this is
    # the one 0039's optimism-of-the-selected series is measured on.
    selected = [e for e in record["scores"] if e["selected"]]
    assert len(selected) == 1
    assert selected[0]["experiment_id"] == out["best_experiment_id"]

    # And the gap on the finalize result belongs to whatever the bundle predicts
    # with, which is where the reader looks first.
    shipped = [e for e in record["scores"] if e.get("shipped")]
    assert len(shipped) == 1
    assert out["holdout_score"] == shipped[0]["holdout_score"]
    assert out["optimism"] == shipped[0]["optimism"]


def test_optimism_is_positive_when_validation_flattered(tmp_path):
    """One direction, always: positive means the validation half overstated. Every
    measurement in this project so far points that way, and a field that flips
    sign with the metric is read wrong exactly once."""
    from tabular_agent.harness.tools import _gain

    # logloss: lower is better, so a *higher* sealed score means val flattered.
    assert _gain(0.30, 0.40, "logloss") > 0
    # auc: higher is better, so a *lower* sealed score means val flattered.
    assert _gain(0.80, 0.70, "auc") > 0


def test_the_sealed_score_never_picks_the_winner(sealed):
    """Rule 2's real content. `best` is derived from the leaderboard, and the
    leaderboard is ranked on the validation half — so a sealed score that
    disagrees changes the report and nothing else."""
    inspect_data(sealed)
    fit_encoding(sealed, ["city"], "frequency")
    out = finalize_run(sealed)

    j = journal.load(sealed.run_dir)
    ranked = [
        r for r in j["leaderboard"]
        if r.get("val_score") is not None and r["model_type"] != "baseline"
    ]
    best_by_val = min(ranked, key=lambda r: r["val_score"])  # logloss
    assert out["best_experiment_id"] == best_by_val["id"]


def test_the_report_prints_both_numbers_and_the_gap(sealed):
    """Reporting only the sealed score would throw away the diagnostic. The gap
    *is* the measurement no run has ever had."""
    inspect_data(sealed)
    fit_encoding(sealed, ["city"], "frequency")
    out = finalize_run(sealed)

    text = open(os.path.join(sealed.run_dir, out["report_path"])).read()
    assert "Shipped validation score" in text
    assert "Shipped holdout score (sealed, scored once)" in text
    assert "## What the search cost" in text
    assert "These rows ranked nothing" in text
    assert "sealed holdout" in text


def test_a_run_without_a_seal_finalizes_unchanged(tmp_path):
    """The whole feature is opt-in, so the report of a run that did not opt in
    must be exactly what it was before this landed."""
    s = _session(tmp_path)
    inspect_data(s)
    fit_encoding(s, ["city"], "frequency")
    out = finalize_run(s)

    assert "holdout_score" not in out
    assert journal.load(s.run_dir).get("holdout") is None
    text = open(os.path.join(s.run_dir, out["report_path"])).read()
    assert "What the search cost" not in text


# -------------------------------------------------------------------- resume


def test_a_resumed_run_keeps_the_split_it_was_cut_with(tmp_path):
    """What is on disk wins over the constructor. Re-cutting to honour a
    `holdout_fraction` the original run did not have would move the validation
    rows under a leaderboard already written against them."""
    first = _session(tmp_path, holdout_fraction=0.2)
    original_val = load_table(first.val_path)
    original_holdout = load_table(first.holdout_path)

    # Resumed *without* the argument: the sealed rows are still there and are
    # still the same rows, because the run directory is the record of the split.
    resumed = _session(tmp_path)
    pd.testing.assert_frame_equal(original_val, load_table(resumed.val_path))
    pd.testing.assert_frame_equal(original_holdout, load_table(resumed.holdout_path))


def test_asking_for_a_seal_on_a_run_that_has_none_does_not_re_split(tmp_path):
    plain = _session(tmp_path)
    original_val = load_table(plain.val_path)

    resumed = _session(tmp_path, holdout_fraction=0.2)
    assert resumed.holdout_path is None
    pd.testing.assert_frame_equal(original_val, load_table(resumed.val_path))
