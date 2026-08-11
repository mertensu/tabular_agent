"""The brief: what a fresh agent is told about a run it was not present for.

Two properties carry the whole design and each has a test that fails loudly if it
is lost: it must be **rendered** (never accumulated, so it costs no tool calls and
cannot drift from the journal) and it must be **flat** (the same size at
experiment 150 as at experiment 2, or it becomes the context growth it exists to
prevent).
"""

import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.brief import MAX_LISTED_TABLES, render_brief
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.stop import StopPolicy
from tabular_agent.harness.tools import (
    fit_encoding,
    finalize_run,
    inspect_data,
    train_tabular_model,
    transform_features,
)
from tabular_agent.taskspec import TaskSpec


@pytest.fixture
def session(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "area": rng.uniform(30, 160, n),
        "rooms": rng.integers(1, 6, n),
        "city": rng.choice(["a", "b", "c"], n),
    })
    df["price"] = df["area"] * 1000 + df["rooms"] * 5000 + rng.normal(0, 5000, n)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    return RunSession(str(tmp_path / "run"), str(path), TaskSpec(target="price"))


def test_a_run_that_has_done_nothing_gets_no_brief(session):
    """Deliberately `None` rather than "0 experiments so far".

    A run with no position has nothing to report, and the absence of the brief
    already says so. It also keeps a first episode byte-identical to a run with
    no episodes at all, which is what makes the two comparable.
    """
    assert render_brief(session) is None


def test_it_states_the_task_the_metric_and_its_direction(session):
    """The agent cannot rank anything without the direction, and a resumed one
    has no way to know it: `inspect_data` reported it to somebody else."""
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")

    brief = render_brief(session)
    assert "'price'" in brief
    assert "rmsle" in brief and "lower is better" in brief


def test_it_reports_the_position_not_the_history(session):
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    enc = fit_encoding(session, ["city"], "frequency")
    train_tabular_model(session, table=enc["table"], model_type="lgbm")

    brief = render_brief(session)
    assert "2 experiments" in brief and "1 feature tables" in brief
    assert "Baseline: e1" in brief
    assert "Best so far: e2" in brief
    # The winner's table is on the leaderboard row, not on the `best` pointer,
    # and it is the actionable half: the next idea gets built on top of it.
    assert enc["table"] in brief


def test_it_names_tables_nobody_ever_trained_on(session):
    """The trace an episode cut mid-idea leaves behind (0024's "seam, not a
    count"). The next agent did not build it and has no memory of meaning to."""
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    enc = fit_encoding(session, ["city"], "frequency")

    brief = render_brief(session)
    assert "never trained on" in brief and enc["table"] in brief


def test_the_list_of_idle_tables_is_capped(session):
    """The one list in the brief, and the one place run length could leak in."""
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    for i in range(MAX_LISTED_TABLES + 2):
        transform_features(session, f"def transform_features(df):\n"
                                    f"    df = df.copy()\n"
                                    f"    df['c{i}'] = df['area'] * {i + 1}\n"
                                    f"    return df")

    brief = render_brief(session)
    assert "and 2 more" in brief
    assert brief.count("tables/t") == MAX_LISTED_TABLES


def _fake_experiment(session, i: int, score: float):
    """One leaderboard row, appended without training a model.

    Synthetic on purpose: the claim under test is about 150 experiments, and
    fitting 150 models to assert a string length would make the test cost more
    than the property is worth. Nothing in the brief reads an artifact.
    """
    journal.append_experiment(session.run_dir, {
        "id": f"e{i}", "model_type": "lgbm", "params": {"n_estimators": 50},
        "target_transform": "none", "source_table": "data/work.parquet",
        "metric": "rmsle", "val_score": score, "cv_score": score,
        "fold_spread": 0.001, "features": ["area", "rooms"],
        "artifact_path": f"experiments/e{i}",
    })


def test_the_brief_does_not_grow_with_the_run(session):
    """The property the whole design rests on, asserted at the length that
    motivated it. If the brief grows, an episode's fixed cost becomes a function
    of run length and the sawtooth turns back into a climb.

    The scores improve steadily so the run never stalls — otherwise this measures
    the stop-line branch flipping rather than growth.
    """
    session.stop_policy = StopPolicy(max_experiments=200)
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    _fake_experiment(session, 2, 0.500)
    short = render_brief(session)

    score = 0.500
    for i in range(3, 152):
        score -= 0.001
        _fake_experiment(session, i, score)
    long = render_brief(session)

    assert len(journal.load(session.run_dir)["leaderboard"]) == 151
    assert "151 experiments" in long
    # Same lines in the same order; only counters and scores moved.
    assert len(long.splitlines()) == len(short.splitlines())
    assert abs(len(long) - len(short)) < 20


def test_it_carries_the_stop_state(session):
    """An agent that does not know it is nearly out of budget spends what is left
    orienting; one that thinks the run is over stops early. Neither is inferable
    from the leaderboard without also knowing the policy."""
    session.stop_policy = StopPolicy(max_experiments=4)
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    train_tabular_model(session, model_type="lgbm")

    assert "2 experiments left of 4" in render_brief(session)

    train_tabular_model(session, model_type="lgbm", params={"n_estimators": 30})
    train_tabular_model(session, model_type="lgbm", params={"n_estimators": 40})
    assert "stopping this run" in render_brief(session)
    assert "finalize_run" in render_brief(session)


def test_it_says_when_the_run_is_already_over(session):
    """The one state in which every tool call left is wasted."""
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    train_tabular_model(session, model_type="lgbm")
    finalize_run(session)

    assert "finalized" in render_brief(session)


def test_it_is_rendered_from_the_journal_and_holds_nothing_of_its_own(session):
    """No state, no cache, no side effect: two calls on an unchanged run dir must
    produce the same string, and calling it must not change the run."""
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")

    before = journal.load(session.run_dir)
    first = render_brief(session)
    second = render_brief(session)

    assert first == second
    assert journal.load(session.run_dir) == before


def test_a_fresh_session_object_on_the_same_run_dir_sees_the_same_brief(session):
    """The actual episode boundary: a new process, a new RunSession, one journal.
    If this diverges, the brief is remembering something instead of rendering it."""
    inspect_data(session)
    train_tabular_model(session, model_type="baseline")
    enc = fit_encoding(session, ["city"], "frequency")
    train_tabular_model(session, table=enc["table"], model_type="lgbm")

    resumed = RunSession(session.run_dir, session.source_path, TaskSpec(target="price"))
    assert render_brief(resumed) == render_brief(session)
