"""`select_features` — the one tool that makes a table narrower.

The properties pinned here are the ones a reader cannot get from the signature:
which rows the search is allowed to rank on, that a column the search never
considered survives, that a table is only written when a subset actually won, and
that the narrowed table still replays.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    SELECTION_BLOCK,
    inspect_data,
    select_features,
    transform_features,
)
from tabular_agent.io import load_table
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.taskspec import TaskSpec


def _frame(n=600, n_noise=12, seed=0):
    """Four columns that carry the target and a pile of pure noise.

    Kept for the tests that need *a* table, but note what it is not: a table this
    tool can profitably narrow. Measured on it, dropping nine pure-noise columns
    moves the score by 0.0014 — a boosted tree simply does not split on a column
    that carries nothing, so removing it wins almost nothing back. Use
    `_drifting_frame` for the cases that need the search to actually fire.
    """
    rng = np.random.default_rng(seed)
    cols = {f"s{i}": rng.normal(size=n) for i in range(4)}
    y = sum(cols.values()) + rng.normal(scale=0.2, size=n)
    for i in range(n_noise):
        cols[f"n{i}"] = rng.normal(size=n)
    return pd.DataFrame({**cols, "y": y})


def _drifting_frame(n=1200, n_drift=2, seed=0):
    """Stable signal, plus columns that predict the target *only in the past*.

    This is the regime feature selection is actually for, and getting that wrong
    is what the first two versions of this tool cost. The Rossmann winner:

        "I soon noted that the spread features made overfitting easy."
        "With all features together, the model easily overfit the train set,
         resulting in suboptimal performance on the test set."

    His problem was never useless columns — they cost a GBDT nothing. It was
    columns that fit the training period and do not transfer. Here `d*` correlate
    with the target over the first 70% of the ordering and are pure noise after,
    so a model that leans on them scores well where it was fitted and badly on the
    block that ranks it. A subset that leaves them out should win.

    **Two drifting columns, and the number is load-bearing.** A random subset
    excludes all of them with probability ``keep_prob ** n_drift``, so at the
    default 0.5 six drifting columns leave 1.6% of draws clean — under one of 40.
    Built that way first, the search had nothing good to find, dropped the stable
    signal along with the rest and scored **−0.489** against its parent. That is a
    real limit of the tool, not of the fixture: a random-subset search needs draws
    exponential in the number of columns that have to be excluded *together*, which
    is why the report this comes from used 500 models rather than 40.
    """
    rng = np.random.default_rng(seed)
    day = np.arange(n)
    cols = {f"s{i}": rng.normal(size=n) for i in range(3)}
    y = sum(cols.values()) + rng.normal(scale=0.2, size=n)
    early = day < int(n * 0.7)
    for i in range(n_drift):
        drift = rng.normal(size=n)
        # In the early period the column *is* the target plus a little noise; in
        # the late period the relationship is gone entirely.
        cols[f"d{i}"] = np.where(early, y + rng.normal(scale=0.3, size=n), drift)
    return pd.DataFrame({"day": day, **cols, "y": y})


@pytest.fixture
def session(tmp_path):
    path = str(tmp_path / "data.csv")
    _frame().to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3))
    inspect_data(s)
    return s


@pytest.fixture
def drifting(tmp_path):
    """An ordered run whose table carries columns that stop working."""
    path = str(tmp_path / "drift.parquet")
    _drifting_frame().to_parquet(path, index=False)
    s = RunSession(
        str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3),
        val_fraction=0.15, split_on="day",
    )
    inspect_data(s)
    return s


def test_it_drops_the_columns_that_stop_working(drifting):
    """The property the tool is actually for — see `_drifting_frame`.

    Two assertions, and the second is the one that matters. "It dropped the
    columns I planted" can be satisfied by a search that drops everything, so the
    test that carries weight is that the narrowed table **beats its parent on the
    validation half** — rows the search never saw. That is exactly the number
    which came back negative on Rossmann and near zero on Restaurant (0069), so it
    is the assertion that would notice this tool regressing.

    Deliberately *not* asserting that every stable column survives. Over the
    period the candidates are fitted on, the planted columns are nearly equal to
    the target, so a weak stable column genuinely earns little there and dropping
    one is not evidence of a defect. Pinning it made the test fail for a reason
    that had nothing to do with the property being tested.
    """
    out = select_features(drifting, n_candidates=40, seed=0)

    assert out["selected"] is True
    assert {f"d{i}" for i in range(2)} <= set(out["dropped_columns"])
    assert out["probe_vs_source_table"] > 0


def test_useless_columns_are_not_what_this_buys(session):
    """A table that is three-quarters pure noise is *not* narrowed, and should not be.

    Pinned because it is the finding that redirected this tool, and because it
    reads as a bug to anyone who has not seen the measurement: dropping nine
    noise columns here moves the score by about 0.001, so the search correctly
    declines. Feature selection pays on columns that *mislead*, not on columns
    that are merely idle.
    """
    out = select_features(session, n_candidates=40, seed=0)

    assert out["selected"] is False
    scores = {c["choice"]: c["selection_score"] for c in out["candidates"]}
    assert min(scores.values()) == pytest.approx(scores["all columns"], abs=5e-3)


def test_the_search_never_ranks_on_the_validation_half(session, monkeypatch):
    """The leakage property, pinned by counting rows rather than by reading code.

    0023 measured what an argmax over 24 candidates does to the rows it is taken
    on. This search takes one over hundreds, so if it ever reached `val` the run's
    only honest ruler would be gone — and the failure would be invisible, because
    a contaminated `val` still returns a number.
    """
    work = load_table(session.data_path)
    val = load_table(session.val_path)
    seen: list[int] = []

    import tabular_agent.harness.tools as tools_mod

    real = tools_mod.fit_probe

    def spy(X, y, *a, **kw):
        seen.append(len(X))
        return real(X, y, *a, **kw)

    monkeypatch.setattr(tools_mod, "fit_probe", spy)
    select_features(session, n_candidates=6, seed=0)

    expected = int(len(work) * (1 - SELECTION_BLOCK))
    # Every fit the search made used the same block, and that block is a prefix
    # of the work half — never the whole work half (which would mean it ranked on
    # rows it fitted on) and never anything the size of validation.
    assert seen, "the search fitted nothing"
    assert set(seen) == {expected}
    assert expected < len(work)
    assert len(val) not in set(seen)


def test_nothing_is_written_when_no_subset_wins(tmp_path):
    """A table where every column pays must come back untouched."""
    rng = np.random.default_rng(1)
    n = 600
    cols = {f"s{i}": rng.normal(size=n) for i in range(6)}
    df = pd.DataFrame({**cols, "y": sum(cols.values()) + rng.normal(scale=0.05, size=n)})
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3))
    inspect_data(s)

    out = select_features(s, n_candidates=20, seed=0)

    assert out["selected"] is False
    assert not journal.load(s.run_dir)["tables"], "a declined search wrote a table"
    assert not os.path.exists(os.path.join(s.run_dir, "tables", "t1.parquet"))


def test_non_numeric_columns_are_never_candidates_and_never_dropped(tmp_path):
    """They do not reach the model, so narrowing them changes nothing a fit saw.

    It would change what later tools can build on, which is why they survive
    rather than being swept out with the rest.
    """
    df = _drifting_frame()
    df["city"] = np.resize(["berlin", "hamburg"], len(df))
    path = str(tmp_path / "d.parquet")
    df.to_parquet(path, index=False)
    s = RunSession(
        str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3),
        val_fraction=0.15, split_on="day",
    )
    inspect_data(s)

    out = select_features(s, n_candidates=40, seed=0)

    assert out["selected"] is True
    assert "city" not in out["dropped_columns"]
    assert "city" in load_table(s.resolve_table(out["table"])).columns


def test_the_ordering_column_survives_the_search(tmp_path):
    """Dropping it would leave a table that cannot be split, blocked or compared."""
    df = _frame(n=800)
    df["day"] = np.arange(len(df))
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"), path,
        TaskSpec(target="y", n_splits=3),
        val_fraction=0.1, split_on="day",
    )
    inspect_data(s)

    out = select_features(s, n_candidates=40, seed=0)

    assert "day" not in out.get("dropped_columns", [])
    assert "day" in load_table(s.resolve_table(out["table"])).columns


def test_the_selection_block_is_the_tail_when_the_run_is_ordered(tmp_path):
    """Ranked on rows later than the ones fitted on — the shape of the task (0043)."""
    from tabular_agent.harness.tools import _selection_split

    df = pd.DataFrame({"day": [5, 1, 4, 2, 3], "y": [1.0, 2, 3, 4, 5]})
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y"), split_on="day")

    fit_pos, rank_pos = _selection_split(df, s)

    # Positions, not labels: the frame is deliberately out of order, and sorting
    # the *positions* instead of the values is the bug 0059 was written around.
    assert df["day"].to_numpy()[rank_pos].min() > df["day"].to_numpy()[fit_pos].max()


def test_the_selection_block_matches_the_validation_block_s_span(tmp_path):
    """The property the first version of this tool got wrong, at a cost of 0.020.

    A fixed *fraction* of the table is a span only by accident. On Rossmann a
    fifth of the work half is six months against a validation block of forty-five
    days, so the search ranked candidates over a horizon the run is never scored
    at, dropped `DayOfWeek` and `Promo`, and lost 0.020 of rmspe on validation.

    Here: 1,000 daily rows, `val_fraction=0.1`, so validation is the last ~100
    days and the selection block must be the ~100 days before it — not the 200
    days that `SELECTION_BLOCK` alone would take.
    """
    from tabular_agent.harness.tools import _selection_split

    n = 1000
    df = pd.DataFrame({
        "day": pd.date_range("2020-01-01", periods=n, freq="D"),
        "x": np.arange(n, dtype=float),
        "y": np.arange(n, dtype=float),
    })
    # Parquet, not CSV: a CSV round-trip turns the dates into strings, which sort
    # but do not subtract — a case `_validation_span` handles by falling back to
    # the fraction, and which would therefore make this test pass vacuously.
    path = str(tmp_path / "d.parquet")
    df.to_parquet(path, index=False)
    s = RunSession(
        str(tmp_path / "run"), path, TaskSpec(target="y"),
        val_fraction=0.1, split_on="day",
    )
    work = load_table(s.data_path)
    val = load_table(s.val_path)
    val_span = val["day"].max() - val["day"].min()

    fit_pos, rank_pos = _selection_split(work, s)
    block = work["day"].to_numpy()[rank_pos]
    block_span = block.max() - block.min()

    # Within a day of the validation block's width, and nowhere near the fifth of
    # the table the fraction alone would have taken.
    assert abs(pd.Timedelta(block_span) - val_span) <= pd.Timedelta(days=1)
    assert len(rank_pos) < len(work) * SELECTION_BLOCK


def test_an_ordering_column_that_does_not_subtract_falls_back(tmp_path):
    """Dates read from a CSV are strings: they sort, so the run is ordered, but
    `max - min` raises. The tool must fall back, not fail — this crashed
    `select_features` outright before `_validation_span` guarded the subtraction.
    """
    from tabular_agent.harness.tools import _selection_split, _validation_span

    n = 400
    df = pd.DataFrame({
        "day": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "x": np.arange(n, dtype=float),
        "y": np.arange(n, dtype=float),
    })
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(
        str(tmp_path / "run"), path, TaskSpec(target="y"),
        val_fraction=0.1, split_on="day",
    )
    work = load_table(s.data_path)

    assert _validation_span(s) is None
    fit_pos, rank_pos = _selection_split(work, s)
    assert len(rank_pos) == len(work) - int(len(work) * (1 - SELECTION_BLOCK))
    # And the ordering still holds: it fell back to a fraction, not to chaos.
    assert work["day"].to_numpy()[rank_pos].min() > work["day"].to_numpy()[fit_pos].max()


def test_an_unordered_run_falls_back_to_the_fraction(tmp_path):
    """There is no span to match when the rows have no order, and none is invented."""
    from tabular_agent.harness.tools import _selection_split

    df = _frame(n=500)
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y"))
    work = load_table(s.data_path)

    fit_pos, rank_pos = _selection_split(work, s)

    assert len(rank_pos) == len(work) - int(len(work) * (1 - SELECTION_BLOCK))


def test_the_narrowed_table_still_replays_into_a_bundle(drifting):
    """A dropped column must be gone from serving too, or the bundle lies."""
    out = select_features(drifting, n_candidates=40, seed=0)
    assert out["selected"] is True

    from tabular_agent.harness.tools import finalize_run, train_tabular_model

    train_tabular_model(drifting, table=out["table"], model_type="lgbm")
    finalize_run(drifting)
    bundle = load_bundle(os.path.join(drifting.run_dir, "bundle"))

    raw = load_table(drifting.data_path).drop(columns=["y"]).head(5)
    preds = predict_bundle(bundle, raw)

    assert len(preds) == 5
    assert np.isfinite(preds).all()


def test_a_dropped_column_is_recorded_in_the_lineage_as_code(drifting):
    """The narrowing has to be replayable from the sidecar, like any other step."""
    out = select_features(drifting, n_candidates=40, seed=0)
    assert out["selected"] is True
    path = drifting.resolve_table(out["table"])
    with open(path.replace(".parquet", ".pipeline.json")) as f:
        steps = json.load(f)["steps"]

    code = [s["code"] for s in steps if s.get("type") == "transform"]
    assert code, "no transform step was stored"
    assert "drop" in code[-1]
    for col in out["dropped_columns"]:
        assert repr(col) in code[-1] or col in code[-1]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"keep_prob": 0.0}, "keep_prob"),
        ({"keep_prob": 1.0}, "keep_prob"),
        ({"top_k": 0}, "top_k"),
        ({"n_candidates": 1}, "n_candidates"),
        ({"n_candidates": 10_000}, "n_candidates"),
    ],
)
def test_it_refuses_nonsense_settings(session, kwargs, match):
    with pytest.raises(ValueError, match=match):
        select_features(session, **kwargs)


def test_it_refuses_a_table_too_narrow_to_search(tmp_path):
    df = pd.DataFrame({"a": [1.0] * 60, "b": [2.0] * 60, "y": np.arange(60.0)})
    path = str(tmp_path / "d.csv")
    df.to_csv(path, index=False)
    s = RunSession(str(tmp_path / "run"), path, TaskSpec(target="y", n_splits=3))
    inspect_data(s)

    with pytest.raises(ValueError, match="too few to search"):
        select_features(s, n_candidates=10)


def test_it_is_reproducible(session):
    """Same seed, same draws, same verdict — including the verdict "no subset won".

    Asserted on `candidates`, which both branches carry, rather than on
    `dropped_columns`, which only exists when a subset actually won. The first
    version of this test asserted the latter and failed for a reason that had
    nothing to do with reproducibility: at a low `n_candidates` the search
    legitimately declines.
    """
    a = select_features(session, n_candidates=20, seed=3)
    b = select_features(session, n_candidates=20, seed=3)

    assert a["selected"] == b["selected"]
    assert a["candidates"] == b["candidates"]
    if a["selected"]:
        assert a["dropped_columns"] == b["dropped_columns"]


def test_it_can_prune_a_table_built_by_another_tool(drifting):
    """The realistic path: widen with `transform_features`, then narrow."""
    wider = transform_features(
        drifting,
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['junk'] = df['s0'] * 0 + 1\n"
        "    return df\n",
    )
    out = select_features(drifting, table=wider["table"], n_candidates=40, seed=0)

    assert out["source_table"] == wider["table"]
    tables = journal.load(drifting.run_dir)["tables"]
    assert [t["id"] for t in tables] == ["t1", "t2"]
