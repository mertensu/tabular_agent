"""The difference interval through the harness: does a run actually get one.

`test_compare.py` checks the instrument. What matters here is the plumbing it
depends on — that a table's probe predictions survive to disk, that the parent's
are still there when a child is scored, and that the mode follows the split
without anyone passing a flag.
"""

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    _load_probe_preds,
    _probe_preds_path,
    explore_journal,
    inspect_data,
    transform_features,
)
from tabular_agent.taskspec import TaskSpec

ADD_A_REAL_FEATURE = (
    "def transform_features(df):\n"
    "    df['signal'] = df['x'] * 2.0 + df['z']\n"
    "    return df\n"
)

ADD_NOISE = (
    "def transform_features(df):\n"
    "    df['noise'] = df['x'] * 0.0\n"
    "    return df\n"
)


def _frame(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    return pd.DataFrame({
        "x": x,
        "z": z,
        "day": np.arange(n),
        "y": 5.0 * x + 3.0 * z + rng.normal(scale=0.5, size=n) + 20.0,
    })


def _session(tmp_path, *, ordered=False, shuffle_file_order=False, seed=0):
    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    df = _frame(seed=seed)
    if shuffle_file_order:
        # The real hazard: an ordered dataset whose file is not in date order.
        df = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    spec = TaskSpec(target="y", task_type="regression", metric="rmse", seed=0)
    return RunSession(
        run_dir=str(tmp_path / "run"),
        data_path=str(path),
        spec=spec,
        split_on="day" if ordered else None,
    )


# ------------------------------------------------------------------- the store


def test_a_probed_table_leaves_its_predictions_on_disk(tmp_path):
    session = _session(tmp_path)
    inspect_data(session)
    out = transform_features(session, ADD_A_REAL_FEATURE)

    preds = _load_probe_preds(session, os.path.join(session.run_dir, out["table"]))
    assert preds is not None
    assert len(preds) == len(pd.read_parquet(session.val_path))


def test_the_work_table_is_stored_too_so_the_first_child_can_be_compared(tmp_path):
    """The work table has no journal row — nothing derived it — and is probed
    lazily the first time anything is built on it. If its predictions were not
    kept there, the very first comparison of a run would have no interval."""
    session = _session(tmp_path)
    inspect_data(session)
    transform_features(session, ADD_A_REAL_FEATURE)

    assert os.path.exists(_probe_preds_path(session, session.data_path))


def test_a_missing_sidecar_costs_the_interval_and_not_the_score(tmp_path):
    session = _session(tmp_path)
    inspect_data(session)
    transform_features(session, ADD_A_REAL_FEATURE)
    os.remove(_probe_preds_path(session, session.data_path))

    out = transform_features(session, ADD_NOISE)
    assert out["probe_score"] is not None
    assert "probe_vs_source_interval" not in out


# -------------------------------------------------------------------- the mode


def test_a_random_split_gets_the_paired_bootstrap(tmp_path):
    session = _session(tmp_path)
    inspect_data(session)
    out = transform_features(session, ADD_A_REAL_FEATURE)

    interval = out["probe_vs_source_interval"]
    assert interval["mode"] == "paired_bootstrap"
    assert interval["paired_se"] >= 0
    # 0036's claim, re-measured on every run rather than cited from one.
    assert interval["pairing_ratio"] < 1.0


def test_an_ordered_split_gets_blocks_instead(tmp_path):
    session = _session(tmp_path, ordered=True)
    inspect_data(session)
    out = transform_features(session, ADD_A_REAL_FEATURE)

    interval = out["probe_vs_source_interval"]
    assert interval["mode"] == "ordered_blocks"
    assert len(interval["block_gains"]) >= 2
    assert "sign_stable" in interval


def test_the_blocks_survive_a_file_that_is_not_in_date_order(tmp_path):
    """`RunSession` writes the validation half in the input file's row order, so
    on a dataset that did not arrive sorted the blocks have to be cut from the
    ordering column. Both files describe the same data; the block gains must
    agree."""
    ordered_file = _session(tmp_path / "a", ordered=True)
    inspect_data(ordered_file)
    a = transform_features(ordered_file, ADD_A_REAL_FEATURE)

    shuffled_file = _session(tmp_path / "b", ordered=True, shuffle_file_order=True)
    inspect_data(shuffled_file)
    b = transform_features(shuffled_file, ADD_A_REAL_FEATURE)

    assert (
        a["probe_vs_source_interval"]["block_gains"]
        == b["probe_vs_source_interval"]["block_gains"]
    )


# ------------------------------------------------------------------- reporting


def test_the_interval_reaches_the_journal_and_the_tables_frame(tmp_path):
    session = _session(tmp_path)
    inspect_data(session)
    transform_features(session, ADD_A_REAL_FEATURE)

    row = (journal.load(session.run_dir)["tables"])[0]
    assert row["probe_vs_source_interval"]["mode"] == "paired_bootstrap"

    out = explore_journal(
        session,
        "def explore(tables):\n"
        "    return tables[['probe_vs_source_table', 'probe_vs_source_se',\n"
        "                   'probe_vs_source_mode']].to_dict('records')\n",
    )
    got = out["result"][0]
    assert got["probe_vs_source_mode"] == "paired_bootstrap"
    assert got["probe_vs_source_se"] is not None


def test_an_empty_run_still_hands_over_a_well_formed_tables_frame(tmp_path):
    """The declared-columns rule (0020's `notes` trap): a frame missing its
    columns raises `KeyError` on the query rather than returning nothing."""
    session = _session(tmp_path)
    inspect_data(session)
    out = explore_journal(
        session,
        "def explore(tables):\n    return list(tables.columns)\n",
    )
    for col in ("probe_vs_source_se", "probe_vs_source_mode",
                "probe_vs_source_sign_stable"):
        assert col in out["result"]
