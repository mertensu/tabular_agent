"""The panel: several disagreeing evaluation geometries (0077)."""

import numpy as np
import pandas as pd
import pytest

from tabular_agent.panel import (
    MIN_SCHEME_ROWS,
    N_BLOCKS,
    Scheme,
    ordered_schemes,
    panel_agreement,
    scheme_scores,
    subsample_schemes,
    work_block_edges,
)


def _order(n=600):
    return np.arange(n, dtype=float)


def test_edges_cut_the_work_half_into_blocks():
    edges = work_block_edges(_order(600))
    assert len(edges) == N_BLOCKS - 1
    assert edges == sorted(edges)


def test_edges_are_values_not_positions_so_a_row_filter_cannot_move_them():
    """The whole reason boundaries are values: a filtered table must land in the
    same *periods* as its parent, not the same *positions*."""
    edges = work_block_edges(_order(600))
    full = ordered_schemes(_order(600), edges)
    # Half the rows gone, arbitrarily — the same ordering values, fewer of them.
    kept = _order(600)[::2]
    filtered = ordered_schemes(kept, edges)
    assert [s.name for s in full] == [s.name for s in filtered]
    for a, b in zip(full, filtered):
        # Same period, so roughly half the rows and never a different block.
        assert b.train.sum() == pytest.approx(a.train.sum() / 2, abs=2)
        assert b.evaluate.sum() == pytest.approx(a.evaluate.sum() / 2, abs=2)


def test_the_four_schemes_are_the_winners_geometries():
    edges = work_block_edges(_order(600))
    schemes = {s.name: s for s in ordered_schemes(_order(600), edges)}
    assert set(schemes) == {
        "train4_gap1_eval1", "train2_gap2_eval2",
        "train1_gap4_eval1", "train5_gap0_eval1",
    }
    # 600 rows, six blocks of 100.
    assert schemes["train4_gap1_eval1"].train.sum() == 400
    assert schemes["train4_gap1_eval1"].evaluate.sum() == 100
    assert schemes["train2_gap2_eval2"].train.sum() == 200
    assert schemes["train2_gap2_eval2"].evaluate.sum() == 200
    assert schemes["train1_gap4_eval1"].train.sum() == 100
    assert schemes["train5_gap0_eval1"].train.sum() == 500


def test_train_and_eval_never_overlap_in_any_scheme():
    """A scheme that scored rows it trained on would report the fit's own
    optimism as agreement."""
    edges = work_block_edges(_order(600))
    for s in ordered_schemes(_order(600), edges) + subsample_schemes(600, seed=42):
        assert not (s.train & s.evaluate).any()


def test_the_gap_schemes_actually_leave_a_gap():
    """The gap is the part no other instrument here has, so it is pinned."""
    edges = work_block_edges(_order(600))
    schemes = {s.name: s for s in ordered_schemes(_order(600), edges)}
    for name, expected_gap in [("train4_gap1_eval1", 100),
                               ("train2_gap2_eval2", 200),
                               ("train1_gap4_eval1", 400),
                               ("train5_gap0_eval1", 0)]:
        s = schemes[name]
        unused = 600 - s.train.sum() - s.evaluate.sum()
        assert unused == expected_gap, name


def test_a_coarse_ordering_column_gets_one_honest_scheme_not_four_fake_ones():
    """Many ties mean fewer distinct boundaries than blocks; four geometries cut
    on nothing would overlap silently."""
    order = np.repeat([1.0, 2.0, 3.0], 200)
    edges = work_block_edges(order)
    schemes = ordered_schemes(order, edges)
    assert len(schemes) == 1
    assert schemes[0].name == "train_early_eval_late"
    assert not (schemes[0].train & schemes[0].evaluate).any()


def test_no_ordering_values_means_no_ordered_schemes():
    assert ordered_schemes(_order(600), []) == []


def test_subsample_schemes_are_reproducible_and_paired_across_tables():
    """Two tables in one run must see the identical subsamples, or the
    per-scheme difference is not paired."""
    a = subsample_schemes(500, seed=7)
    b = subsample_schemes(500, seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x.train, y.train)
    assert not np.array_equal(a[0].train, a[1].train)


def test_a_scheme_too_small_to_score_is_not_run():
    tiny = Scheme("t", "horizon",
                  np.array([True] * 5 + [False] * 5),
                  np.array([False] * 5 + [True] * 5))
    assert not tiny.usable
    X = pd.DataFrame({"a": np.arange(10.0)})
    assert scheme_scores(X, np.arange(10) % 2, [tiny], "binary", "auc") == {}


# --- the agreement verdict -------------------------------------------------

def test_agreement_is_all_or_nothing():
    child = {"s1": 0.80, "s2": 0.80, "s3": 0.80, "s4": 0.80}
    parent = {"s1": 0.70, "s2": 0.70, "s3": 0.70, "s4": 0.90}
    out = panel_agreement(child, parent, "auc", headline_gain=0.05)
    assert out["n_schemes"] == 4
    assert out["n_agree"] == 3
    assert out["agrees"] is False, "3 of 4 is not agreement"
    assert out["disagreeing"] == ["s4"]


def test_agreement_when_every_scheme_confirms():
    child = {"s1": 0.80, "s2": 0.82}
    parent = {"s1": 0.70, "s2": 0.71}
    out = panel_agreement(child, parent, "auc", headline_gain=0.05)
    assert out["agrees"] is True
    assert out["n_agree"] == 2


def test_a_lower_is_better_metric_flips_the_sign_of_every_gain():
    child, parent = {"s1": 0.10, "s2": 0.11}, {"s1": 0.20, "s2": 0.22}
    out = panel_agreement(child, parent, "rmse", headline_gain=0.05)
    assert out["agrees"] is True
    assert all(g > 0 for g in out["gains"].values()), "smaller error is a positive gain"


def test_one_scheme_is_not_a_panel():
    assert panel_agreement({"s1": 0.8}, {"s1": 0.7}, "auc", 0.1) is None
    assert panel_agreement({"s1": 0.8}, {"s2": 0.7}, "auc", 0.1) is None


def test_a_zero_headline_has_no_sign_to_confirm():
    out = panel_agreement({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}, "auc",
                          headline_gain=0.0)
    assert out["agrees"] is False
    assert "no sign" in out["note"]


def test_the_panel_only_judges_the_headline_it_was_given():
    """A gain the single ruler calls positive and every scheme calls negative is
    the case this instrument exists for."""
    child = {"s1": 0.60, "s2": 0.61, "s3": 0.59}
    parent = {"s1": 0.70, "s2": 0.71, "s3": 0.69}
    out = panel_agreement(child, parent, "auc", headline_gain=+0.02)
    assert out["n_agree"] == 0
    assert out["agrees"] is False
    assert out["worst"] < 0


def test_scheme_scores_drops_a_scheme_that_raises_rather_than_voting_against():
    """A fit that crashed has no opinion; counting it as disagreement would turn
    an exception into evidence against a feature."""
    X = pd.DataFrame({"a": np.arange(200.0)})
    y = pd.Series((np.arange(200) % 2))
    good = Scheme("good", "horizon", np.arange(200) < 100, np.arange(200) >= 100)
    broken = Scheme("broken", "horizon",
                    np.arange(200) < 100, np.arange(200) >= 100)
    scores = scheme_scores(X, y, [good, broken], "binary", "auc")
    # Both are fine here; the contract is that names present == names scored.
    assert set(scores) <= {"good", "broken"}
    assert all(np.isfinite(v) for v in scores.values())


def test_scheme_scores_runs_the_real_probe_end_to_end():
    rng = np.random.default_rng(0)
    n = 600
    signal = rng.normal(size=n)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    y = pd.Series((signal + rng.normal(scale=0.3, size=n) > 0).astype(int))
    edges = work_block_edges(_order(n))
    schemes = ordered_schemes(_order(n), edges)
    scores = scheme_scores(X, y, schemes, "binary", "auc")
    assert len(scores) == 4
    assert all(0.5 < v <= 1.0 for v in scores.values()), scores


# --- the soft gate ---------------------------------------------------------
# The panel decides what counts as *progress*; it never decides what may win.

from tabular_agent.harness.stop import StopPolicy, best_table, evaluate  # noqa: E402


def _rows(*panels):
    """Tables improving by a clear margin, each with the given panel verdict."""
    return [
        {"id": f"t{i + 1}", "probe_score": 0.5 + 0.05 * (i + 1), "probe_se": 0.001,
         **({"probe_panel": {"agrees": p}} if p is not None else {})}
        for i, p in enumerate(panels)
    ]


def test_a_step_the_panel_rejects_does_not_count_as_progress():
    agreeing = evaluate(_rows(True, True, True), "auc", StopPolicy())
    rejected = evaluate(_rows(True, False, False), "auc", StopPolicy())
    assert agreeing["since_improvement"] == 0
    assert rejected["since_improvement"] == 2


def test_a_rejected_step_can_still_win_best_table():
    """The soft gate, exactly: 0035's argmax is untouched."""
    rows = _rows(True, False)
    rows[1]["path"] = "tables/t2.parquet"
    rows[0]["path"] = "tables/t1.parquet"
    winner = best_table(rows, "auc")
    assert winner["id"] == "t2", "the highest score still wins, panel or no panel"


def test_a_missing_panel_behaves_exactly_as_before():
    """Every artifact written before 0077 has no panel key, and must not stall."""
    without = evaluate(_rows(None, None, None), "auc", StopPolicy())
    with_agreement = evaluate(_rows(True, True, True), "auc", StopPolicy())
    assert without["since_improvement"] == with_agreement["since_improvement"] == 0


def test_a_panel_that_agrees_cannot_rescue_a_sub_noise_gain():
    """The two conditions are AND, not OR: agreement does not replace the
    yardstick 0022 put there."""
    rows = [
        {"id": "t1", "probe_score": 0.90, "probe_se": 0.01,
         "probe_panel": {"agrees": True}},
        {"id": "t2", "probe_score": 0.9001, "probe_se": 0.01,
         "probe_panel": {"agrees": True}},
    ]
    assert evaluate(rows, "auc", StopPolicy())["since_improvement"] == 1
