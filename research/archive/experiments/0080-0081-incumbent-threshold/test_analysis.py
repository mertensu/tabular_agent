"""Unit tests for the offline incumbent block-threshold reconstruction."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("analyze.py")


def _module():
    spec = importlib.util.spec_from_file_location("probe_incumbent_block_threshold", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_high_capacity_edges_recover_branch_utilities_and_historical_incumbents():
    mod = _module()
    edges = [
        {"parent": "t1", "candidate": "t2", "gain_high": 0.10},
        {"parent": "t2", "candidate": "t3", "gain_high": -0.04},
        {"parent": "t1", "candidate": "t4", "gain_high": 0.08},
        {"parent": "t4", "candidate": "t5", "gain_high": 0.05},
    ]

    utility = mod.accumulate_high_utilities(edges, anchor="t1")
    assert utility == pytest.approx({"t1": 0.0, "t2": 0.10, "t3": 0.06, "t4": 0.08, "t5": 0.13})

    contrasts = mod.high_incumbent_contrasts(
        table_order=["t1", "t2", "t3", "t4", "t5"], utility=utility
    )
    assert contrasts["t2"] == {"incumbent": "t1", "gain": pytest.approx(0.10)}
    assert contrasts["t3"] == {"incumbent": "t2", "gain": pytest.approx(-0.04)}
    assert contrasts["t4"] == {"incumbent": "t2", "gain": pytest.approx(-0.02)}
    assert contrasts["t5"] == {"incumbent": "t2", "gain": pytest.approx(0.03)}


def test_low_incumbents_follow_metric_direction_and_keep_work_separate():
    mod = _module()
    tables = [
        {"id": "t1", "path": "tables/t1.parquet", "source_table": "data/work.parquet", "probe_score": 0.40},
        {"id": "t2", "path": "tables/t2.parquet", "source_table": "tables/t1.parquet", "probe_score": 0.30},
        {"id": "t3", "path": "tables/t3.parquet", "source_table": "tables/t1.parquet", "probe_score": 0.35},
    ]

    derived = mod.historical_low_incumbents(tables, metric="logloss")
    assert derived["t2"]["incumbent"] == "t1"
    assert derived["t3"]["incumbent"] == "t2"
    assert derived["t3"]["gain"] == pytest.approx(-0.05)

    with_work = mod.historical_low_incumbents(
        tables, metric="logloss", work_probe={"probe_score": 0.25}
    )
    assert with_work["t1"]["incumbent"] == "work"
    assert with_work["t3"]["incumbent"] == "work"
    assert with_work["t3"]["gain"] == pytest.approx(-0.10)

    auc_tables = [{**row, "probe_score": 1.0 - row["probe_score"]} for row in tables]
    auc = mod.historical_low_incumbents(auc_tables, metric="auc")
    assert auc["t3"]["incumbent"] == "t2"
    assert auc["t3"]["gain"] == pytest.approx(-0.05)


def test_rule_metrics_use_all_true_improvements_as_recall_denominator():
    mod = _module()
    rows = [
        {"gain": 0.10, "block_sd": 0.20, "truth": True},   # accepted at k=.25
        {"gain": 0.02, "block_sd": 0.20, "truth": True},   # rejected at k=.25
        {"gain": 0.08, "block_sd": 0.20, "truth": False},  # accepted at k=.25
        {"gain": -0.01, "block_sd": 0.20, "truth": False},
    ]

    current = mod.rule_metrics(rows, k=0.0)
    assert current == {"accepted": 3, "true_positive": 2, "truth_positive": 2,
                       "precision": pytest.approx(2 / 3), "recall": pytest.approx(1.0)}

    quarter = mod.rule_metrics(rows, k=0.25)
    assert quarter == {"accepted": 2, "true_positive": 1, "truth_positive": 2,
                       "precision": pytest.approx(0.5), "recall": pytest.approx(0.5)}


def test_score_gain_control_allows_only_six_decimal_storage_rounding():
    mod = _module()
    mod.assert_score_gain_close(0.123456, 0.123457, "t1")
    with pytest.raises(ValueError, match="score gain"):
        mod.assert_score_gain_close(0.123456, 0.123459, "t1")


def test_tied_block_boundary_count_reports_split_ordering_values():
    mod = _module()
    assert mod.count_tied_block_boundaries([1, 1, 1, 1, 2, 2, 2, 3, 3], n_blocks=3) == 2
    assert mod.count_tied_block_boundaries(list(range(9)), n_blocks=3) == 0
