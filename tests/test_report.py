import os

from tabular_agent.report import build_report, write_report
from tabular_agent.runctx import RunContext


def test_build_report_contains_key_numbers():
    text = build_report(
        task_type="regression",
        metric="rmse",
        selection_score=0.123,
        fold_metrics=[0.11, 0.12, 0.13, 0.12, 0.14],
        baseline_score=0.500,
        model_types=["lgbm", "xgb"],
        feature_names=["a", "b", "c_te"],
    )
    assert "regression" in text
    assert "rmse" in text
    assert "0.123" in text
    assert "0.500" in text  # baseline shown for honesty
    assert "lgbm" in text and "xgb" in text


def test_report_flags_when_baseline_not_beaten():
    text = build_report(
        task_type="regression",
        metric="rmse",
        selection_score=0.6,
        fold_metrics=[0.6],
        baseline_score=0.5,  # model worse than baseline
        model_types=["lgbm"],
        feature_names=["a"],
    )
    assert "WARNING" in text


def test_holdout_line_present_when_given():
    text = build_report(
        task_type="binary",
        metric="auc",
        selection_score=0.9,
        fold_metrics=[0.9],
        baseline_score=0.5,
        model_types=["lgbm"],
        feature_names=["a"],
        holdout_score=0.88,
    )
    assert "0.88" in text
    assert "holdout" in text.lower()


def test_write_report_creates_file(tmp_path):
    ctx = RunContext(str(tmp_path / "run"))
    path = write_report(ctx, "# hello")
    assert os.path.exists(path)
    assert path.endswith("run_report.md")


def test_the_feature_story_reads_oldest_first():
    """The chain is stored newest-first; the report has to tell it forwards."""
    text = build_report(
        task_type="binary",
        metric="logloss",
        selection_score=0.27,
        fold_metrics=[0.27, 0.28],
        baseline_score=0.69,
        model_types=["lgbm"],
        feature_names=["a"],
        attribution=[
            {"table": "tables/t2.parquet", "added": ["city_freq"], "method": "frequency",
             "n_experiments": 1, "best_score": 0.27, "measured": True},
            {"table": "tables/t1.parquet", "added": ["ratio"], "method": "transform",
             "n_experiments": 1, "best_score": 0.31, "measured": True},
        ],
    )
    assert text.index("tables/t1.parquet") < text.index("tables/t2.parquet")
    assert "Attribution gap" not in text


def test_an_unmeasured_link_is_called_out():
    text = build_report(
        task_type="binary",
        metric="logloss",
        selection_score=0.27,
        fold_metrics=[0.27, 0.28],
        baseline_score=0.69,
        model_types=["lgbm"],
        feature_names=["a"],
        attribution=[
            {"table": "tables/t6.parquet", "added": ["city_freq"], "method": "frequency",
             "n_experiments": 4, "best_score": 0.27, "measured": True},
            {"table": "tables/t5.parquet", "added": ["log_price", "price_sq"],
             "method": "transform", "n_experiments": 0, "best_score": None, "measured": False},
        ],
    )
    assert "never trained on" in text
    assert "Attribution gap" in text and "tables/t5.parquet" in text
