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
