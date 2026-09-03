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


def test_both_scores_are_reported_so_the_gap_is_visible():
    """The validation score picked the winner; the CV score says what the folds
    thought. Reporting only one of them hides how far the search drifted."""
    text = build_report(
        task_type="binary",
        metric="auc",
        selection_score=0.9,
        cv_score=0.94,
        fold_metrics=[0.94],
        baseline_score=0.5,
        model_types=["lgbm"],
        feature_names=["a"],
    )
    assert "Prediction output" in text and "0.5 threshold" in text
    assert "Shipped validation score" in text
    assert "0.900" in text
    assert "0.940" in text
    assert "validation" in text.lower()


def test_the_split_line_says_the_score_is_a_public_leaderboard():
    """Every experiment was ranked against these rows, so the number the report
    leads with is the one the search was steering towards. Say so, or it reads as
    an independent check on the winner."""
    text = build_report(
        task_type="binary",
        metric="auc",
        selection_score=0.9,
        cv_score=0.94,
        fold_metrics=[0.94],
        baseline_score=0.5,
        model_types=["lgbm"],
        feature_names=["a"],
        split={"n_work": 160, "n_val": 40, "split_on": None},
    )
    assert "160 work / 40 validation" in text
    assert "public leaderboard" in text.lower()


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


def test_classification_report_names_stratification_balance_and_row_uncertainty():
    text = build_report(
        task_type="binary", metric="accuracy", selection_score=0.8,
        fold_metrics=[0.78, 0.82], baseline_score=0.6,
        model_types=["lgbm"], feature_names=["x"],
        split={
            "n_work": 60, "n_val": 20, "n_holdout": 20, "split_on": None,
            "stratified": True,
            "target_composition": {
                "work": [{"label": "0", "count": 42, "share": 0.7},
                         {"label": "1", "count": 18, "share": 0.3}],
                "validation": [{"label": "0", "count": 14, "share": 0.7},
                               {"label": "1", "count": 6, "share": 0.3}],
                "holdout": [{"label": "0", "count": 14, "share": 0.7},
                            {"label": "1", "count": 6, "share": 0.3}],
            },
        },
        holdout={"scores": [{
            "experiment_id": "blend", "model_type": "blend", "selected": True,
            "shipped": True, "val_score": 0.8, "holdout_score": 0.85,
            "optimism": -0.05,
        }]},
    )
    assert "random, stratified" in text
    assert "Target composition" in text
    assert "work [0: 70.0%, 1: 30.0%]" in text
    assert text.count("row-level 95% interval") == 2


def test_ordered_accuracy_report_does_not_claim_binomial_uncertainty():
    text = build_report(
        task_type="binary", metric="accuracy", selection_score=0.8,
        fold_metrics=[0.8], baseline_score=0.6, model_types=["lgbm"],
        feature_names=["x"],
        split={"n_work": 80, "n_val": 20, "n_holdout": 0, "split_on": "date"},
    )
    assert "row-level 95% interval" not in text
