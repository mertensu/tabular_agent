import json

import pytest

from tabular_agent.capability_review import (
    REVIEW_FILENAME,
    build_review_context,
    capability_backlog,
    review_capabilities,
)
from tabular_agent.dashboard import build_dashboard_data, write_dashboard_artifacts
from tabular_agent.runctx import RunContext


def _finalized_run(path):
    path.mkdir(parents=True, exist_ok=True)
    data = build_dashboard_data(
        task_type="multiclass",
        metric="logloss",
        selection_score=0.48,
        baseline_score=2.1,
        model_types=["lgbm", "catboost"],
        feature_names=["x"],
        split={"profile": "full", "n_work": 70, "n_val": 15, "n_holdout": 15},
        holdout={
            "scores": [
                {
                    "experiment_id": "e1",
                    "model_type": "lgbm",
                    "selected": True,
                    "shipped": False,
                    "val_score": 0.48,
                    "holdout_score": 0.50,
                    "optimism": 0.02,
                },
                {
                    "experiment_id": "blend",
                    "model_type": "blend",
                    "selected": False,
                    "shipped": True,
                    "val_score": 0.49,
                    "holdout_score": 0.52,
                    "optimism": 0.03,
                },
            ]
        },
        blend_weights=[0.5, 0.5],
    )
    write_dashboard_artifacts(RunContext(str(path)), "# Run Report\n", data)
    (path / "run_journal.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": {"target": "private_target", "task_type": "multiclass", "metric": "logloss"},
                "split": {"profile": "full"},
                "tables": [
                    {
                        "id": "t1",
                        "path": "/sensitive/tables/t1.parquet",
                        "source_table": "/sensitive/raw.parquet",
                        "method": "transform_features",
                        "new_columns": ["row_total"],
                        "probe_score": 0.51,
                        "code": "TOP_SECRET_TRANSFORM_CODE",
                    }
                ],
                "leaderboard": [
                    {
                        "id": "e1",
                        "model_type": "lgbm",
                        "source_table": "/sensitive/raw.parquet",
                        "metric": "logloss",
                        "val_score": 0.48,
                        "features": ["x"],
                        "artifact_path": "/sensitive/experiments/e1",
                    }
                ],
                "stop": {"refusals": 1, "finalized": True},
            }
        )
    )
    return path


def _finding(evidence="model:e1"):
    return {
        "overall_assessment": "Features were not the demonstrated bottleneck.",
        "findings": [
            {
                "category": "model_selection_gap",
                "title": "Model portfolio limited the run <script>alert(1)</script>",
                "generic_capability_key": "stronger_multiclass_model_portfolio",
                "generic_capability": "Evaluate stronger general-purpose multiclass estimators.",
                "evidence_refs": [evidence, "shipped"],
                "existing_tools_considered": ["transform_features"],
                "why_existing_tools_insufficient": "The feature operation was expressible and lost.",
                "alternative_explanation": "The split may be unusually difficult.",
                "leakage_or_serving_risk": "Selection must remain validation-only.",
                "testable_hypothesis": "Pre-register one estimator and test it on an unseen run.",
                "confidence": "high",
            }
        ],
    }


def test_review_context_is_allowlisted_and_contains_no_code_or_paths(tmp_path):
    run_dir = _finalized_run(tmp_path / "run")
    context = build_review_context(run_dir)
    serialized = json.dumps(context)

    assert "TOP_SECRET_TRANSFORM_CODE" not in serialized
    assert "/sensitive/" not in serialized
    assert "artifact_path" not in serialized
    assert "table:t1" in serialized
    assert context["profile"] == "full"
    assert "transform_features" in context["available_tools"]


def test_reviewer_is_write_once_and_updates_all_report_views(tmp_path):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    run_dir = _finalized_run(tmp_path / "run")
    model = TestModel(custom_output_args=_finding())
    from tabular_agent import Agent

    first = Agent(model=model).review_capabilities(run_dir)
    second = review_capabilities(run_dir, model=object())

    assert second == first
    assert first["findings"][0]["fingerprint"]
    assert (run_dir / REVIEW_FILENAME).exists()
    report = json.loads((run_dir / "run_report.json").read_text())
    assert report["capability_review"] == first
    markdown = (run_dir / "run_report.md").read_text()
    assert "What limited this run?" in markdown
    assert "<script>alert(1)</script>" not in markdown
    html = (run_dir / "run_report.html").read_text()
    assert 'id="capability-review"' in html
    assert "Model portfolio limited the run" in html
    assert "<script>alert(1)</script>" not in html
    assert "ship this model" not in html.lower()


def test_unknown_evidence_is_refused_before_an_artifact_is_written(tmp_path):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    run_dir = _finalized_run(tmp_path / "run")
    with pytest.raises(ValueError, match="unknown evidence IDs"):
        review_capabilities(run_dir, model=TestModel(custom_output_args=_finding("invented:42")))
    assert not (run_dir / REVIEW_FILENAME).exists()


def test_backlog_marks_recurrence_but_never_accepts_it(tmp_path):
    run_a = _finalized_run(tmp_path / "run-a")
    run_b = _finalized_run(tmp_path / "run-b")
    review = {
        "schema_version": 1,
        "status": "proposed",
        "findings": [
            {
                **_finding()["findings"][0],
                "fingerprint": "same-fingerprint",
            }
        ],
    }
    for run_dir in (run_a, run_b):
        (run_dir / REVIEW_FILENAME).write_text(json.dumps(review))

    backlog = capability_backlog([run_a, run_b, run_a])

    assert backlog["n_runs"] == 2
    assert backlog["items"][0]["status"] == "recurring"
    assert backlog["items"][0]["n_runs"] == 2
    assert {item["status"] for item in backlog["items"]} <= {"proposed", "recurring"}
