import json

import pytest

from tabular_agent.dashboard import (
    build_dashboard_data,
    render_dashboard_html,
    update_dashboard_contract,
    write_dashboard_artifacts,
)
from tabular_agent.runctx import RunContext


def _data(**overrides):
    values = {
        "task_type": "regression",
        "metric": "rmse",
        "selection_score": 0.12,
        "baseline_score": 0.5,
        "model_types": ["lgbm", "xgb"],
        "feature_names": ["amount", "city_te"],
        "split": {
            "n_work": 80,
            "n_val": 10,
            "n_holdout": 10,
            "split_on": None,
        },
        "holdout": {
            "scores": [
                {
                    "experiment_id": "e1",
                    "model_type": "lgbm",
                    "selected": True,
                    "shipped": False,
                    "val_score": 0.11,
                    "holdout_score": 0.14,
                    "optimism": 0.03,
                },
                {
                    "experiment_id": "blend",
                    "model_type": "blend",
                    "selected": False,
                    "shipped": True,
                    "val_score": 0.12,
                    "holdout_score": 0.13,
                    "optimism": 0.01,
                },
            ]
        },
        "feature_catalog": [
            {
                "feature": "city_te",
                "kind": "target",
                "source_columns": ["city"],
                "pipeline_step": 1,
                "description": "OOF city statistic",
                "definition": '{"columns": ["city"]}',
            }
        ],
        "blend_weights": [0.6, 0.4],
    }
    values.update(overrides)
    return build_dashboard_data(**values)


def test_dashboard_is_offline_accessible_and_escapes_feature_content():
    data = _data(feature_catalog=[{
        "feature": "<script>alert(1)</script>",
        "kind": "row_transform",
        "source_columns": ["raw"],
        "pipeline_step": 1,
        "description": "derived",
        "definition": "</pre><script>alert(2)</script>",
    }])
    html = render_dashboard_html(data)
    assert "<script>alert(1)</script>" not in html
    assert "<script>alert(2)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "https://" not in html and "http://" not in html
    assert 'tabindex="0"' in html
    assert 'aria-label="Validation:' in html
    assert 'id="feature-search"' in html
    assert 'class="score-bar"' in html and 'class="optimism-bar bad"' in html


def test_dashboard_json_contains_scores_and_safe_feature_recipes_not_lookup_maps(tmp_path):
    data = _data()
    paths = write_dashboard_artifacts(RunContext(str(tmp_path)), "# report", data)
    stored = json.loads((tmp_path / "run_report.json").read_text())
    assert stored["summary"]["validation_score"] == 0.12
    assert stored["summary"]["test_score"] == 0.13
    assert stored["features"]["catalog"][0]["definition"] == '{"columns": ["city"]}'
    serialized = json.dumps(stored)
    assert "mapping" not in serialized and "lookup" not in serialized
    assert (tmp_path / "run_report.md").read_text() == "# report"
    assert paths["html"].endswith("run_report.html")


def test_confirmed_facade_contract_enriches_json_and_html(tmp_path):
    data = _data(serving_contract={"target": "y", "required_streams": ["events"]})
    write_dashboard_artifacts(RunContext(str(tmp_path)), "# report", data)
    update_dashboard_contract(str(tmp_path), {
        "target": "y",
        "serving_columns": ["id", "amount"],
        "identifier_columns": ["id"],
        "joins": [{"right_table": "accounts.parquet"}],
    })
    stored = json.loads((tmp_path / "run_report.json").read_text())
    contract = stored["serving_contract"]
    assert contract["identifier_columns"] == ["id"]
    assert contract["joins"] == ["accounts.parquet"]
    assert contract["required_streams"] == ["events"]
    html = (tmp_path / "run_report.html").read_text()
    assert "accounts.parquet" in html and "Identifiers kept out of models" in html


def test_metric_direction_controls_baseline_gain():
    accuracy = _data(
        task_type="binary",
        metric="accuracy",
        selection_score=0.8,
        baseline_score=0.6,
    )
    rmse = _data(selection_score=0.4, baseline_score=0.5)
    assert accuracy["summary"]["gain_vs_baseline"] == pytest.approx(0.2)
    assert rmse["summary"]["gain_vs_baseline"] == pytest.approx(0.1)
    assert accuracy["summary"]["beats_baseline"] is True
    assert rmse["summary"]["beats_baseline"] is True


def test_dashboard_names_the_forecast_cutoff_contract():
    html = render_dashboard_html(_data(split={
        "workflow": "forecast",
        "n_work": 80,
        "n_val": 10,
        "n_holdout": 10,
        "split_on": "Date",
        "forecast_horizon": 42,
    }))
    assert "forecast ordered by Date, horizon 42" in html
