"""The v0.2 minimal product profile: one evaluation path, one shipped model."""

import json
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pydantic_ai")

from tabular_agent.harness import journal
from tabular_agent.harness.agent import build_agent
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import finalize_run, inspect_data, transform_features
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.taskspec import TaskSpec


MINIMAL_TOOLS = {
    "inspect_data",
    "fit_encoding",
    "fit_group_stats",
    "filter_rows",
    "transform_features",
    "finalize_run",
}


def _session(tmp_path, *, ordered=False):
    rng = np.random.default_rng(42)
    n = 240
    x = rng.normal(size=n)
    df = pd.DataFrame({
        "x": x,
        "category": rng.choice(["a", "b", "c"], n),
        "day": np.arange(n),
        "y": 3 * x + rng.normal(scale=0.2, size=n) + 10,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return RunSession(
        str(tmp_path / "run"), str(path), TaskSpec(target="y", n_splits=3),
        split_on="day" if ordered else None,
        profile="minimal",
    )


def test_minimal_agent_has_only_the_product_tools():
    agent = build_agent(profile="minimal")
    assert set(agent._function_toolset.tools) == MINIMAL_TOOLS


def test_minimal_profile_refuses_a_third_internal_split(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": range(30), "y": range(30)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="one train/evaluation split"):
        RunSession(
            str(tmp_path / "run"), str(path), TaskSpec(target="y"),
            profile="minimal", holdout_fraction=0.1,
        )


def test_explicitly_empty_serving_schema_is_enforced(tmp_path):
    base = _session(tmp_path)
    session = RunSession(
        str(tmp_path / "empty-serving-run"), base.source_path,
        TaskSpec(target="y", n_splits=3), profile="minimal", serving_columns=[],
    )
    inspect_data(session)
    with pytest.raises(ValueError, match="cannot exist at prediction time"):
        finalize_run(session)


def test_persisted_profile_cannot_change_on_resume(tmp_path):
    minimal = _session(tmp_path)
    inspect_data(minimal)
    with pytest.raises(ValueError, match="was created with profile 'minimal'"):
        RunSession(minimal.run_dir, minimal.source_path, TaskSpec(target="y"))
    resumed = RunSession(
        minimal.run_dir, minimal.source_path, TaskSpec(target="y"), profile="minimal"
    )
    assert resumed.profile == "minimal"

    full = RunSession(
        str(tmp_path / "full-run"), minimal.source_path, TaskSpec(target="y")
    )
    inspect_data(full)
    with pytest.raises(ValueError, match="was created with profile 'full'"):
        RunSession(
            full.run_dir, full.source_path, TaskSpec(target="y"), profile="minimal"
        )


def test_minimal_table_payload_has_scores_not_uncertainty(tmp_path):
    session = _session(tmp_path, ordered=True)
    info = inspect_data(session)
    assert "n_splits" not in info
    out = transform_features(
        session,
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['x2'] = df['x'] ** 2\n"
        "    return df",
    )
    assert out["evaluation_score"] is not None
    assert "gain_vs_parent" in out
    assert "probe_se" not in out
    assert "probe_vs_source_interval" not in out
    assert not os.path.exists(os.path.join(session.run_dir, "probe_preds", "t1.npy"))


def test_minimal_finalize_ships_one_model_without_cv_or_blend(tmp_path):
    session = _session(tmp_path)
    inspect_data(session)
    transform_features(
        session,
        "def transform_features(df):\n"
        "    df = df.copy()\n"
        "    df['x2'] = df['x'] ** 2\n"
        "    return df",
    )

    out = finalize_run(session)
    assert out["model_type"] == "lgbm"
    assert out["evaluation_score"] is not None
    assert out["baseline_score"] is not None
    assert "blend_members" not in out
    assert "cv_score" not in out
    assert "val_se" not in out

    j = journal.load(session.run_dir)
    trained = [row for row in j["leaderboard"] if row["model_type"] != "baseline"]
    assert len(trained) == 1
    assert trained[0]["model_type"] == "lgbm"
    assert "cv_score" not in trained[0]
    assert "fold_spread" not in trained[0]

    exp_dir = os.path.join(session.run_dir, trained[0]["artifact_path"])
    assert not os.path.exists(os.path.join(exp_dir, "oof.parquet"))

    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))
    raw = pd.read_csv(session.source_path).drop(columns=["y"]).head(10)
    preds = predict_bundle(bundle, raw)
    assert len(preds) == 10 and np.isfinite(preds).all()

    meta = json.loads(open(os.path.join(exp_dir, "meta.json")).read())
    assert "cv_score" not in meta and meta["evaluation_score"] == out["evaluation_score"]

    report = open(os.path.join(session.run_dir, out["report_path"])).read().lower()
    for removed_concept in ("cross-validation", "bootstrap", "sealed", "standard error"):
        assert removed_concept not in report


def test_minimal_finalize_retry_does_not_duplicate_journal_rows(tmp_path, monkeypatch):
    from tabular_agent.harness import tools as tools_module

    session = _session(tmp_path)
    inspect_data(session)
    original = tools_module.write_report

    def fail_once(*args, **kwargs):
        raise RuntimeError("simulated report interruption")

    monkeypatch.setattr(tools_module, "write_report", fail_once)
    with pytest.raises(RuntimeError, match="interruption"):
        finalize_run(session)
    assert journal.load(session.run_dir)["leaderboard"] == []
    experiments = os.path.join(session.run_dir, "experiments")
    assert not os.path.exists(experiments) or os.listdir(experiments) == []

    monkeypatch.setattr(tools_module, "write_report", original)
    finalize_run(session)
    rows = journal.load(session.run_dir)["leaderboard"]
    assert [row["model_type"] for row in rows] == ["baseline", "lgbm"]


def test_minimal_finalize_is_serialized(tmp_path):
    from tabular_agent.harness import tools as tools_module

    session = _session(tmp_path)
    with tools_module._minimal_finalize_lock(session):
        with pytest.raises(ValueError, match="already finalizing"):
            with tools_module._minimal_finalize_lock(session):
                pass


def test_minimal_classification_golden_fixture(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    df = pd.DataFrame({"x": x, "y": (x + rng.normal(scale=0.4, size=300) > 0).astype(int)})
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    session = RunSession(
        str(tmp_path / "classification-run"), str(path), TaskSpec(target="y"),
        profile="minimal",
    )
    inspect_data(session)
    out = finalize_run(session)
    bundle = load_bundle(os.path.join(session.run_dir, out["bundle_dir"]))
    preds = predict_bundle(bundle, df.drop(columns=["y"]).head(12))
    assert out["metric"] == "logloss"
    assert len(preds) == 12 and np.isfinite(preds).all()


def test_minimal_profile_refuses_unsupported_positive_class(tmp_path):
    path = tmp_path / "labels.csv"
    pd.DataFrame({"x": range(40), "y": ["no", "yes"] * 20}).to_csv(path, index=False)
    session = RunSession(
        str(tmp_path / "labels-run"), str(path),
        TaskSpec(target="y", positive_class="no"), profile="minimal",
    )
    with pytest.raises(ValueError, match="positive_class.*minimal profile"):
        inspect_data(session)


@pytest.mark.parametrize(
    ("labels", "metric"),
    [
        (np.array(["no", "yes"] * 100), "logloss"),
        (np.resize(np.array([10, 20, 30]), 210), "accuracy"),
    ],
)
def test_minimal_classification_preserves_real_labels(tmp_path, labels, metric):
    rng = np.random.default_rng(11)
    path = tmp_path / f"{metric}.csv"
    pd.DataFrame({"x": rng.normal(size=len(labels)), "y": labels}).to_csv(path, index=False)
    session = RunSession(
        str(tmp_path / f"{metric}-run"), str(path),
        TaskSpec(target="y", metric=metric), profile="minimal",
    )
    inspect_data(session)
    out = finalize_run(session)
    assert out["evaluation_score"] is not None


def test_minimal_random_accuracy_report_names_balance_and_row_interval(tmp_path):
    path = tmp_path / "classification.csv"
    pd.DataFrame(
        {"x": range(100), "y": [0] * 90 + [1] * 10}
    ).to_csv(path, index=False)
    session = RunSession(
        str(tmp_path / "classification-run"), str(path),
        TaskSpec(target="y", task_type="binary", metric="accuracy"),
        profile="minimal",
    )

    inspect_data(session)
    out = finalize_run(session)
    report = open(os.path.join(session.run_dir, out["report_path"])).read()

    assert "random, stratified" in report
    assert "Target composition" in report
    assert "work [0: 90.0%, 1: 10.0%]" in report
    assert "Evaluation row-level 95% interval" in report


def test_minimal_ordered_evaluation_golden_fixture(tmp_path):
    session = _session(tmp_path, ordered=True)
    inspect_data(session)
    out = finalize_run(session)
    split = journal.load(session.run_dir)["split"]
    assert split["mode"] == "ordered" and split["split_on"] == "day"
    assert split["profile"] == "minimal"
    assert out["evaluation_score"] is not None
    assert "optimism" not in out and "holdout_score" not in out
    report = open(os.path.join(session.run_dir, out["report_path"])).read().lower()
    for removed_concept in (
        "validation score", "public leaderboard", "uncertainty", "resampling", "sealed"
    ):
        assert removed_concept not in report
