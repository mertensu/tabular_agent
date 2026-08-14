"""What ships is the average of the terminal fits, not the best of them (0044).

Three claims, and they are separable. The bundle *contains* every candidate at
equal weight; it *combines* them on the target's own scale rather than the
model's; and the run still records what the argmax would have shipped, on the
sealed rows, so the choice stays falsifiable by the next run rather than settled
by the one that made it.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from tabular_agent.harness import journal
from tabular_agent.harness.session import RunSession
from tabular_agent.harness.tools import (
    BLEND_ID,
    TERMINAL_MODEL_TYPES,
    fit_encoding,
    finalize_run,
    inspect_data,
)
from tabular_agent.models import predict_model
from tabular_agent.pipeline import load_bundle, predict_bundle
from tabular_agent.target import invert_target
from tabular_agent.taskspec import TaskSpec


def _frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "city": rng.choice(["berlin", "hamburg", "koeln"], size=n),
            "y": (x1 + 0.5 * x2 + rng.normal(scale=0.3, size=n) > 0).astype(int),
        }
    )


def _priced(n=300, seed=1):
    """A regression target on a scale where `rmsle` implies a log1p transform.

    Needed for the inversion-order test: on a linear transform (or none) the two
    orders agree, so only a non-linear one can tell them apart.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "city": rng.choice(["berlin", "hamburg", "koeln"], size=n),
            "price": np.exp(9.0 + 0.6 * x1 + 0.3 * x2 + rng.normal(scale=0.2, size=n)),
        }
    )


def _session(tmp_path, df, spec, name="run", **kwargs):
    path = str(tmp_path / f"{name}.csv")
    df.to_csv(path, index=False)
    return RunSession(str(tmp_path / name), path, spec, **kwargs)


@pytest.fixture
def run(tmp_path):
    return _session(tmp_path, _frame(), TaskSpec(target="y", n_splits=3))


@pytest.fixture
def sealed(tmp_path):
    return _session(
        tmp_path, _frame(), TaskSpec(target="y", n_splits=3), holdout_fraction=0.2
    )


# --------------------------------------------------------------- what is in it


def test_the_bundle_holds_every_terminal_fit_at_equal_weight(run):
    inspect_data(run)
    fit_encoding(run, ["city"], "frequency")
    out = finalize_run(run)

    bundle = load_bundle(os.path.join(run.run_dir, out["bundle_dir"]))
    assert len(bundle["models"]) == len(out["blend_members"]) > 1
    # Equal, and summing to one: no weight was fitted to anything, which is the
    # whole reason the blend costs no selection.
    assert np.allclose(bundle["weights"], 1.0 / len(bundle["models"]))
    assert bundle["weights"].sum() == pytest.approx(1.0)
    assert out["model_type"].startswith("blend(")


def test_a_failed_family_is_left_out_rather_than_ending_the_run(run, monkeypatch):
    """Three families exist so one can be wrong for the data. The blend averages
    whatever survived — it does not require the full ballot."""
    import tabular_agent.harness.tools as tools

    monkeypatch.setattr(tools, "TERMINAL_MODEL_TYPES", (tools.PROBE_MODEL_TYPE, "nonsense"))
    inspect_data(run)
    out = finalize_run(run)

    bundle = load_bundle(os.path.join(run.run_dir, out["bundle_dir"]))
    # One survivor is not a blend, and must not be dressed up as one.
    assert len(bundle["models"]) == 1
    assert "blend_members" not in out
    assert not out["model_type"].startswith("blend(")


def test_every_member_shares_one_table_so_one_pipeline_serves_them_all(run):
    inspect_data(run)
    fit_encoding(run, ["city"], "frequency")
    out = finalize_run(run)

    metas = []
    for exp_id in out["blend_members"]:
        with open(os.path.join(run.run_dir, "experiments", exp_id, "meta.json")) as f:
            metas.append(json.load(f))
    assert len({m["source_table"] for m in metas}) == 1
    assert len({m["target_transform"] for m in metas}) == 1
    assert all(m["feature_names"] == metas[0]["feature_names"] for m in metas)


# ------------------------------------------------------- how they are combined


def test_predictions_are_averaged_after_the_transform_is_inverted(tmp_path):
    """The order is the estimator. Averaging on the log scale is a geometric mean
    of price and a different number, and it is not the one that was measured."""
    run = _session(
        tmp_path, _priced(), TaskSpec(target="price", metric="rmsle", n_splits=3)
    )
    inspect_data(run)
    out = finalize_run(run)
    bundle_dir = os.path.join(run.run_dir, out["bundle_dir"])
    bundle = load_bundle(bundle_dir)
    assert bundle["target_transform"] == "log1p"
    assert len(bundle["models"]) > 1

    raw = _priced(n=25, seed=7).drop(columns=["price"])
    got = predict_bundle(bundle, raw)

    X = bundle["pipeline"].transform(raw)[bundle["feature_names"]]
    each = [predict_model(m, X, bundle["task_type"]) for m in bundle["models"]]
    arithmetic = np.mean([invert_target("log1p", p) for p in each], axis=0)
    geometric = invert_target("log1p", np.mean(each, axis=0))

    assert np.allclose(got, arithmetic)
    # And the two really are distinguishable on this data, or the test proves
    # nothing: Jensen puts the geometric mean strictly below the arithmetic one
    # wherever the members disagree at all.
    assert not np.allclose(arithmetic, geometric)


def test_a_single_model_bundle_predicts_exactly_as_it_did_before(tmp_path):
    """The inversion order changed in 0044. For one model it is a no-op, and that
    covers every bundle written before it."""
    run = _session(
        tmp_path, _priced(), TaskSpec(target="price", metric="rmsle", n_splits=3)
    )
    inspect_data(run)
    out = finalize_run(run)
    bundle = load_bundle(os.path.join(run.run_dir, out["bundle_dir"]))

    raw = _priced(n=25, seed=7).drop(columns=["price"])
    one = dict(bundle, models=bundle["models"][:1], weights=np.array([1.0]))
    X = one["pipeline"].transform(raw)[one["feature_names"]]
    direct = invert_target(
        "log1p", predict_model(one["models"][0], X, one["task_type"])
    )
    assert np.allclose(predict_bundle(one, raw), direct)


# ------------------------------------------------- and what it is measured against


def test_the_blend_and_the_argmax_are_both_scored_on_the_seal(sealed):
    """The A/B, recomputed every run. `selected` is what the search crowned and
    `shipped` is what the bundle predicts with, and keeping both is what stops
    0044 from being a decision no later run can overturn."""
    inspect_data(sealed)
    fit_encoding(sealed, ["city"], "frequency")
    out = finalize_run(sealed)

    scores = journal.load(sealed.run_dir)["holdout"]["scores"]
    selected = [e for e in scores if e["selected"]]
    shipped = [e for e in scores if e.get("shipped")]
    assert len(selected) == len(shipped) == 1
    # Different rows, and the blend is not one of the candidates it averages.
    assert shipped[0]["experiment_id"] == BLEND_ID
    assert selected[0]["experiment_id"] != BLEND_ID
    assert selected[0]["experiment_id"] == out["best_experiment_id"]
    # Both have a sealed score, so the comparison is always available.
    assert selected[0]["holdout_score"] is not None
    assert shipped[0]["holdout_score"] is not None
    assert set(shipped[0]["members"]) == set(out["blend_members"])


def test_the_blend_is_scored_on_the_validation_half_too(sealed):
    """Its own `optimism` needs a val number in the same units as the others'.
    The expectation is that it is small: an average selects nothing, so it has no
    luck in its validation score to give back."""
    inspect_data(sealed)
    fit_encoding(sealed, ["city"], "frequency")
    out = finalize_run(sealed)

    shipped = next(
        e for e in journal.load(sealed.run_dir)["holdout"]["scores"] if e.get("shipped")
    )
    assert shipped["val_score"] is not None
    assert shipped["optimism"] is not None
    # The headline the report leads with is the blend's, not the argmax winner's.
    assert out["val_score"] == shipped["val_score"]


def test_the_seal_still_picks_nothing(sealed):
    """0023 rule 2 survives blending: the members are fixed before any sealed row
    is read, so no holdout score can change what ships."""
    inspect_data(sealed)
    fit_encoding(sealed, ["city"], "frequency")
    out = finalize_run(sealed)

    scores = journal.load(sealed.run_dir)["holdout"]["scores"]
    candidates = [e for e in scores if e["experiment_id"] != BLEND_ID]
    # Every terminal fit is in the blend, including whichever one the sealed rows
    # happened to like least. Nothing was dropped after the fact.
    assert set(out["blend_members"]) == {e["experiment_id"] for e in candidates}


def test_a_run_without_a_seal_still_blends(run):
    """Blending is not a feature of the third split — it is what finalize does."""
    inspect_data(run)
    out = finalize_run(run)
    assert len(out["blend_members"]) > 1
    assert "holdout_score" not in out
    assert journal.load(run.run_dir).get("holdout") is None


def test_the_report_names_the_blend_and_prints_the_gap(sealed):
    inspect_data(sealed)
    fit_encoding(sealed, ["city"], "frequency")
    out = finalize_run(sealed)

    text = open(os.path.join(sealed.run_dir, out["report_path"])).read()
    assert "What shipped is not that candidate" in text
    assert "(shipped)" in text and "(search's pick)" in text
    # A blend was never cross-validated, so the report must neither print folds
    # for it nor send the reader to a CV score it does not have.
    assert "## Per-fold metrics" not in text
    assert "- **CV score:**" not in text
    assert "The gap to the CV score" not in text
