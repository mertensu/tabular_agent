"""The stop rule (§19.3), tested against hand-written leaderboards.

`stop.evaluate` is a pure function of the leaderboard precisely so these cases
can be written out instead of trained: the interesting histories (a creeping
series of sub-noise gains, a dip before a real improvement) would each take a
dozen model fits to produce by accident.
"""

from tabular_agent.harness.stop import StopPolicy, advisory, evaluate


def row(id, score, spread=0.04, model_type="lgbm"):
    return {"id": id, "model_type": model_type, "cv_score": score, "fold_spread": spread}


POLICY = StopPolicy(patience=3, min_gain_ratio=0.25, max_experiments=25)


def test_an_empty_or_baseline_only_run_never_stops():
    assert not evaluate([], "logloss", POLICY)["should_stop"]
    only_baseline = [row("e1", 0.69, model_type="baseline")]
    status = evaluate(only_baseline, "logloss", POLICY)
    assert not status["should_stop"]
    # The mandatory baseline must not spend the patience budget.
    assert status["since_improvement"] == 0


def test_real_improvements_reset_the_patience():
    lb = [
        row("e1", 0.69, model_type="baseline"),
        row("e2", 0.50),
        row("e3", 0.40),
        row("e4", 0.30),
    ]
    status = evaluate(lb, "logloss", POLICY)
    assert status["since_improvement"] == 0
    assert not status["should_stop"]


def test_gains_inside_the_fold_noise_do_not_count():
    """The observed failure: six experiments moving the score by ~0.003."""
    lb = [row("e1", 0.69, model_type="baseline"), row("e2", 0.300, spread=0.04)]
    for i, score in enumerate([0.299, 0.298, 0.2975]):
        lb.append(row(f"e{i + 3}", score, spread=0.04))

    status = evaluate(lb, "logloss", POLICY)
    # Each step improved the number, and none of them improved the model.
    assert status["since_improvement"] == 3
    assert status["should_stop"]
    assert "inside the noise" in status["reason"]


def test_a_gain_above_the_noise_floor_counts_even_if_small():
    lb = [row("e1", 0.300, spread=0.01), row("e2", 0.29)]  # 0.01 > 0.25 * 0.01
    assert evaluate(lb, "logloss", POLICY)["since_improvement"] == 0


def test_a_worse_experiment_costs_patience_but_is_not_fatal():
    """A run branches: trying something that turns out worse is the search working."""
    lb = [row("e1", 0.30), row("e2", 0.45), row("e3", 0.20)]
    status = evaluate(lb, "logloss", POLICY)
    assert status["since_improvement"] == 0  # e3 was a real improvement
    assert not status["should_stop"]


def test_higher_is_better_metrics_are_read_the_right_way_round():
    lb = [row("e1", 0.70), row("e2", 0.85)]
    assert evaluate(lb, "auc", POLICY)["since_improvement"] == 0
    assert evaluate(lb, "logloss", POLICY)["since_improvement"] == 1


def test_the_experiment_ceiling_stops_a_run_that_keeps_improving():
    policy = StopPolicy(patience=99, max_experiments=4)
    lb = [row(f"e{i}", 0.5 - i * 0.1) for i in range(4)]
    status = evaluate(lb, "logloss", policy)
    assert status["should_stop"]
    assert "budget" in status["reason"]
    assert status["experiments_remaining"] == 0


def test_the_advisory_says_something_actionable_in_each_state():
    fresh = evaluate([row("e1", 0.3)], "logloss", POLICY)
    assert "best result so far" in advisory(fresh)

    stalling = evaluate([row("e1", 0.30), row("e2", 0.2999)], "logloss", POLICY)
    assert "2 more without one" in advisory(stalling)

    stopped = evaluate(
        [row("e1", 0.30), row("e2", 0.2999), row("e3", 0.2998), row("e4", 0.2997)],
        "logloss",
        POLICY,
    )
    assert "finalize_run" in advisory(stopped)
