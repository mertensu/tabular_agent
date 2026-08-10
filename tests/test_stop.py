"""The stop rule (§19.3), tested against hand-written leaderboards.

`stop.evaluate` is a pure function of the leaderboard precisely so these cases
can be written out instead of trained: the interesting histories (a creeping
series of sub-noise gains, a dip before a real improvement) would each take a
dozen model fits to produce by accident.
"""

from tabular_agent.harness.stop import StopPolicy, advisory, evaluate, table_status


def row(id, score, spread=0.04, model_type="lgbm", source_table="t1.parquet"):
    return {
        "id": id,
        "model_type": model_type,
        # The stop rule reads the leaderboard's ranking score, which is the
        # validation half (0019) — not the cross-validated one.
        "val_score": score,
        "fold_spread": spread,
        "source_table": source_table,
    }


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


def test_a_long_search_over_new_tables_is_not_stopped():
    """Eight ideas that did not work, to find the ninth that does. That is the job."""
    lb = [row("e1", 0.40, source_table="t1.parquet")]
    lb += [row(f"e{i + 2}", 0.41 + i * 0.01, source_table=f"t{i + 2}.parquet") for i in range(8)]

    status = evaluate(lb, "logloss", StopPolicy())
    assert status["since_improvement"] == 8
    assert not status["should_stop"]  # patience is 10: the search continues


def test_tuning_one_table_exhausts_that_table_and_not_the_run():
    """Six hyper-parameter settings on one table are one experiment costed six times."""
    lb = [row("e1", 0.40)] + [row(f"e{i + 2}", 0.41 + i * 0.005) for i in range(5)]

    run = evaluate(lb, "logloss", StopPolicy())
    assert not run["should_stop"]  # the run is fine

    table = table_status(lb, "logloss", StopPolicy(), "t1.parquet")
    assert table["exhausted"]  # this table is not
    assert "hyper-parameters" in table["reason"]


def test_a_fresh_table_is_never_exhausted():
    lb = [row("e1", 0.40)] + [row(f"e{i + 2}", 0.41) for i in range(5)]
    assert not table_status(lb, "logloss", StopPolicy(), "t2.parquet")["exhausted"]


def test_a_table_that_keeps_improving_stays_open():
    lb = [row(f"e{i + 1}", 0.5 - i * 0.1) for i in range(6)]
    assert not table_status(lb, "logloss", StopPolicy(), "t1.parquet")["exhausted"]


def test_experiments_on_other_tables_do_not_exhaust_this_one():
    lb = [
        row("e1", 0.40, source_table="t1.parquet"),
        row("e2", 0.41, source_table="t2.parquet"),
        row("e3", 0.42, source_table="t2.parquet"),
        row("e4", 0.43, source_table="t2.parquet"),
        row("e5", 0.44, source_table="t2.parquet"),
    ]
    assert not table_status(lb, "logloss", StopPolicy(), "t1.parquet")["exhausted"]
    assert table_status(lb, "logloss", StopPolicy(), "t2.parquet")["exhausted"]


def test_the_table_advisory_points_at_features_not_hyperparameters():
    lb = [row("e1", 0.40)] + [row(f"e{i + 2}", 0.41) for i in range(3)]
    note = advisory(
        evaluate(lb, "logloss", StopPolicy()),
        table_status(lb, "logloss", StopPolicy(), "t1.parquet"),
    )
    assert "transform_features" in note and "fit_encoding" in note


def test_the_advisory_says_something_actionable_in_each_state():
    fresh = evaluate([row("e1", 0.3)], "logloss", POLICY)
    assert "best result so far" in advisory(fresh)

    stalling = evaluate([row("e1", 0.30), row("e2", 0.2999)], "logloss", POLICY)
    assert "1 experiment(s) since the last real improvement" in advisory(stalling)

    stopped = evaluate(
        [row("e1", 0.30), row("e2", 0.2999), row("e3", 0.2998), row("e4", 0.2997)],
        "logloss",
        POLICY,
    )
    assert "finalize_run" in advisory(stopped)
