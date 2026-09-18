"""The stop rule (§19.3), tested against hand-written table histories.

`stop.evaluate` is a pure function of the journal's table list precisely so these
cases can be written out instead of built: the interesting histories (a creeping
series of sub-noise gains, a dip before a real improvement) would each take a
dozen fits to produce by accident.

Since 0035 the unit is the table, not the experiment — see the module docstring
in `stop.py` for why the two levels collapsed into one.
"""

from tabular_agent.harness.stop import (
    StopPolicy,
    advisory,
    best_table,
    best_probe_table,
    evaluate,
)


def row(id, score, se=0.04):
    return {"id": id, "path": f"tables/{id}.parquet", "probe_score": score, "probe_se": se}


POLICY = StopPolicy(patience=3, min_gain_ratio=0.25, max_tables=25)


def test_an_empty_run_never_stops():
    assert not evaluate([], "logloss", POLICY)["should_stop"]


def test_a_table_the_ruler_could_not_score_is_not_charged():
    """The agent did the work and the harness failed to weigh it. Ending a run
    over that would punish the wrong party."""
    unscored = [{"id": "t1", "probe_score": None, "probe_unavailable": "boom"}]
    status = evaluate(unscored, "logloss", POLICY)
    assert not status["should_stop"]
    assert status["since_improvement"] == 0


def test_real_improvements_reset_the_patience():
    tables = [row("t1", 0.50), row("t2", 0.40), row("t3", 0.30)]
    status = evaluate(tables, "logloss", POLICY)
    assert status["since_improvement"] == 0
    assert not status["should_stop"]


def test_gains_inside_the_noise_do_not_count():
    """The observed failure: six experiments moving the score by ~0.003."""
    tables = [row("t1", 0.300)]
    for i, score in enumerate([0.299, 0.298, 0.2975]):
        tables.append(row(f"t{i + 2}", score))

    status = evaluate(tables, "logloss", POLICY)
    # Each step improved the number, and none of them improved the model.
    assert status["since_improvement"] == 3
    assert status["should_stop"]
    assert "inside the noise" in status["reason"]


def test_the_incumbent_is_the_best_score_not_the_last_one():
    """A creeping series cannot be laundered into progress by resetting the
    comparison point every time."""
    tables = [row("t1", 0.30, se=0.04)] + [
        row(f"t{i + 2}", 0.30 - (i + 1) * 0.001, se=0.04) for i in range(3)
    ]
    assert evaluate(tables, "logloss", POLICY)["since_improvement"] == 3


def test_a_sub_noise_gain_is_reported_without_being_refused():
    """The advisory half of 0028, and the distinction it rests on.

    The enforced bar is a *quarter* of a standard error; this warns at a whole
    one. So a table can be credited as progress and still be told its gain is
    inside the noise — which is exactly 0022's case, where 0.008 on a 0.012
    standard error turned out to be worth 0.0007.
    """
    tables = [row("t1", 0.12162, se=0.0122), row("t2", 0.11354, se=0.0122)]
    status = evaluate(tables, "rmsle", POLICY)

    # Credited: 0.008 clears 0.25 x 0.0122. Nothing is refused, nothing stops.
    assert status["since_improvement"] == 0
    assert not status["should_stop"]

    warned = status["latest_gain_inside_one_se"]
    assert warned["id"] == "t2" and round(warned["gain"], 5) == 0.00808
    assert "inside one standard error" in advisory(status)


def test_nothing_is_said_when_there_is_nothing_to_say():
    """A gain bigger than the noise, and a result that did not improve at all,
    must both stay quiet — a warning on every row is a warning on none."""
    big = [row("t1", 0.40, se=0.01), row("t2", 0.20, se=0.01)]
    assert evaluate(big, "rmsle", POLICY)["latest_gain_inside_one_se"] is None

    worse = [row("t1", 0.20, se=0.01), row("t2", 0.21, se=0.01)]
    assert evaluate(worse, "rmsle", POLICY)["latest_gain_inside_one_se"] is None

    no_se = [row("t1", 0.20, se=None), row("t2", 0.199, se=None)]
    assert evaluate(no_se, "rmsle", POLICY)["latest_gain_inside_one_se"] is None
    assert "standard error" not in advisory(evaluate(no_se, "rmsle", POLICY))


def test_a_missing_standard_error_floors_the_bar_rather_than_raising_it():
    """Ordered splits have no `probe_se` — a bootstrap over a held-out *period*
    would confidently understate what actually varies. Zero noise makes every
    positive gain count, which spends experiments rather than ending a run early
    on a number we cannot compute."""
    from tabular_agent.harness.stop import _noise

    assert _noise({"probe_se": 0.03}) == 0.03
    assert _noise({"probe_se": None}) == 0.0
    assert _noise({}) == 0.0

    tables = [row("t1", 0.300, se=None), row("t2", 0.2999, se=None)]
    assert evaluate(tables, "logloss", POLICY)["since_improvement"] == 0


def test_a_worse_table_costs_patience_but_is_not_fatal():
    """A run branches: trying something that turns out worse is the search working."""
    tables = [row("t1", 0.30), row("t2", 0.45), row("t3", 0.20)]
    status = evaluate(tables, "logloss", POLICY)
    assert status["since_improvement"] == 0  # t3 was a real improvement
    assert not status["should_stop"]


def test_higher_is_better_metrics_are_read_the_right_way_round():
    tables = [row("t1", 0.70), row("t2", 0.85)]
    assert evaluate(tables, "auc", POLICY)["since_improvement"] == 0
    assert evaluate(tables, "logloss", POLICY)["since_improvement"] == 1


def test_the_table_ceiling_stops_a_run_that_keeps_improving():
    policy = StopPolicy(patience=99, max_tables=4)
    tables = [row(f"t{i}", 0.5 - i * 0.1) for i in range(4)]
    status = evaluate(tables, "logloss", policy)
    assert status["should_stop"]
    assert "budget" in status["reason"]
    assert status["tables_remaining"] == 0


def test_a_long_search_is_allowed_but_not_an_endless_one():
    """Eight ideas that did not work, to find the ninth that does, is the job —
    but with one counter there is a limit, where the old two-level rule let a run
    alternate between levels forever."""
    tables = [row("t1", 0.40)] + [row(f"t{i + 2}", 0.41 + i * 0.01) for i in range(5)]
    status = evaluate(tables, "logloss", StopPolicy())
    assert status["since_improvement"] == 5
    assert not status["should_stop"]  # default patience is 6

    tables.append(row("t8", 0.47))
    assert evaluate(tables, "logloss", StopPolicy())["should_stop"]


def test_a_no_op_table_no_longer_buys_anything():
    """0034's footnote: a table that only copied the frame used to reset the
    per-table counter and buy the run more experiments. With one counter over
    tables it is simply a table that did not improve."""
    same = 0.30
    tables = [row("t1", same)] + [row(f"t{i + 2}", same) for i in range(3)]
    status = evaluate(tables, "logloss", POLICY)
    assert status["since_improvement"] == 3
    assert status["should_stop"]


def interval(gain, spread, mode="ordered_blocks"):
    key = "block_sd" if mode == "ordered_blocks" else "paired_se"
    return {"gain": gain, key: spread, "mode": mode}


def test_the_terminal_table_is_best_among_demonstrated_gains():
    tables = [
        {**row("t1", 0.30), "probe_vs_source_interval": interval(0.02, 0.01)},
        {**row("t2", 0.20), "probe_vs_source_interval": interval(0.03, 0.01)},
        {**row("t3", 0.25), "probe_vs_source_interval": interval(0.001, 0.01)},
    ]
    assert best_table(tables, "logloss")["id"] == "t2"
    assert best_table(tables, "auc")["id"] == "t1"
    assert best_table([], "auc") is None
    # A table the ruler could not score cannot win by default.
    assert best_table([{"id": "t9", "probe_score": None}], "auc") is None


def test_terminal_gate_rejects_noise_no_ops_and_missing_intervals():
    tables = [
        {**row("noise", 0.10), "probe_vs_source_interval": interval(0.009, 0.01)},
        {**row("noop", 0.09), "probe_vs_source_interval": interval(0.0, 0.0)},
        row("missing", 0.08),
    ]
    assert best_table(tables, "logloss") is None


def test_random_split_uses_the_paired_standard_error():
    tables = [
        {**row("weak", 0.10), "probe_vs_source_interval": interval(0.01, 0.02, "paired_bootstrap")},
        {**row("strong", 0.20), "probe_vs_source_interval": interval(0.03, 0.02, "paired_bootstrap")},
    ]
    assert best_table(tables, "auc")["id"] == "strong"


def test_search_guidance_remains_the_ungated_probe_argmax():
    tables = [row("t1", 0.30), row("t2", 0.20), row("t3", 0.25)]
    assert best_probe_table(tables, "logloss")["id"] == "t2"


def test_ties_go_to_the_earlier_table():
    """Fewer steps, so less lineage to go wrong at predict time."""
    tables = [
        {**row("t1", 0.30), "probe_vs_source_interval": interval(0.02, 0.01)},
        {**row("t2", 0.30), "probe_vs_source_interval": interval(0.02, 0.01)},
    ]
    assert best_table(tables, "logloss")["id"] == "t1"
