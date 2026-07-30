"""When a run is done (spec §19.3).

An agent left to itself does not stop. The first live run trained eleven models;
the last six moved the score by about 0.003 while the fold spread was 0.04 — six
experiments spent re-rolling a die. Nothing in the loop objected, because nothing
in the loop had an opinion about *done*.

Two layers, the same split as everywhere else in this harness:

* **Advisory** — every training result carries how many experiments have passed
  without a real improvement. A cooperative model reads that and finalizes.
* **Structural** — past the patience budget the harness stops accepting training
  calls at all. Guidance for the cooperative case, structure for the rest.

The interesting part is what counts as an improvement. Not an absolute
threshold — that would need tuning per dataset and per metric, and getting it
wrong is silent. The yardstick is already in the leaderboard: ``fold_spread``,
how much the incumbent's own score moves between folds. A gain far below that is
not a smaller gain, it is noise, and treating it as progress is how a run talks
itself into another six experiments.

The second interesting part is *which* failures should count against a run at
all. Trying eight ideas that do not work in order to find the ninth that does is
not a failing search — it is the only kind of search there is, and a rule that
punishes it is worse than no rule. But those eight experiments have to be eight
different *ideas*. Six hyper-parameter settings on one feature table are not a
search; they are the same experiment costed six times.

So the run-level budget (:attr:`StopPolicy.patience`) is generous, and the sharp
constraint is per feature table (:attr:`StopPolicy.table_patience`), enforced by
declining *that table* rather than the run. A refused table redirects the agent
back to features and costs it nothing; a refused run ends it. Only one of those
deserves a hair trigger.
"""

from dataclasses import dataclass

from ..metrics import greater_is_better


@dataclass(frozen=True)
class StopPolicy:
    """Knobs for "is this run finished". Defaults are deliberately generous."""

    #: Experiments allowed to pass with no real improvement before the harness
    #: ends the run. This is the *exploration* budget and it is deliberately
    #: large: trying eight things that do not work in order to find the ninth
    #: that does is not a failing search, it is the only kind there is. The
    #: cheaper corrective is :attr:`table_patience` below, which redirects a run
    #: without ending it.
    patience: int = 10

    #: A gain counts only if it exceeds this fraction of the incumbent's fold
    #: spread. At 0.25, beating the best model by a quarter of its own
    #: fold-to-fold wobble is the bar for "something happened".
    min_gain_ratio: float = 0.25

    #: Experiments on **one feature table** with no improvement before the
    #: harness declines that table specifically. Tight, because experiments on a
    #: fixed table can only differ by model and hyper-parameters, and that is the
    #: low-information move: it costs the same as a feature experiment and tells
    #: you far less. Refusing the table is a *redirect*, not a stop — the run
    #: continues the moment it builds new features.
    table_patience: int = 3

    #: Hard ceiling on experiments, whatever the scores are doing. Backstop
    #: against a run that keeps clearing the bar by luck, and against cost.
    max_experiments: int = 25

    #: Training calls refused before the harness gives up on being asked nicely,
    #: packages the run itself and closes it.
    max_refusals: int = 3


def _attempts(leaderboard: list[dict]) -> list[dict]:
    """Scored, non-baseline rows: the experiments that were actually attempts.

    The baseline is mandatory and is not a try at doing better; charging it
    against any budget would punish a run for following the workflow.
    """
    return [
        r
        for r in leaderboard
        if r.get("cv_score") is not None and r.get("model_type") != "baseline"
    ]


def _since_improvement(rows: list[dict], higher_better: bool, min_gain_ratio: float) -> int:
    """How many of ``rows`` have passed since one of them was a real improvement."""
    incumbent = None
    incumbent_spread = 0.0
    last_progress = -1
    for i, r in enumerate(rows):
        score = r["cv_score"]
        if incumbent is None:
            progress = True
        else:
            gain = (score - incumbent) if higher_better else (incumbent - score)
            # The incumbent's own fold-to-fold movement is the noise floor. A
            # "gain" underneath it is a coin flip that landed our way.
            progress = gain > min_gain_ratio * max(incumbent_spread, 1e-12)

        better = incumbent is None or (
            score > incumbent if higher_better else score < incumbent
        )
        if better:
            # The incumbent tracks the true best, so a creeping series of
            # sub-noise gains cannot be laundered into progress by resetting the
            # comparison point each time.
            incumbent = score
            incumbent_spread = float(r.get("fold_spread") or 0.0)
        if progress:
            last_progress = i

    return len(rows) - 1 - last_progress if rows else 0


def table_status(
    leaderboard: list[dict],
    metric: str,
    policy: StopPolicy,
    source_table: str,
    label: str | None = None,
) -> dict:
    """Whether one feature table has been squeezed dry.

    Experiments sharing a source table can only differ by model type and
    hyper-parameters. Once several of those in a row have produced nothing, the
    informative move is not another one — it is new features. So this is checked
    separately from :func:`evaluate`, and answered by refusing *the table* rather
    than the run: the search continues the moment the agent builds something new.

    This is where "features beat tuning" stops being advice in a prompt and
    becomes a property of the tool.
    """
    rows = [r for r in _attempts(leaderboard) if r.get("source_table") == source_table]
    since = _since_improvement(rows, greater_is_better(metric), policy.min_gain_ratio)
    exhausted = bool(rows) and since >= policy.table_patience
    return {
        "exhausted": exhausted,
        "source_table": source_table,
        "n_experiments_on_table": len(rows),
        "since_improvement_on_table": since,
        "table_patience": policy.table_patience,
        "reason": (
            f"{since} experiments on {label or source_table} in a row produced no real "
            "improvement, so it is only the model and its hyper-parameters changing "
            "now. That is the cheapest experiment available and the least "
            "informative one"
            if exhausted
            else None
        ),
    }


def evaluate(leaderboard: list[dict], metric: str, policy: StopPolicy) -> dict:
    """Whether the run has stopped making progress, and the numbers behind it.

    Pure function of the leaderboard — no run dir, no side effects — so the stop
    rule can be unit-tested against a hand-written history instead of by training
    twenty models.
    """
    rows = _attempts(leaderboard)
    since_improvement = _since_improvement(
        rows, greater_is_better(metric), policy.min_gain_ratio
    )
    n_experiments = len(leaderboard)

    stalled = bool(rows) and since_improvement >= policy.patience
    exhausted = n_experiments >= policy.max_experiments

    if exhausted:
        reason = (
            f"this run has used its budget of {policy.max_experiments} experiments"
        )
    elif stalled:
        reason = (
            f"the last {since_improvement} experiments produced no real improvement "
            f"(each gain was under {policy.min_gain_ratio:g}x the best model's "
            f"fold spread, i.e. inside the noise)"
        )
    else:
        reason = None

    return {
        "should_stop": stalled or exhausted,
        "reason": reason,
        "n_experiments": n_experiments,
        "since_improvement": since_improvement,
        "patience": policy.patience,
        "experiments_remaining": max(policy.max_experiments - n_experiments, 0),
    }


def advisory(status: dict, table: dict | None = None) -> str:
    """One line for the model, in every training result. The guidance half."""
    if status["should_stop"]:
        return f"The harness is stopping this run: {status['reason']}. Call finalize_run."

    if table is not None and table["exhausted"]:
        note = (
            f"This table is used up: {table['since_improvement_on_table']} experiments "
            "on it without an improvement, so the harness will not train on it again. "
            "Build new features with transform_features or fit_encoding and train on "
            "the table that produces."
        )
    elif table is not None and table["since_improvement_on_table"] > 0:
        left = table["table_patience"] - table["since_improvement_on_table"]
        note = (
            f"{table['since_improvement_on_table']} experiment(s) on this table without "
            f"an improvement; {left} more and the harness will stop training on it. New "
            "features move the score more reliably than new hyper-parameters."
        )
    elif status["since_improvement"] == 0:
        note = "That is the best result so far."
    else:
        note = f"{status['since_improvement']} experiment(s) since the last real improvement."

    return (
        f"{note} {status['experiments_remaining']} experiments remain in this run's "
        f"budget; the run ends after {status['patience']} in a row without an improvement."
    )
