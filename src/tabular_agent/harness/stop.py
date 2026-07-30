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
"""

from dataclasses import dataclass

from ..metrics import greater_is_better


@dataclass(frozen=True)
class StopPolicy:
    """Knobs for "is this run finished". Defaults are deliberately generous."""

    #: Experiments allowed to pass with no real improvement before the harness
    #: refuses to train. This is *patience*, exactly as in early stopping, and it
    #: is >1 for the same reason: a run branches. Trying an encoder that turns out
    #: worse, then backing up and trying another, is the search working — not the
    #: search failing.
    patience: int = 4

    #: A gain counts only if it exceeds this fraction of the incumbent's fold
    #: spread. At 0.25, beating the best model by a quarter of its own
    #: fold-to-fold wobble is the bar for "something happened".
    min_gain_ratio: float = 0.25

    #: Hard ceiling on experiments, whatever the scores are doing. Backstop
    #: against a run that keeps clearing the bar by luck, and against cost.
    max_experiments: int = 25

    #: Training calls refused before the harness gives up on being asked nicely,
    #: packages the run itself and closes it.
    max_refusals: int = 3


def evaluate(leaderboard: list[dict], metric: str, policy: StopPolicy) -> dict:
    """Whether the run has stopped making progress, and the numbers behind it.

    Pure function of the leaderboard — no run dir, no side effects — so the stop
    rule can be unit-tested against a hand-written history instead of by training
    twenty models.

    Baseline rows are excluded from the patience count. The baseline is mandatory
    and is not an attempt to do better; charging it against the budget would
    punish a run for following the workflow.
    """
    rows = [
        r
        for r in leaderboard
        if r.get("cv_score") is not None and r.get("model_type") != "baseline"
    ]
    higher_better = greater_is_better(metric)

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
            progress = gain > policy.min_gain_ratio * max(incumbent_spread, 1e-12)

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

    since_improvement = len(rows) - 1 - last_progress if rows else 0
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


def advisory(status: dict) -> str:
    """One line for the model, in every training result. The guidance half."""
    if status["should_stop"]:
        return f"The harness is stopping this run: {status['reason']}. Call finalize_run."
    left = status["patience"] - status["since_improvement"]
    if status["since_improvement"] == 0:
        return (
            f"That is the best result so far. {status['experiments_remaining']} "
            "experiments remain in this run's budget."
        )
    return (
        f"{status['since_improvement']} experiment(s) since the last real improvement; "
        f"{left} more without one and the harness will stop accepting training calls. "
        "Changing the features is more likely to move the score than another "
        "hyper-parameter."
    )
