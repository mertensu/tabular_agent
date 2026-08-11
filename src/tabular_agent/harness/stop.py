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
wrong is silent. The yardstick is the incumbent's own uncertainty: ``val_se``,
the standard error of the validation score, obtained by resampling the rows it
was measured on. A gain far below that is not a smaller gain, it is noise, and
treating it as progress is how a run talks itself into another six experiments.

This was ``fold_spread`` until 0022 — the fold-to-fold wobble of ``cv_score``,
which is a sound estimate for a quantity nothing has been ranked by since 0019.
The two were numerically close on the dataset where it hid, and the bar is a
*quarter* of the yardstick, so the bar came out four times smaller than the noise
in the number it gated.

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
        if r.get("val_score") is not None and r.get("model_type") != "baseline"
    ]


def _noise(row: dict) -> float:
    """The incumbent's own uncertainty, in the units of the number being compared.

    `val_se` is the standard error of `val_score` — the quantity every comparison
    here is made on since 0019 — obtained by resampling the validation rows the
    score was computed from. It is the right ruler, and 0022 is the record of
    what went wrong while it was the wrong one: on house-prices the bar was
    `0.25 x fold_spread = 0.0025` while the noise in the compared number was
    `0.0098`, so every sub-noise gain passed and was recorded as progress.

    `fold_spread` is the fallback, not the default: it measures how much
    `cv_score` moved between folds, which is a fine estimate of the wrong
    quantity. It is used only when `val_se` is absent — a validation score that
    could not be computed, a validation half too small to resample, or a
    leaderboard row written before `val_se` existed.
    """
    se = row.get("val_se")
    if se is not None:
        return float(se)
    return float(row.get("fold_spread") or 0.0)


def _since_improvement(rows: list[dict], higher_better: bool, min_gain_ratio: float) -> int:
    """How many of ``rows`` have passed since one of them was a real improvement."""
    incumbent = None
    incumbent_noise = 0.0
    last_progress = -1
    for i, r in enumerate(rows):
        score = r["val_score"]
        if incumbent is None:
            progress = True
        else:
            gain = (score - incumbent) if higher_better else (incumbent - score)
            # The incumbent's own uncertainty is the noise floor. A "gain"
            # underneath it is a coin flip that landed our way.
            progress = gain > min_gain_ratio * max(incumbent_noise, 1e-12)

        better = incumbent is None or (
            score > incumbent if higher_better else score < incumbent
        )
        if better:
            # The incumbent tracks the true best, so a creeping series of
            # sub-noise gains cannot be laundered into progress by resetting the
            # comparison point each time.
            incumbent = score
            incumbent_noise = _noise(r)
        if progress:
            last_progress = i

    return len(rows) - 1 - last_progress if rows else 0


def _latest_gain_inside_one_se(rows: list[dict], higher_better: bool) -> dict | None:
    """Did the newest experiment beat the incumbent by less than one standard error?

    **Advisory, and deliberately not wired to anything that refuses.** The bar the
    stop rule enforces is ``min_gain_ratio`` (0.25) times the yardstick, i.e. a
    *quarter* of a standard error. This reports the other threshold — a whole one
    — which is what 0022's prose actually argues for and what the arithmetic in
    0028 shows the enforced bar does not catch.

    Which of the two should gate the run is an open question that needs a second
    dataset. Until then the agent is *told* and decides, because telling costs
    nothing and refuses nothing — and whether it listens is itself the evidence
    for which way to settle it.

    ``None`` unless there is something to say: no gain, a gain larger than the
    noise, or no `val_se` to compare against.
    """
    if len(rows) < 2:
        return None
    latest = rows[-1]
    se = latest.get("val_se")
    if se is None or latest.get("val_score") is None:
        return None

    earlier = [r["val_score"] for r in rows[:-1] if r.get("val_score") is not None]
    if not earlier:
        return None
    incumbent = max(earlier) if higher_better else min(earlier)

    score = latest["val_score"]
    gain = (score - incumbent) if higher_better else (incumbent - score)
    if gain <= 0 or gain > se:
        return None
    return {"id": latest.get("id"), "gain": float(gain), "val_se": float(se)}


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
    inside_noise = _latest_gain_inside_one_se(rows, greater_is_better(metric))

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
        # Advisory only: this does not gate anything. See `_latest_gain_inside_one_se`.
        "latest_gain_inside_one_se": inside_noise,
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

    line = (
        f"{note} {status['experiments_remaining']} experiments remain in this run's "
        f"budget; the run ends after {status['patience']} in a row without an improvement."
    )

    # Said last so it is the thing most recently read, and phrased as a fact about
    # the measurement rather than an instruction: the run is not being stopped and
    # nothing is being refused. What to do about it is the reader's call.
    noise = status.get("latest_gain_inside_one_se")
    if noise:
        line += (
            f" Note that this beat the previous best by {noise['gain']:.4f}, which is "
            f"inside one standard error of the validation score ({noise['val_se']:.4f}) "
            "— a gain that size is as likely to be the split as the model."
        )
    return line
