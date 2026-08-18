"""When a run is done (spec §19.3).

An agent left to itself does not stop. The first live run trained eleven models;
the last six moved the score by about 0.003 while the fold spread was 0.04 — six
experiments spent re-rolling a die. Nothing in the loop objected, because nothing
in the loop had an opinion about *done*.

Two layers, the same split as everywhere else in this harness:

* **Advisory** — every table result carries how many tables have passed without a
  real improvement. A cooperative model reads that and finalizes.
* **Structural** — past the patience budget the harness stops accepting new
  tables at all. Guidance for the cooperative case, structure for the rest.

The interesting part is what counts as an improvement. Not an absolute
threshold — that would need tuning per dataset and per metric, and getting it
wrong is silent. The yardstick is the incumbent's own uncertainty: the standard
error of the score, obtained by resampling the rows it was measured on. A gain
far below that is not a smaller gain, it is noise, and treating it as progress is
how a run talks itself into another six experiments.

This was ``fold_spread`` until 0022 — the fold-to-fold wobble of ``cv_score``,
which is a sound estimate for a quantity nothing has been ranked by since 0019.
The two were numerically close on the dataset where it hid, and the bar is a
*quarter* of the yardstick, so the bar came out four times smaller than the noise
in the number it gated.

## One level, since 0035

This rule used to have two: a generous run-level budget, and a tight per-table
one that refused *the table* once several experiments in a row on it had changed
only the model and its hyper-parameters. That second level existed because
"features beat tuning" needed to be a property of the tool rather than a line in
a prompt.

There are no model experiments any more. The agent builds tables; the harness
scores each one with a fixed ruler as it is written. So the low-information move
the table budget existed to refuse is not available, and the two levels collapse
into one whose unit is the table — which is also the unit the run's work is
actually done in.

The old defaults are preserved in effect rather than in number. Run-level
patience was 10 *experiments*, and across the 16 runs in `_runs/` 44% of
experiments changed only the model, so ten experiments bought roughly six ideas.
:attr:`StopPolicy.patience` is six for that reason and not because six is round.

One consequence worth naming: a table that adds no columns — the no-op
``df.copy()`` — used to reset the per-table counter and buy the run further
experiments. It was the loophole that produced the best model this harness has
made (0034). With one counter over tables there is nothing to reset: a no-op
table is simply a table that did not improve, and it counts against patience like
any other.
"""

from dataclasses import dataclass

from ..metrics import greater_is_better


@dataclass(frozen=True)
class StopPolicy:
    """Knobs for "is this run finished". Defaults are deliberately generous."""

    #: Tables allowed to pass with no real improvement before the harness ends
    #: the run. Deliberately large: trying eight things that do not work in order
    #: to find the ninth that does is not a failing search, it is the only kind
    #: there is. Six rather than the old ten because the unit changed from
    #: experiment to table — see the module docstring.
    patience: int = 6

    #: A gain counts only if it exceeds this fraction of the incumbent's own
    #: uncertainty. At 0.25, beating the best table by a quarter of its standard
    #: error is the bar for "something happened".
    #:
    #: 0028 left open whether this should be a *whole* standard error, and 0036
    #: proposes a better instrument than either: the paired standard deviation of
    #: the difference between a table and its parent, measured at 0.0024-0.0041
    #: where `val_se` was 0.01116. Neither is settled, so the enforced bar is
    #: unchanged and the advisory below reports the whole-standard-error case.
    min_gain_ratio: float = 0.25

    #: Hard ceiling on tables, whatever the scores are doing. Backstop against a
    #: run that keeps clearing the bar by luck, and against cost. Looser than the
    #: old experiment ceiling because a table now costs one fit rather than
    #: `n_splits + 1`.
    max_tables: int = 30

    #: Tool calls refused before the harness gives up on being asked nicely,
    #: packages the run itself and closes it.
    max_refusals: int = 3


def _attempts(tables: list[dict]) -> list[dict]:
    """Tables the ruler could actually score.

    A table it could not measure is not evidence of anything and is not charged
    against the budget: the agent did the work, and the harness failed to weigh
    it. Charging that against patience would end runs for the harness's own
    shortcomings.
    """
    return [t for t in tables if t.get("probe_score") is not None]


def _noise(row: dict) -> float:
    """The incumbent's own uncertainty, in the units of the number being compared.

    `probe_se` is the standard error of `probe_score`, obtained by resampling the
    validation rows the score was computed from. Zero when the run uses an ordered
    split, where a bootstrap over the held-out period would confidently
    understate the thing that actually varies — which period it is — and a
    too-small yardstick is worse than none. A zero noise floor makes every
    positive gain count, which is the honest failure direction: it spends
    experiments rather than ending a run early on a number we cannot compute.
    """
    se = row.get("probe_se")
    return float(se) if se is not None else 0.0


def _since_improvement(rows: list[dict], higher_better: bool, min_gain_ratio: float) -> int:
    """How many of ``rows`` have passed since one of them was a real improvement."""
    incumbent = None
    incumbent_noise = 0.0
    last_progress = -1
    for i, r in enumerate(rows):
        score = r["probe_score"]
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
    """Did the newest table beat the incumbent by less than one standard error?

    **Advisory, and deliberately not wired to anything that refuses.** The bar the
    stop rule enforces is ``min_gain_ratio`` (0.25) times the yardstick, i.e. a
    *quarter* of a standard error. This reports the other threshold — a whole
    one — which is what 0022's prose actually argues for and what the arithmetic
    in 0028 shows the enforced bar does not catch.

    Which of the two should gate the run is an open question that needs a second
    dataset. Until then the agent is *told* and decides, because telling costs
    nothing and refuses nothing — and whether it listens is itself the evidence
    for which way to settle it.

    ``None`` unless there is something to say.
    """
    if len(rows) < 2:
        return None
    latest = rows[-1]
    se = latest.get("probe_se")
    if se is None or latest.get("probe_score") is None:
        return None

    earlier = [r["probe_score"] for r in rows[:-1] if r.get("probe_score") is not None]
    if not earlier:
        return None
    incumbent = max(earlier) if higher_better else min(earlier)

    score = latest["probe_score"]
    gain = (score - incumbent) if higher_better else (incumbent - score)
    if gain <= 0 or gain > se:
        return None
    return {"id": latest.get("id"), "gain": float(gain), "probe_se": float(se)}


def best_table(tables: list[dict], metric: str) -> dict | None:
    """The table the terminal fit should be run on: highest `probe_score`.

    One ruler over every table, so this is a comparison between tables and
    nothing else — which is the whole reason the ruler is fixed. Ties go to the
    earlier table, which is the one with fewer steps and therefore the one whose
    lineage has less to go wrong at predict time.
    """
    rows = _attempts(tables)
    if not rows:
        return None
    higher = greater_is_better(metric)
    return max(rows, key=lambda r: r["probe_score"] if higher else -r["probe_score"])


def evaluate(tables: list[dict], metric: str, policy: StopPolicy) -> dict:
    """Whether the run has stopped making progress, and the numbers behind it.

    Pure function of the journal's table list — no run dir, no side effects — so
    the stop rule can be unit-tested against a hand-written history instead of by
    building twenty tables.
    """
    rows = _attempts(tables)
    since_improvement = _since_improvement(
        rows, greater_is_better(metric), policy.min_gain_ratio
    )
    n_tables = len(tables)
    inside_noise = _latest_gain_inside_one_se(rows, greater_is_better(metric))

    stalled = bool(rows) and since_improvement >= policy.patience
    exhausted = n_tables >= policy.max_tables

    if exhausted:
        reason = f"this run has used its budget of {policy.max_tables} tables"
    elif stalled:
        reason = (
            f"the last {since_improvement} tables produced no real improvement "
            f"(each gain was under {policy.min_gain_ratio:g}x the best table's "
            f"standard error, i.e. inside the noise)"
        )
    else:
        reason = None

    return {
        "should_stop": stalled or exhausted,
        "reason": reason,
        "n_tables": n_tables,
        "since_improvement": since_improvement,
        "patience": policy.patience,
        "tables_remaining": max(policy.max_tables - n_tables, 0),
        # Advisory only: this does not gate anything. See `_latest_gain_inside_one_se`.
        "latest_gain_inside_one_se": inside_noise,
    }


def advisory(status: dict) -> str:
    """One line for the model, in every table result. The guidance half."""
    if status["should_stop"]:
        return f"The harness is stopping this run: {status['reason']}. Call finalize_run."

    if status["since_improvement"] == 0:
        note = "That is the best table so far."
    else:
        note = f"{status['since_improvement']} table(s) since the last real improvement."

    line = (
        f"{note} {status['tables_remaining']} tables remain in this run's budget; "
        f"the run ends after {status['patience']} in a row without an improvement."
    )

    # Said last so it is the thing most recently read, and phrased as a fact about
    # the measurement rather than an instruction: the run is not being stopped and
    # nothing is being refused. What to do about it is the reader's call.
    noise = status.get("latest_gain_inside_one_se")
    if noise:
        line += (
            f" Note that this beat the previous best by {noise['gain']:.4f}, which is "
            f"inside one standard error of the score ({noise['probe_se']:.4f}) — a gain "
            "that size is as likely to be the split as the columns."
        )
    return line
