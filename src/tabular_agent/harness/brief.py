"""Where a fresh agent wakes up: the run's state, rendered from the journal.

An episode is a fresh agent on the same ``run_dir`` (0024), which means it wakes
up knowing nothing — not the task, not the metric, not that eleven experiments
have already happened. It can find all of that with ``explore_journal``, and the
live runs showed it does so unprompted. The problem is what that costs:
orientation ate 26-50% of every episode at both caps tested (0025), and one
episode spent seven calls orienting and trained nothing.

So this is the other half. The brief is **rendered, never accumulated**: it is a
function of the journal as it stands right now, so it costs no tool calls, and it
is the same size at experiment 150 as at experiment 2.

That last property is the design constraint, and it is easy to break. Every
addition here must be O(1) in the length of the run. A "recent experiments"
section would be the natural thing to add and is exactly what must not go in:
history is *queried*, not recited, or this becomes the context growth it exists
to prevent. The one list allowed (:data:`MAX_LISTED_TABLES`) is capped and says
so when it truncates.

What belongs here rather than in ``SYSTEM_PROMPT``: anything that varies with the
state of *this run at this moment*. The system prompt is the rules of the game
and is identical for every run; this is the position on the board.
"""

from ..metrics import greater_is_better
from . import journal, stop
from .session import RunSession

#: Never-trained tables named before the list truncates. Capped because the brief
#: must not grow with the run: an agent that builds ten tables and trains none
#: would otherwise turn its own idle work into a permanent context cost.
MAX_LISTED_TABLES = 3


def _fmt(x, places=4):
    return "unknown" if x is None else f"{x:.{places}f}"


def _never_trained(j: dict) -> list[str]:
    """Tables that exist and have no experiment on them.

    The most useful thing the brief can say that is not a number: an episode cut
    mid-idea leaves exactly this trace (0024 calls episode length "a seam, not a
    count"), and the next agent has no memory of having meant to train it.
    """
    trained = {r.get("source_table") for r in j.get("leaderboard") or []}
    return [t["path"] for t in j.get("tables") or [] if t.get("path") not in trained]


def render_brief(session: RunSession) -> str | None:
    """The run's current position, or ``None`` when nothing has happened yet.

    ``None`` on a fresh run is deliberate rather than an empty-state message: a
    run that has done nothing has no position to report, and a brief announcing
    "0 experiments so far" would spend tokens telling the agent something the
    absence of the brief already says. It also keeps episode 1 byte-identical to
    a run with no episodes at all, which is what makes the two comparable.
    """
    j = journal.load(session.run_dir)
    task = j.get("task")
    lb = j.get("leaderboard") or []
    tables = j.get("tables") or []

    if not task or (not lb and not tables):
        return None

    metric = task.get("metric")
    direction = "higher is better" if greater_is_better(metric) else "lower is better"
    status = stop.evaluate(lb, metric, session.stop_policy)

    # The heading earns its tokens by saying where this came from: an agent told
    # facts about a run it does not remember should know they were read off disk
    # and can be read again, not handed down.
    lines = [
        "RUN STATE — rendered from the journal, not remembered.",
        f"Task: {task.get('task_type')} on {task.get('target')!r}, "
        f"metric {metric} ({direction}).",
    ]

    split = j.get("split") or {}
    if split:
        lines.append(
            f"Split: {split.get('n_work')} rows you work on, "
            f"{split.get('n_val')} held back for val_score ({split.get('mode')})."
        )

    lines.append(
        f"Done so far: {len(lb)} experiments, {len(tables)} feature tables built."
    )

    baseline, best = j.get("baseline"), j.get("best")
    if baseline:
        lines.append(
            f"Baseline: {baseline['experiment_id']} at val_score "
            f"{_fmt(baseline.get('val_score'))}."
        )
    if best:
        # `best` is a pointer, not a row — the table it was trained on lives on
        # the leaderboard row, and it is the half of "best" that is actionable:
        # the next idea is usually built on top of whatever is currently winning.
        row = next((r for r in lb if r.get("id") == best["experiment_id"]), {})
        lines.append(
            f"Best so far: {best['experiment_id']} at val_score "
            f"{_fmt(best.get('val_score'))}, trained on "
            f"{row.get('source_table') or 'the work table'}."
        )

    # The stop state is the one thing an agent cannot infer and will act wrongly
    # on: an agent that thinks it has all night keeps tuning, and an agent that
    # does not know it is nearly out of budget spends the rest of it orienting.
    if status["should_stop"]:
        lines.append(f"The harness is stopping this run: {status['reason']}. Call finalize_run.")
    else:
        lines.append(
            f"Budget: {status['experiments_remaining']} experiments left of "
            f"{session.stop_policy.max_experiments}; "
            f"{status['since_improvement']} since the last real improvement "
            f"(the run ends at {status['patience']})."
        )

    idle = _never_trained(j)
    if idle:
        shown = ", ".join(idle[:MAX_LISTED_TABLES])
        more = len(idle) - MAX_LISTED_TABLES
        lines.append(
            f"Built but never trained on: {shown}"
            + (f", and {more} more" if more > 0 else "")
            + " — an earlier episode may have stopped mid-idea."
        )

    if j.get("stop", {}).get("finalized"):
        lines.append("This run is finalized. The bundle and report are written.")

    # The line that makes the brief a *starting point* rather than the whole
    # record. Without it the obvious reading of a short state summary is that it
    # is all there is, and the agent re-learns by experiment what it could ask.
    lines.append(
        "Anything else — which methods paid, what a table holds, what was already "
        "tried on it — is in the journal. Query it rather than spending "
        "experiments to find out again."
    )
    return "\n".join(lines)
