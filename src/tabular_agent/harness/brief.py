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
    """Tables that exist and could not be scored.

    The most useful thing the brief can say that is not a number: since 0035
    every table is scored as it is written, so a table without a number is one
    the ruler failed on — an episode cut mid-idea, or a lineage that could not be
    replayed. The next agent has no memory of either.
    """
    return [
        t["path"] for t in j.get("tables") or [] if t.get("probe_score") is None
    ]


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
    status = stop.evaluate(tables, metric, session.stop_policy)

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

    lines.append(f"Done so far: {len(tables)} feature tables built.")

    baseline, best = j.get("baseline"), j.get("best")
    if baseline:
        lines.append(
            f"Baseline: {baseline['experiment_id']} at val_score "
            f"{_fmt(baseline.get('val_score'))}."
        )
    # The best *table*, which is the actionable half: the next idea is usually
    # built on top of whatever is currently winning. `best` on the journal points
    # at an experiment and only exists after finalize, so it is not what a
    # mid-run agent needs.
    winner = stop.best_table(tables, metric)
    if winner:
        lines.append(
            f"Best table so far: {winner['path']} at probe_score "
            f"{_fmt(winner.get('probe_score'))}."
        )

    # The stop state is the one thing an agent cannot infer and will act wrongly
    # on: an agent that thinks it has all night keeps tuning, and an agent that
    # does not know it is nearly out of budget spends the rest of it orienting.
    if status["should_stop"]:
        lines.append(f"The harness is stopping this run: {status['reason']}. Call finalize_run.")
    else:
        lines.append(
            f"Budget: {status['tables_remaining']} tables left of "
            f"{session.stop_policy.max_tables}; "
            f"{status['since_improvement']} since the last real improvement "
            f"(the run ends at {status['patience']})."
        )

    # Read off the raw spec rather than `session.resolved()`, which would parse
    # the source table to detect the task type — a file read at the top of every
    # episode, on a dataset that may be hundreds of megabytes.
    spec = session.spec
    if spec.notes or spec.column_notes:
        # A pointer, not the dictionary itself: it is O(columns) and the brief is
        # O(1) by construction (0026). Worth the line because the prompt now tells
        # a resumed agent to *skip* `inspect_data`, so without this the data's own
        # documentation is unreachable by an agent that was not there for it.
        lines.append(
            "This dataset came with documentation — per-column descriptions and "
            "what the data is — which `inspect_data` returns and nothing else has."
        )

    # Read off the session, not the journal: a stream is declared on the run and
    # is true from the first call, where the journal only learns about it once
    # `inspect_data` has run. An episode that skips `inspect_data` — which the
    # prompt tells a resumed agent to do — would otherwise never hear that the
    # run has a second input at all, and no other tool mentions one.
    if session.streams:
        names = ", ".join(session.stream_names())
        lines.append(
            f"Second inputs declared: {names}. These are streams, not tables — no "
            "tool returns rows from them. Reach one with fit_stream_aggregate; "
            "`inspect_data` describes what each holds."
        )

    # Read off the session like streams and for the same reason: future_known is
    # declared on the run and true from the first call, but no tool other than
    # `inspect_data` mentions it, and the prompt tells a resumed agent to skip
    # that. Without this line an episode never learns the run has forward-legal
    # covariates at all (0090).
    if getattr(session, "future_known", None):
        names = ", ".join(session.future_known)
        lines.append(
            f"Future-known covariates declared: {names}. Their horizon values are "
            "known at issue time, so aggregate them forward with "
            "fit_future_covariate_window (the exogenous counterpart to fit_history)."
        )

    idle = _never_trained(j)
    if idle:
        shown = ", ".join(idle[:MAX_LISTED_TABLES])
        more = len(idle) - MAX_LISTED_TABLES
        lines.append(
            f"Built but never scored: {shown}"
            + (f", and {more} more" if more > 0 else "")
            + " — the ruler could not measure them, so they are not in the ranking."
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
