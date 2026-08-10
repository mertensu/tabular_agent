"""Context growth over a run — the baseline any context-saving change is judged against.

The claim behind 0003 and 0020 is that the store is the state and the context is
a *view* of it. That claim is measurable, and this module is how: every model
response records the ``input_tokens`` its request carried, which is exactly the
size of the context at that turn. Plot it against turn number and the shape of
the run's memory is visible.

Two numbers matter and they answer different questions:

* **peak** — the largest context any single request carried. This is what a run
  dies of: exceed the model's window and the run ends, whatever the cost so far.
* **total** — the sum over requests, i.e. what was actually paid for. A run whose
  context grows linearly pays quadratically in the number of turns, which is the
  cost that makes 150 experiments unaffordable rather than merely impossible.

`slope_per_turn` is the average growth per request. It is the number that should
*fall* when history moves out of the context and into a queryable store: the
prediction is that it goes from roughly the size of a tool result to roughly
zero, with `peak` flattening rather than climbing.

Kept as its own module rather than folded into the runner because the comparison
is between runs made weeks apart, and a measurement that is redefined between
its two readings measures nothing.
"""

import json

__all__ = ["context_profile", "format_profile"]


def context_profile(messages: list[dict], experiments: int | None = None) -> dict:
    """Per-request context sizes plus the summary, from a saved transcript.

    ``messages`` is the decoded ``all_messages_json()`` of a run. Requests with
    no usage recorded are skipped rather than counted as zero — a provider that
    omits the field should leave a gap, not a dip.
    """
    turns = []
    for m in messages:
        if m.get("kind") != "response":
            continue
        usage = m.get("usage") or {}
        if not usage.get("input_tokens"):
            continue
        turns.append({
            "turn": len(turns) + 1,
            "input_tokens": int(usage["input_tokens"]),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cache_read_tokens": int(usage.get("cache_read_tokens") or 0),
        })

    if not turns:
        return {"turns": [], "n_requests": 0}

    inputs = [t["input_tokens"] for t in turns]
    peak = max(inputs)
    total = sum(inputs)
    # First to last over the gaps between them: the average cost of one more turn.
    slope = (inputs[-1] - inputs[0]) / (len(inputs) - 1) if len(inputs) > 1 else 0.0

    summary = {
        "n_requests": len(turns),
        "first_input_tokens": inputs[0],
        "peak_input_tokens": peak,
        "final_input_tokens": inputs[-1],
        "total_input_tokens": total,
        "total_output_tokens": sum(t["output_tokens"] for t in turns),
        "cache_read_tokens": sum(t["cache_read_tokens"] for t in turns),
        "slope_per_turn": round(slope, 1),
    }
    if experiments:
        # The unit that actually matters for planning an overnight run: a budget
        # is in experiments, not in turns.
        summary["input_tokens_per_experiment"] = round(total / experiments, 1)
        summary["peak_per_experiment"] = round(peak / experiments, 1)
    return {"turns": turns, **summary}


def format_profile(profile: dict, label: str = "") -> str:
    """One-screen summary, for a notebook or a run log."""
    if not profile.get("n_requests"):
        return f"{label}: no usage recorded"
    lines = [
        f"{label}  ({profile['n_requests']} requests)",
        f"  context   {profile['first_input_tokens']:>8,} -> {profile['final_input_tokens']:>8,} tokens"
        f"   (peak {profile['peak_input_tokens']:,})",
        f"  growth    {profile['slope_per_turn']:>8,.0f} tokens/turn",
        f"  paid      {profile['total_input_tokens']:>8,} input"
        f"  + {profile['total_output_tokens']:,} output",
    ]
    if "input_tokens_per_experiment" in profile:
        lines.append(f"  per exp   {profile['input_tokens_per_experiment']:>8,.0f} input tokens")
    return "\n".join(lines)


def profile_run(run_dir: str) -> dict:
    """Read ``transcript.json`` + the journal from a run directory."""
    import os

    with open(os.path.join(run_dir, "transcript.json")) as f:
        messages = json.load(f)
    n_exp = None
    journal_path = os.path.join(run_dir, "run_journal.json")
    if os.path.exists(journal_path):
        with open(journal_path) as f:
            n_exp = len(json.load(f).get("leaderboard") or [])
    return context_profile(messages, experiments=n_exp)
