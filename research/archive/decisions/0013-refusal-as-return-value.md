# 0013 — A refusal is a return value, not an exception

- **Date:** 2026-07-30
- **Status:** accepted
- **Commit:** `6936d4b`
- **Related:** [0007](0007-retryable-arguments.md), [0012](0012-harness-owns-the-stop.md)

## The problem

Decision 0012 has the harness refuse to train once the run has stopped making
progress. How should a tool say no?

The obvious answer is to raise. Decision 0007 already built that path: raise
`ValueError`, it becomes a `ModelRetry`, the model gets told and tries something
else.

Using it here would be a mistake, for a reason worth stating precisely.

## Background: the retry budget is a scarce resource

`TOOL_RETRIES = 3`. Exhausting it ends the run — and ending the run discards every
completed experiment (0007). That budget exists for *malformed arguments*: a
mistyped column, an invented path, an undeclared ordinal level.

Being asked to train one more time is not a malformed argument. It is a model
that has not yet read the news. If each such request consumed a retry, a model
that asked three times — entirely plausible, since nothing about "you may not
train" tells it what *to* do unless it reads carefully — would destroy the run it
was in the middle of finishing successfully.

The severity is inverted: the *most* recoverable situation would trigger the
*least* recoverable outcome.

## The decision

**A refusal is a normal, successful return value.**

```json
{
  "trained": false,
  "reason": "the last 4 experiments produced no real improvement ...",
  "next": "finalize_run",
  "note": "..."
}
```

No exception, no retry consumed, no cost. The model reads it and acts.

Structurally it is still a refusal — nothing was trained, no leaderboard row was
appended, no artifacts written. The agent is simply not punished for asking.

### The escalation

A refusal only works on a model that takes it. One that keeps calling
`train_tabular_model` would loop until the context died — and the run would end
with **no bundle**, which is the one outcome that makes the whole run worthless.

So the response escalates by what the repetition means:

| call | response | why |
|---|---|---|
| the table is used up | refuse that table, name the next move | it just needs telling; the run is fine |
| the same dead table again | escalate to the run level | this is a loop, not a redirect |
| the `max_refusals`th | **the harness finalises the run itself** and returns the bundle | the deliverable matters more than being obeyed |
| after that | `RuntimeError` | the run is over; there is nothing left to protect |

The third row is the important one. When the harness gives up on being asked
nicely, it does not simply stop — it packages the run, because a loop that ends
without a bundle has thrown away everything the run did.

## Alternatives considered

**Raise `ModelRetry`.** Rejected above.

**Raise a fatal exception immediately.** Ends the run without a bundle. Same
objection, sooner.

**Silently return a fake success.** Never seriously considered, but worth
recording as rejected: the leaderboard would gain a row for an experiment that
did not happen, breaking 0004's guarantee that the record cannot drift.

## Consequences

**Good**

- Refusals are free, so the harness can afford to refuse readily — which is what
  makes the per-table redirect in 0014 viable at all.
- The run always ends with a deliverable.

**Bad**

- `train_tabular_model` now has two return shapes. Callers must check
  `trained`, and a model that ignores the field will misread a refusal as a
  result. Mitigated by putting `trained: false` first and explaining it in the
  tool docstring, but it is a genuine wart.
- The auto-finalise path means the harness sometimes takes an action the agent
  did not request. Defensible — it is the action the agent was *told* to take —
  but it is the one place the harness stops being purely reactive.
- One special case had to be carved out: `finalize_run` auto-trains a baseline if
  the run never did, and refusing *that* would leave the report unable to answer
  "did this beat doing nothing". Hence `enforce_stop=False` on that internal call.

## Where it lives in the code

- `src/tabular_agent/harness/tools.py` — `_refuse_if_finished`,
  `_refuse_if_table_exhausted`, and the `enforce_stop` parameter.
- `tests/test_harness_tools.py` — asserts a refusal appends no leaderboard row,
  costs no retry, and that repetition ends with a bundle on disk.
