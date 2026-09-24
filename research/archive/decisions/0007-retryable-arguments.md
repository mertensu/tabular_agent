# 0007 — A bad argument is a correction, not the end of the run

- **Date:** 2026-07-29
- **Status:** accepted
- **Commits:** `afffa57`, `283716e`
- **Found by:** running the harness against a real model for the first time

## The problem

The tool layer validates its inputs carefully. Ask it to encode a column that
does not exist, to encode the target as a feature, or to use `method="ordinal"`
without stating the level order, and it raises `ValueError` with a precise
message.

That is correct behaviour for a library called by a programmer. Called by a
model, it was catastrophic: **one malformed argument ended the entire run**, and
with it the eight experiments already completed. Hours of tokens discarded
because of a single mistyped column name.

The failure was not hypothetical. It happened on the first live run.

## Background: what `ModelRetry` is

PydanticAI has an exception type, `ModelRetry`, meaning "this call did not work;
here is why; try again". Raising it feeds the message back to the model as a tool
result, and the model gets another turn to fix its arguments. It costs one round
trip rather than the run.

## The decision

Wrap every tool in a small helper that converts *argument problems* into
`ModelRetry`, and lets everything else propagate:

- `ValueError` — our own input validation. The messages are already written to
  say what was wrong and what to pass instead, so they double as the correction.
- `FileNotFoundError` — a path the model invented.
- `SandboxError` — code the model wrote that failed, timed out, or imported a
  blocked library (added later, with 0011). The argument here happens to be a
  program, and those messages are written to teach.

Anything else — `KeyError`, `TypeError`, anything unexpected — is a bug in *our*
pipeline. Those still stop the run loudly, because handing a model a stack trace
from our own broken code and asking it to guess is not a recovery strategy, it is
a way to get plausible nonsense.

The validation messages are therefore written for two readers at once: a human
debugging, and a model correcting itself in one turn. That changed how they are
phrased — `"no such table: 'tables/t9.parquet'. Available: None (the raw input
table), 'tables/t1.parquet'"` is more useful to both than "file not found".

## A second, separate bug

Fixing the above immediately exposed the next one: PydanticAI allows **one**
retry per tool by default. The corrected run died anyway, on the second bad
argument, having burned its single retry on the first.

`TOOL_RETRIES = 3` on the agent. Bounded on purpose — a model looping on the same
rejected argument should stop rather than spend the budget — but three is a very
different thing from one in a run that makes dozens of calls.

## Alternatives considered

**Return errors as ordinary values from every tool.** More uniform, and it is
what we ended up doing for *refusals* later (0013). Rejected here because a bad
argument genuinely is an error: the tool did nothing, and the retry mechanism
already exists to handle exactly this.

**Retry everything.** Rejected: it hides our bugs behind a model politely trying
something else, and the run ends with a plausible summary built on a broken
pipeline.

## Consequences

**Good**

- A run survives its own mistakes. Live runs since have completed with zero
  fatal errors while making several corrections along the way.
- Validation messages got better because they acquired a second audience.

**Bad**

- The line between "argument problem" and "our bug" is a judgement call encoded
  as a tuple of exception types. A `ValueError` raised from deep inside pandas,
  rather than from our validation, would be misclassified as retryable and handed
  to the model as if it were its fault.
- Three retries per tool is a number with no principle behind it.

## Where it lives in the code

- `src/tabular_agent/harness/agent.py` — `_retryable`, and `TOOL_RETRIES`.
- `tests/test_harness_agent.py` — asserts a bad argument becomes `ModelRetry`
  and that a genuine bug (`KeyError`) still escapes.
