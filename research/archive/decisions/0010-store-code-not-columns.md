# 0010 — Store the transform code, not the columns it produced

- **Date:** 2026-07-30 (the mechanism dates from `66cc54a`, 2026-07-25)
- **Status:** accepted
- **Commit:** `17b981e`

## The problem

The agent writes code that produces new columns. Those columns get trained on. At
prediction time they have to exist again. What do we persist to make that happen?

Two options:

1. **The output** — save the computed columns, look them up later.
2. **The program** — save the code, re-run it later.

## Background: why option 1 is the tempting one

Saving the output is what a feature *store* does, and for good reasons: it is
fast, it is auditable, and for expensive aggregations it may be the only
practical choice.

It also cannot work here, for a reason that is easy to miss: **a new row was
never in the training data.** If the bundle stores `price_per_m2` as a column of
800 values, then serving a flat that has never been seen means looking up a value
that was never computed. There is nothing to look up. The stored column answers a
question about the past, and prediction is a question about a row that did not
exist then.

You could store the columns *and* the code, using the code for new rows. Then you
have two sources of truth for the same feature and a way for them to disagree,
which is the definition of train/serve skew.

## The decision

**Store only the code**, as an ordered step in the table's lineage
(`{"type": "transform", "code": "...", "func_name": "transform_features"}`), and
re-run it in the same sandbox at prediction time.

A new row therefore goes through the identical arithmetic as a training row. Not
"equivalent" arithmetic — the same string of Python, executed by the same
sandbox. There is no separate serving implementation to keep in sync, which is
the usual place feature pipelines rot.

This only holds if the code is genuinely stateless, which is decision 0011 and
was not free.

## Alternatives considered

**Store the produced columns.** Rejected as above.

**Require the agent to write a `fit`/`transform` pair, like sklearn.** That is
the general solution and it is what `fit_encoding` is. Rejected for free code
because it makes the escape hatch as heavy as the structured tools, defeating the
point — and because a stateful object written by a model is exactly the thing we
have the least ability to verify.

## Consequences

**Good**

- Reproducible by construction. Ask what a feature means and the answer is a
  program, stored in the run, not a number whose provenance you have to trust.
- Auditable in a way that is unusually useful with an LLM in the loop. The
  journal stores the code too, so you can read *what the agent tried*, not merely
  what it achieved — including the ideas that did not work.
- Falls out of the existing lineage design (0006) with no new machinery.

**Bad**

- A subprocess launch per transform step at prediction time. Invisible for a
  batch of a million rows, real if you serve single requests at high QPS. The
  mitigation is batching, not removing the sandbox — and a bundle with no code
  step never imports the sandbox at all, because `Pipeline.transform` imports it
  lazily.
- The code must be deterministic and row-local, and *nothing about pandas
  enforces that*. See 0011, which exists entirely because this decision needs it.
- Prediction now depends on the sandbox working — a heavier runtime dependency
  than a lookup table.

## Where it lives in the code

- `src/tabular_agent/pipeline.py` — `Pipeline.add_transform` stores the code;
  `Pipeline.transform` re-runs it.
- `src/tabular_agent/sandbox.py` — `run_transform`, shared by fit and predict.
- `src/tabular_agent/harness/tools.py` — writes the step into the sidecar and
  the code into the journal row.
