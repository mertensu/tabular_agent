# 0009 — The target column is invisible to agent-written code

- **Date:** 2026-07-30
- **Status:** accepted
- **Commit:** `17b981e`

## The problem

`transform_features` lets the agent write its own pandas code to derive new
columns — `price / area`, `2026 - year_built`, `date.dt.month`. This is the
escape hatch for everything the structured tools cannot express, and without it
the agent is stuck with whatever encoders we happened to build.

The code runs in a sandbox that blocks the model and sklearn libraries, so it
cannot fit an estimator. But the frame it receives is the dataset, and the
dataset contains the target column. Which means this is one line away:

```python
df["cheat"] = df["sold_fast"]
```

Pure pandas. Nothing to import, nothing to block.

## The decision

**Drop the target column before the code sees it.** The agent's function receives
the features only. Reaching for the target raises, and the error explains why
rather than surfacing a bare `KeyError: 'sold_fast'` (which a model reads as a
typo and retries).

There are two independent justifications, and it is worth knowing which one is
load-bearing.

### Reason 1: leakage

A derived column cannot be a function of the target if the target was never in
the frame. This is the same posture as blocking sklearn — make the mistake
*unexpressible* rather than discouraged. Guidance in a prompt does not survive a
model that is trying to improve a score.

### Reason 2: there is no target at prediction time — this is the deciding one

At serving time you receive a flat to price, not a flat plus the answer. The
target column does not exist.

Now recall that only the **code** is stored, and it is re-run at prediction time
(0010). So code referencing the target would either crash at prediction, or —
worse — silently compute something different if a column of that name happened to
exist. The frame must have the same shape in both places or "re-run the code at
predict" is not true.

This is why the rule is enforced rather than merely encouraged: **even a
well-intentioned, non-leaky use of the target would break the replay.** A model
might reasonably write `df["above_avg"] = df["price"] > df["price"].mean()` while
glancing at the target for a sanity check — no leakage intended, and the pipeline
still breaks.

## Alternatives considered

**Pass the whole frame and scan the code for target references.** Rejected:
static analysis of Python is defeated by `df[cols[-1]]` and a hundred other
spellings. Removing the column is exact.

**Pass the frame and check the output columns for suspiciously high correlation
with the target.** A real technique, and a heuristic — it would flag legitimate
strong features and miss laundered ones. Also it does nothing about reason 2.

## Consequences

**Good**

- The whole class of "derived a feature from the label" is gone, with no
  detection logic to maintain.
- Training and prediction receive frames of the same shape, so the replay
  contract is honest.

**Bad**

- Genuinely target-dependent features are impossible here, which is correct but
  restrictive. They belong in `fit_encoding(method="target")`, which does it
  properly: out-of-fold values to train on, an all-rows encoder to serve.
- A model that has not read the tool description will try it, waste a turn, and
  need a good error message. The message therefore had to be written carefully —
  see the note appended when the target name appears in a sandbox error.

## Where it lives in the code

- `src/tabular_agent/harness/tools.py` — `transform_features` drops
  `spec.target` before the sandbox call, rejects an output column named after the
  target, and rewrites the resulting error.
- `tests/test_harness_tools.py` — `test_transform_code_cannot_see_the_target`.
- `notebooks/phase3_harness_step3_transforms.ipynb` — §6.
