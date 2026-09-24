# 0005 — Ordinal level order is a required argument, never inferred

- **Date:** 2026-07-29
- **Status:** accepted
- **Commit:** `195f94d`
- **Prompted by:** Ulf asking whether the label encoding he had been doing by
  hand was wrong

## The problem

A model of the kind we train (gradient-boosted trees) needs numbers. A column
like `size` with values `small`, `medium`, `large` has to become numeric
somehow, and the reflex is:

```python
df["size"] = pd.factorize(df["size"])[0]              # or
df["size"] = df["size"].astype("category").cat.codes
```

This is **not** a leakage bug — neither line looks at the target. It has two
other problems, and the second is the dangerous one.

**Problem 1: invented order.** Trees split on thresholds, so encoding a category
as a number asserts an ordering. `factorize` numbers categories by order of first
appearance; `.cat.codes` numbers them alphabetically. For `small/medium/large`,
alphabetical gives `large=0, medium=1, small=2` — a confident, completely wrong
ordering, and nothing complains. When the categories are genuinely unordered
(cities), any ordering is meaningless and the model is worse off than with a
frequency or target encoding.

**Problem 2: the mapping is not stateful.** This is the one that causes outages.
`factorize` derives its mapping from the data in front of it. Run it on the
training frame and on the serving frame and you get two different dictionaries
whenever the rows differ. The model was trained believing `1` means one thing and
is served rows where `1` means another. Nothing errors. The score just quietly
degrades in production.

## The decision

A first-class `OrdinalEncoder`, in which **the level order is a required
constructor argument and is never inferred**:

```python
OrdinalEncoder(categories={"size": ["small", "medium", "large"]})
```

The order is stated once, stored in the encoder, and serialised with it into the
bundle. Train and serve cannot disagree, because neither derives anything.

Two further rules, deliberately asymmetric:

- **Strict at fit.** A value present in the training data but missing from the
  declared order raises. That is a typo or a level you did not know existed;
  silently inventing a rank for it is how the original problem comes back.
- **Lenient at predict.** A category first seen at prediction time maps to `-1`,
  below every declared level, so one split isolates it. Unseen categories at
  serving time are normal life, not a bug.

## Alternatives considered

**Infer the order and warn.** Rejected. A warning is not read, and the failure it
warns about is invisible. If we cannot know the order, we must be told it.

**Only offer frequency and target encoding.** Safe, and loses real information:
when an ordering genuinely exists, the rank *is* the signal. Ordered categoricals
are common enough to deserve support.

**Sort alphabetically as a default.** Rejected for the reason above — it produces
a plausible-looking wrong answer, which is worse than refusing.

## Consequences

**Good**

- Round-trips through JSON with the rest of the pipeline, so the ordering reaches
  the deployed bundle. A test reverses row order and removes a level entirely; the
  codes do not move.
- Forces the caller to state domain knowledge explicitly. That is the right place
  for it — the data cannot supply it.

**Bad**

- More typing for the caller, every time.
- **It made ordinal encoding effectively unusable by the agent at first.** The
  agent has to state an order for categories it has never seen, and `inspect_data`
  originally reported only `n_unique: 3`. The agent sent an empty list and the run
  died. Fixed by having `inspect_data` list the actual category values for
  non-numeric columns with ≤20 uniques — and on the very next live run the agent
  chose `ordinal` for `size`, having gone `frequency` three runs running. It was
  never reluctance; it was missing information. A good reminder that a tool's
  *inputs* are only as usable as the tool that describes the data.

## Where it lives in the code

- `src/tabular_agent/encoders.py` — `OrdinalEncoder`, registered in
  `ENCODER_CLASSES` so it serialises.
- `src/tabular_agent/harness/tools.py` — `inspect_data` lists categories for
  low-cardinality non-numeric columns.
- `notebooks/phase3_harness_step2_encoding.ipynb` — walks through the trap.
