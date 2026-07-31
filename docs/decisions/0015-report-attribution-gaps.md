# 0015 — Attribution gaps are reported, not prevented

- **Date:** 2026-07-30
- **Status:** accepted
- **Decided by:** Ulf (chose option 2 of 3)
- **Spotted by:** Ulf, reading a live run's table tree
- **Commit:** `84dd8bb`
- **Long version:** `docs/feature_attribution_and_agent_iteration.md`

## The problem

A live run produced this (a *table* is a file: the dataset plus columns added by
one tool call):

```
t4   date parts + frequency-encoded city    14 cols  ->  0.3160
t5   raw + date parts + ratios, logs, ...   23 cols  ->  never trained
t6   t5 + frequency-encoded city            25 cols  ->  0.2741   (best)
```

`t5` was built and immediately extended into `t6`, and only `t6` was ever
trained. Ulf's observation:

> this means frequency-encoded city could be the feature which leads to the
> massive jump in the score, not the big pile of features. so it mistakenly thinks
> this huge table is the best whereas it's actually only this single transform

Exactly right. The chain reads `0.3160 → (nothing) → 0.2741`, and one unmeasured
step in a chain means the jump cannot be attributed to either half of it.

The run is not *wrong* — the model that ships is the model that scored best. What
is missing is the explanation, and the agent is about to write a summary
confidently claiming which feature was the breakthrough.

(In this particular case the attribution is recoverable by luck: `t4` already
contained the frequency encoding and scored 0.3160, so comparing it with `t6`
isolates `t5`'s extra columns at about 0.042. But nothing *arranged* that
control. Had the run gone `t2 → t5 → t6` it would simply be gone.)

## The decision

**Report the gap; do not prevent it.**

`finalize_run` walks the winning table's ancestry, marks which links had
experiments on them, and says so — in the report and in an `unmeasured_tables`
field the agent sees before writing its summary:

> **Attribution gap:** `tables/t5.parquet` was built and then extended without
> ever being trained on. The winning score therefore cannot be split between the
> columns that table added and the ones added after it — there is no measurement
> in between. The model is valid; the explanation of *which features earned the
> score* is not available from this run.

Note the last sentence. Confusing "I cannot attribute this" with "this is wrong"
would be its own error.

## Alternatives considered

**1. Refuse to build on a table that was never trained on.** Forces a measurement
at every link, so attribution is always possible. **Rejected:** it constrains the
search to buy a reporting improvement, and the whole direction of that session
(0014) was to constrain the search *less*. The same goal was later achieved by
incentive instead — `vs_parent_table` only exists when the parent was measured
(0016).

**3. Leave it.** Rejected: the agent writes a confident summary and nothing
contradicts it.

**Group ablation** — retrain the winner with each block's columns dropped in turn
— was considered as a way to answer the question outright. Rejected: it costs *k*
retrains and still produces an order-dependent number that *reads* like a
definitive one, which is worse than saying nothing.

## The deeper limit, which this does not fix

Measuring every link would **not** have answered Ulf's original question. A fully
measured chain `raw → A → A+B → A+B+C` gives the marginal value of C *given A and
B*. It never gives the value of C. And that marginal value is order-dependent in
two opposite ways:

- **Redundancy inflates whoever goes first.** If two features carry the same
  signal, the first added takes the whole gain and the second measures ~0. Swap
  the order, swap the credit.
- **Interaction punishes isolation.** `price` and `area` are each nearly useless
  for the target; `price / area` is the whole thing. A feature measured alone can
  score zero and be essential.

Complete attribution needs every subset: 2ⁿ retrains. **This is exactly the
argument that motivates Shapley values** — average over all orders, because no
single order is privileged. Spec §18 (post-hoc SHAP on the winning model) is the
right machinery, and is cheap because it reads a trained model instead of
retraining.

## Consequences

**Good**

- The search stays unconstrained.
- The harness says out loud what it does not know. Same move as reporting
  `fold_spread` next to the mean: *do not stop someone believing a number; make
  sure they can see what the number does not cover.*

**Bad**

- It reports a gap rather than closing one. Anyone wanting the answer must run
  the ablation by hand.
- It only inspects the winner's ancestry. Unmeasured tables on abandoned branches
  are not mentioned, which is defensible but incomplete.

## Where it lives in the code

- `src/tabular_agent/harness/journal.py` — `ancestry`, `attribution`.
- `src/tabular_agent/report.py` — the "How the winning feature set was built"
  section and the warning.
- `tests/test_harness_tools.py` — reproduces the `t5`/`t6` case exactly.
