# 0046 — The mean is a lossy summary of a group, and the loss was not sayable

- **Date:** 2026-08-14
- **Status:** accepted, implemented, **not yet validated by a run**
- **Related:** [0032](0032-fitted-aggregates-and-a-step-that-is-not-replayed.md)
  (`fit_group_stats`, which has the whole statistic menu and cannot point it at
  the target), [0011](0011-stability-over-syntax.md) (why none of this can be
  written in `transform_features`), [0030](0030-a-model-that-cannot-be-served-is-not-trained.md)
  (every column here is computed from a column that will not exist at predict
  time), [0021](0021-the-agent-picks-the-target-scale.md) (a menu entry should
  be worth an experiment), and `docs/time_aware_features.md` §2 Gap A, which is
  the argument this implements.

## The gap

Rossmann v7 spent half its run re-drilling one hole. `t8` target-encoded a
hand-made composite key — "average sales for this store on a Monday" — for
**+0.0243** on the probe, the largest late gain of the run. Tables `t9`–`t15`
then built nine more composite keys through the same tool for **+0.0006 total**,
three of them negative.

That is not the agent being weak. Every per-group feature it could build went
through one mechanism, and once `t8` existed there was nothing else the harness
let it *say* about a group. It exhausted its vocabulary and repeated itself.

Stated without the dataset: **the harness could learn a group's average target
and nothing else about that group's target distribution.**

Two categories with the same average can have completely different distributions.
An average standing on three rows should not be read like one standing on three
hundred. Neither fact was expressible, and the second is the sharper loss,
because the group's count was *already being computed here* as the smoothing
denominator and then thrown away.

## Why the obvious workarounds do not exist

- **`transform_features`.** A `groupby(...).transform("std")` recomputes from
  whatever rows are in the frame. 0011's stability check rejects it, correctly —
  one row alone gets a different answer than the same row in a batch.
- **`fit_group_stats`.** It already has the full `mean/median/std/count` menu
  and it **refuses the target as its value column**, deliberately (0032): a
  per-group statistic of the target is target encoding, which needs smoothing
  and out-of-fold fitting, and the refusal exists so nobody reaches the leaky
  path through the non-leaky tool.

So the capability was blocked from both directions, and correctly from both.

## The decision

`fit_encoding(method="target")` takes `stats=`, defaulting to `["mean"]`.
`TargetEncoder` emits one column per requested statistic:

    mean   -> <col>_te            (unchanged: the classical target encoding)
    std    -> <col>_te_std
    median -> <col>_te_median
    count  -> <col>_te_count

The rule this settles, and the reason it is one tool rather than two: **the
target's per-group distribution is one concept and lives in one place.** The
alternative — lift `fit_group_stats`'s target refusal for the non-mean
statistics — is less code and reuses tested machinery, and it produces an
indefensible seam where a group's target *mean* comes from one tool and its
target *median* from another.

### Only the mean is smoothed, and that is the load-bearing call

Smoothing shrinks a group's estimate toward the population value in proportion to
how little evidence stands behind it, and the weighted-average formula that does
it is a statement about **means**. Shrinking a median toward the global median by
a count weight, or a standard deviation toward the global standard deviation,
would be inventing an estimator and giving it the name of a standard one.

What replaces it is better anyway: **`count` is emitted as its own column**, so
the model learns how far to trust a thin group instead of having a fixed answer
baked in by a smoothing constant nobody tuned. Same information, expressed rather
than assumed.

### The menu is four entries, and the exclusions are the argument

`TARGET_STATS = (mean, std, median, count)`. Deliberately absent:

- **`min` / `max`** — an extreme of the target within a group is essentially one
  row's target. It survives out-of-fold fitting only by the luck of which fold
  that row landed in, which makes it the one shape of per-group statistic whose
  leak the existing discipline does not reliably contain.
- **`sum`** — exactly `count * mean`, and a model that can see both learns
  nothing from being handed the product. 0021's "a menu entry should be worth an
  experiment" cuts against it.

### Every statistic is out-of-fold, including `count`

`count` does not read the target's *value* at all, only that a row was there, so
in principle it could take a looser path. It does not get one. A second, weaker
route through the same tool is a thing somebody has to keep correct forever in
exchange for nothing measurable, and the per-fold pass is already being made.

## What this cost, and what nearly went wrong

**The serialized encoder changed shape.** `mapping` was `{column: {category:
value}}` and is now `{column: {stat: {category: value}}}`. `from_dict` detects the
old shape by the absence of a `stats` key and loads it as mean-only, because a
bundle on disk has to keep predicting exactly what it predicted when it was
written. One existing test asserted on the flat shape and was updated; that was
the only reader.

**A leak test that would have passed while the leak was real.** The mean's
out-of-fold test correlates the naive encoding with `y` and shows it fabricates
signal. Applied to `std` it *fails* — and not because `std` is safe. A row
inflates its group's spread whether its own target is unusually high **or**
unusually low, so the leak lands on `|y - mean|` and can sit near zero against
`y` itself. Correlation is simply the wrong probe for a statistic that is not
monotonic in the row's own value.

The replacement tests the property directly: **move one row's target and watch
that row's own feature.** Fitted on all rows it moves; fitted out-of-fold it does
not. One mechanism, all four statistics, and it asserts the thing that actually
matters rather than a proxy that happens to work for means.

Worth keeping in mind generally: *a leak test written for one statistic does not
transfer to another statistic of the same column.*

## What is left open

- **No run has used this.** Whether an agent reaches for `std` and `count`
  unprompted is the measurement, and 0032 is the cautionary precedent —
  `fit_group_stats` was the tool the agent explicitly asked for and then never
  called once in 69 tool calls.
- **`count` under an ordered split is a partial count**, over the folds preceding
  the row rather than the whole history. That is correct for leakage and means
  the training column and the served column measure subtly different things: the
  stored encoder's count covers all rows. The same is already true of the mean,
  which is why this is noted rather than fixed.
- This is slice 1 of three in `docs/time_aware_features.md`. It closes Gap A. Gap
  B — a feature that can look at another row in time — is untouched, and is the
  larger half.
