# 0032 — Fitted group aggregates, and the first step that is not replayed

- **Date:** 2026-08-11
- **Status:** accepted, implemented, **validated** (`_runs/rossmann_v5`) — with one
  of the two premises refuted by the run; see "What the validation run measured"
- **Related:** [0011](../../../docs/decisions/0011-stability-over-syntax.md) (the stability check these
  two work around from opposite sides), [0010](0010-store-code-not-columns.md)
  (why a stored transform is code and therefore replayed), [0009](../../../docs/decisions/0009-the-target-is-not-in-the-frame.md)
  (the target withheld from `transform_features`, and why a filter is the
  exception), [0029](0029-the-harness-owns-the-exam.md) (the line a row filter
  sits next to without crossing), [0030](../../../docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md)
  (the serving check both of these have to satisfy)

## The two gaps, and where they came from

Both were found the way the project intends them to be found — by running a real
dataset and watching what the harness could not express — and both are recorded
here in the general form, not the dataset's.

**1. A per-group statistic of a non-target column had no path at all.** Rossmann
v4's *first attempt* died in 2.5 minutes with zero experiments on the table. The
agent tried to build per-store rates from another column, 0011's stability check
refused it three times (correctly — a `groupby(...).transform("mean")` written in
`transform_features` recomputes from whatever rows are in the frame), the tool's
retry budget ran out, and `UnexpectedModelBehavior` ended the run. The refusal
message pointed at `fit_encoding`, **which cannot do this**: `fit_encoding` gives
per-group statistics of the *target*.

So the agent reached, unprompted, for a capability that did not exist, and the
one hint it was given was wrong.

**2. No rows could be excluded from the fit.** `transform_features` must preserve
the row index, so there was no way to train on a subset. The general shape:
*some rows should not be trained on* — rows the metric does not score, rows from a
regime the model is not being asked to predict, outliers whose squared error
swamps the fit.

## The decisions

### `fit_group_stats(by, value, stat)` — fit once, look up forever

Mechanically `FrequencyEncoder` with `value_counts` swapped for a stored
`groupby`. It adds one column, `<by>_<value>_<stat>`, and the mapping goes in the
table's lineage sidecar.

What makes it legal where the free-code version is not is not a loophole in
0011 — it is the opposite. 0011 rejects batch-dependence, and the whole point
here is that **the group's number is stored**, so one row maps to the same value a
million rows would. The stability check would pass this if it could see it; it
cannot, because the value never survives a `transform_features` call to be
checked.

**It is always out-of-fold, and there is no way to turn that off.** The first
version of this design said a per-group statistic of a *non-target* column needs
no out-of-fold discipline at all. That is too strong and would have shipped a
leak. It holds for a genuinely target-independent column. It fails for one
correlated with the target: a per-group mean of such a column is very nearly a
per-group mean of the target, and computing it including the row's own
contribution is target encoding with the guard removed. Large groups hide it — a
row moves a thousand-row group's mean by a thousandth — but group size is a
property of the data, and **the tool cannot inspect the correlation to decide**.
One pass per fold is cheap; a silent leak under a name that does not mention the
target is not.

The consequence worth having: **the value column does not need to exist at predict
time.** Only the `by` columns are read at lookup. A quantity measurable in history
but not on the day you predict — the exact shape of 0030's refusal — becomes
usable *as its group's history*, never as itself. 0030 turns from a dead end into
a redirect.

Only the **time-windowed** variant stays out: "the last quarter's mean for this
group" needs a cutoff, per-fold discipline and a frozen version for predict. Not
built; the plain version covers what the agent actually asked for.

### `filter_rows(code)` — the first train-time-only step

The design point that makes this a decision rather than a patch: **a row filter is
the only pipeline step that is not replayed at prediction time**, and it must not
be, for two independent reasons.

- **Scoring.** `_score_on_val` replays the winning table's lineage over raw
  validation rows and lines the predictions up against held-back targets. A filter
  in that replay drops some of those rows and misaligns the two — and worse, would
  grade a filtered experiment on an easier exam than an unfiltered one, making the
  two `val_score`s incomparable, which is the number the whole run ranks by.
- **Serving.** Every row handed to a deployed model needs a prediction back. A
  model that silently declines to answer for some fraction of requests is not a
  model, and nothing downstream would know which fraction.

So the rows are dropped **once**, when the table is written, and the lineage
records the step with `train_only: True` so `Pipeline.transform` skips it. Kept in
the lineage rather than omitted, because the lineage is also the record of how a
table was built and a table with fewer rows than its parent has to say why.

Two properties follow, and both read backwards at first:

- **A batch-dependent predicate is fine here.** `x > x.mean()` has to be rejected
  in `transform_features` because the stored code is recomputed on whatever rows
  arrive at predict. A filter's code is evaluated exactly once, against one frame,
  and never again — there is no second run for it to disagree with. This is the
  only agent-written code in the harness that needs no stability check.
- **The target is visible to it, and that is not a hole in 0009.** 0009 withholds
  the target from `transform_features` because that code *is* replayed on rows
  that have no target. Nothing a filter produces is replayed anywhere, and the
  validation half never passes through it, so dropping training rows by their
  target value — outlier removal — is ordinary modelling rather than leakage.

This stays inside 0029's line. The harness owns the exam; filtering *training*
rows is modelling, and the validation half is untouched.

## What the validation run measured

Rossmann v5: 14 experiments, 21.1 min, two attempts (the first died on
`explore_experiment` exhausting its retry budget — v4's failure shape in a
different tool — and the driver's restart-on-error salvaged it). Best `e13`,
`rmspe` **0.13760** against a baseline of 0.44541. **The first run on this dataset
to finalize itself and write a `submission.csv`.**

Both premises above were tested. **One held and one did not, and the one that
failed is the one this record argued hardest for.**

**`filter_rows` was reached for unprompted, twice.** Nothing in the prompt mentions
it; the docstring was the whole channel. The first use (`t3`) was confounded — the
agent stacked four more tables on top before training, so the branch's bad score
said nothing about the filter. It came back and did it properly:

| | table | features | model | `val_score` |
|---|---|---|---|---|
| `e8` | `t9` | 29 | catboost / log1p | 0.15589 |
| `e10` | `t13` = `t9` + filter | **the same 29** | **the same** | 0.15924 |

`vs_source_table = -0.00336`, on a `fold_spread` of ~0.07. **Excluding the 17% of
rows the metric does not score cost nothing and gained nothing** — the difference
is a twentieth of the noise. The agent read it correctly and abandoned the branch.

That refutes the premise this capability was prioritised on. Two earlier entries
called row filtering "the cheapest gap to close, biggest effect here", reasoning
from the 17% figure alone. The reason it does not pay is visible in hindsight and
was not argued anywhere: **`Open` is a feature.** The model can already learn
"closed → 0", so removing those rows takes information away and gives nothing
back. The step-8 notebook reproduces this independently on synthetic data.

The capability is still right to have — "some rows should not be trained on" is a
real gap and this is a clean way to close it — but **the argument for it is
generality, not this dataset's 17%.** A filter earns its keep only where the
regime is *not* expressible in the features, or the metric genuinely ignores the
dropped rows *and* the fit is being distorted by them. Neither holds here.

**`fit_group_stats` was never called.** Zero calls in 69 tool calls across both
attempts. The tool the agent *asked for* in v4 — the one whose absence killed that
run — went untouched, while the tool nobody asked for got used twice. n=1, one
model, and the agent had a working per-store target encoding already (`Store_te`),
which covers much of the same ground; so this is not evidence the tool is
unwanted. It is evidence that **building the thing an agent once reached for does
not make it reach again**, which is the fourth time a change to what the agent is
told has failed to change what it does (0024, 0027, 0028).

Also worth recording: `experiments_using_customers: 0` for the second run running,
so 0030's guard is holding.

### The submission scored, and the exam was honest for the first time

**Kaggle public: 0.135**, against our `val_score` of 0.13760. Rank ~2140/3300.

The absolute number is modest. **The calibration is not, and it is the most
important thing this run produced.** Every previous run's validation half was
*optimistic* — house-prices run 3 said 0.11354 and Kaggle said 0.12298, a 0.00944
gap that was 91% of the gain the run thought it had made (0022). Here the
validation half was **pessimistic by 0.0026**: the model did slightly better on
genuinely unseen data than our own exam predicted.

That is 0029 paying off, and it is the first direct evidence for it. An ordered
split on `Date` plus forward-chaining folds asks the same question the competition
asks — train on earlier rows, predict later ones — so there is no interpolation
premium to give back. It also means this dataset's `val_score` can be trusted in a
way house-prices' never could, which matters for 0028's open `min_gain_ratio`
question: this is the second dataset that question was waiting for, and it arrives
with a validated yardstick.

## Why this and not the alternatives

**Not `method="group_stats"` on `fit_encoding`.** Different arguments (`by`,
`value`, `stat` against `columns`, `categories`, `smoothing`) and a different
question. Folding it in would have produced one tool with two disjoint signatures
and a docstring that has to explain which half applies.

**Not an arbitrary aggregation callable.** The encoder is serialized and replayed,
so whatever computes the number has to survive a round trip through JSON. A name
does; a lambda does not. Hence a fixed menu.

**Not "filter and remember the mask".** Storing which rows were dropped would make
the artifact the rows rather than the rule, and 0010 settled that the other way for
transforms. Storing the code and marking it train-only keeps one convention.

**Not a `train_only` flag the agent passes to `transform_features`.** A boolean
that silently changes whether a step is replayed is exactly the kind of thing that
gets passed wrong once and produces a bundle that refuses to predict. A separate
tool with a separate contract (`keep_rows` returning a mask) cannot be confused
for the other thing.

**Not a mask-shaped return of *rows*.** `keep_rows` returns a boolean Series, not a
filtered frame, because a frame could also reorder, rename, or invent rows and each
would have to be checked for. A boolean Series indexed like the input can only
select.

## What it costs

**Good**

- The capability the agent reached for unprompted now exists, and the refusal
  message that used to misdirect it now names the right tool.
- A column that 0030 could only refuse is now usable in the one form that is
  honest.
- `val_score` stays comparable across filtered and unfiltered experiments, so the
  leaderboard needs no special case.

**Bad**

- **`cv_score` quietly changes meaning under a filter** — it is now the mean over
  the kept rows only, while `val_score` is over everything. The note says so; the
  agent has to read it. Two numbers on one row measuring different populations is
  the sort of thing that gets compared anyway.
- **Out-of-fold group statistics inherit the NaN block.** Under forward chaining
  the first period is validated by nothing, so those rows have no value — the same
  gap out-of-fold target encoding leaves (0029's gotcha), now on a second column
  family, and the missingness is time-correlated.
- **The out-of-fold requirement costs a fit-time pass per fold** on a column that
  may well be target-independent, where a single pass would have done.
- **A filter's floor is a guess.** `max(n_splits + 1, 10)` rows is computability,
  not adequacy — the same shape as `MIN_SPLIT_ROWS`, with the same silence about
  whether what survives is enough to learn from.
- **Nothing stops the agent filtering its way to a flattering `cv_score`** by
  dropping hard rows. `val_score` is unmoved, which is what makes this safe rather
  than merely bounded — but the run's own fold numbers can be gamed, and
  `explain_table`'s contributions are measured on the filtered rows too.
- **Two more tools in a surface whose smallness is the argument** (0012): ten now,
  up from eight.

## Where it lives

- `encoders.py` — `GroupStatsEncoder`, `fit_group_stats`, `GROUP_STATS`,
  `_EXTENSIVE_STATS`; registered in `ENCODER_CLASSES` so a bundle can revive it.
- `rows.py` — `select_rows`, and the module docstring carrying the
  not-replayed argument.
- `sandbox.py` — `run_filter`, the third mode, with the `keep_rows` contract.
- `pipeline.py` — `Pipeline.add_filter` and the `train_only` skip in `transform`.
- `harness/tools.py` — `fit_group_stats`, `filter_rows`, `MIN_FILTERED_ROWS`, the
  `n_dropped` column on `explore_journal`'s `tables` frame.
- `harness/agent.py` — both tool registrations, which is where the agent learns
  they exist.
- `transform.py` — the stability refusal now names `fit_group_stats` instead of
  saying no such tool exists.
- `tests/test_group_stats.py`, `tests/test_rows.py`,
  `tests/test_harness_rows_and_groups.py`.
