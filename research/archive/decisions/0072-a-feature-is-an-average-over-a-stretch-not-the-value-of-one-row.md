# 0072 — A feature is an average over a stretch, not the value of one row

- **Date:** 2026-08-17
- **Status:** accepted (a rule, a removal, and a compatibility guarantee)
- **Supersedes** the part of [0049](0049-the-last-observation-is-a-sample-of-size-one.md)
  and [0050](0050-the-anchor-is-what-matters-not-the-history.md) that kept the
  last-known pair as a feature worth emitting. Their *diagnoses* stand and are the
  evidence for this record; their conclusion — that pairing the value with its age
  makes it honest enough to keep — does not.
- **Answers** [0071](0071-the-gap-was-not-the-vocabulary-it-was-a-column-nobody-asked-for.md),
  which measured the price and deliberately left the fix unmade pending a second
  dataset.
- **Related:** [0011](../../../docs/decisions/0011-stability-check.md) (batch independence, which this
  never violated), [0048](0048-a-target-lag-is-not-target-encoding.md) (the per-fold freeze
  that turns out to be the mechanism), [0055](0055-a-rename-makes-every-earlier-artifact-unreplayable.md)
  (why the machinery is not deleted), [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md)
  (the interval both readings carry), [0063](0063-the-grain-is-measured-from-the-table-not-required-of-the-caller.md)
  (Ulf's rule that a precondition the library never states is a trap),
  [0069](0069-feature-selection-pays-on-columns-that-mislead-not-on-columns-that-are-idle.md)
  (why an *idle* column is cheap and this one is not),
  `src/tabular_agent/encoders.py`, `src/tabular_agent/harness/tools.py`,
  `src/tabular_agent/harness/agent.py`,
  `scripts/probe_restaurant_latest_pair.py`,
  `notebooks/phase5_the_column_nobody_asked_for.ipynb`.

## The rule

> **`fit_history` emits averages over a stretch of a group's past, and nothing
> else.** No last-known value, no lags. A feature that is the value of one row is
> not built here.

Ulf, arriving at it from the general direction rather than from the measurement:

> *"feature die auf dem wert eines tags (einer row) beruhen sollten nicht
> existieren, wenn sollten es (rolling)averages sein, die sind viel stabiler."*

## Why — and the reason is not the one the rule sounds like

The obvious argument is variance: one draw carries one draw's noise, an average
over ninety does not. True, and not what makes this expensive. The expensive part
is specific to this encoder and was invisible until it was plotted.

**The two sides of the fold loop disagree about what the column is.**

`fit_history` freezes history **per fold** (0048), and correctly: each fold's
history stops at the end of its own training prefix, so a validation row reads
what a served row the same distance ahead would read. The consequence nobody had
looked at:

| `Store_Sales_latest`, Rossmann v7's split | distinct | zeros | age |
|---|---|---|---|
| training rows (freeze per fold) | 4,041 | 0.1% | 1..305 days |
| served rows (final freeze) | 21 | **98.2%** | 1..45 days |

In training the column is a genuinely informative recent-sales number that varies
across four thousand values, and a tree learns to split on it. At serving it is
**one draw per group, constant for the whole horizon**, with only its age moving.
The model is fitted on one column and served another that shares its name.

And the one draw can land anywhere. v7's freeze is **1 May 2015**, a public
holiday on which 1,095 of 1,115 stores were shut, so 98% of stores were served
*"this shop's most recent sales figure is zero"* for all 45 days.

> **The per-fold freeze is correct, and its correctness is what makes the pair
> dangerous.** A naive single freeze would hand training rows the same degenerate
> value; the column would be near-constant and a tree would ignore it — harmless.
> Correctness makes fit and serve **procedurally consistent** and lets them drift
> **distributionally**, and no guard can fire on that: 0011 sees batch
> independence, 0030 sees servability, the fold logic sees no future. There is no
> error to find. Only a price.

That is also why the price is not the 0.0014 that 0069 measured for dropping nine
pure-noise columns. A boosted tree loses almost nothing to an *idle* column. This
one is not idle; it is confidently wrong.

**And `_ago` cannot rescue it.** The pair's original defence (0049, 0050) was that
the age lets a model discount a stale value. The age reports how *old* the number
is, never how *representative*. The draw that cost 0.014 was one day old.

## The measurements

Two datasets, both through the real tools, both with 0059's interval.

| | probe | interval | sealed |
|---|---|---|---|
| **Rossmann** — v7's split, winner's salesmodel rebuild (0071) | **+0.01428** | sd 0.00576, **5/5 blocks**, `sign_stable: True` | ~ |
| **Restaurant** — v2's `t7`, the table that shipped | **+0.00216** | sd 0.00205, 4/5, **`sign_stable: False`** | **+0.00349** |

`scripts/probe_restaurant_latest_pair.py` copies v2's split, drives the real
`finalize_run`, and puts **one table in each sealed arm** so there is no argmax to
go astray — the gotcha that made 0071's first printed number wrong. The control
passed: the `with_pair` arm reproduces v2's journalled blend val **0.519707** and
sealed **0.505713** to six decimals.

> **The mechanism transfers; the price is a lottery.** One value per served group
> for the whole horizon is structural — the freeze does not move while the rows
> do. But Restaurant's freeze is an ordinary Saturday, its train and serve
> distributions nearly coincide (82 against 87 distinct values, no zeros), and the
> column is genuinely informative on the scored rows (corr 0.55, against 0.69 for
> the 30-day window). **Rossmann's 0.014 was a holiday landing on the freeze.**

Two things nobody was looking for, recorded because they contradict things this
repo currently believes:

- **Sealed came back *larger* than probe** (+0.00349 against +0.00216), the
  opposite of the probe→sealed halving 0068 measured for capacity and 0071 assumed
  when it estimated Rossmann's sealed effect at ~0.008. That estimate is
  unsupported on a second dataset.
- **`val` prefers the pair.** The Restaurant blend gets *worse* on validation
  without it (0.51971 → 0.52229) and *better* sealed (0.50571 → 0.50222). 0068's
  signature again: **no default here may be read off a validation score.**

## What was decided, in the order it was decided

The measurements settle that the pair is never worth having and settle nothing
about the default, because +0.002 inside its own block noise cannot carry one.
Three shapes were put up; Ulf took the third:

- **A — pure opt-out.** `latest=True` stays the default. Nothing existing changes.
  Rejected: the agent would have to know about the defect to avoid it, and neither
  of these two readings could ever arise from a run, because no run performs an
  ablation.
- **B — the pair is emitted when it *is* the request.** Emit only when neither
  windows nor offsets were asked for. Chosen first, then withdrawn: it leaves the
  bare call as a door to a single-draw feature.
- **D — windows only.** Both the pair *and* offsets go, because both are the value
  of one row.

**Offsets go on weaker evidence than the pair — a correlation rather than an
ablation — and it points somewhere more interesting than expected.**

First, answerability. A lag reads `row - offset`, and stored history stops at the
freeze, so for a row `d` past it the lookup falls back to the last-known value
whenever `offset < d`. Measured by hand on v7's split:

| offset | answered exactly, served rows |
|---|---|
| 7 days | **15.6%** |
| 45 days (≈ the block) | 100% |
| 364 days | 100% |

0050 read 11% for a 7-day offset on v8; this is the same phenomenon on a different
split. **But that argument alone would only justify a stricter gate**, not removal:
past the block span a lag is answerable for every row. So the case has to be made
on the exactly-answerable lag, and it is:

| key | point, `364_ago` | window, 364-median | obs in the window |
|---|---|---|---|
| `Store` | 0.8292 | **0.8184** | 337 |
| `Store × DayOfWeek × Promo` | 0.8292 | **0.9500** | 27 |

Correlation with the target on the scored rows. Three readings, and the middle one
is the one worth keeping:

1. **The point lag is identical on both keys**, because 364 days is 52 weeks — "a
   year back" is the same dated row however you group.
2. **A window is not automatically better.** On the coarse key it is *worse* than
   the lag, because a median over 337 days averages every weekday and promotion
   together and washes out the structure. **An average over the wrong stretch is
   worse than a point.**
3. **The lag's strength is borrowed.** What makes `364_ago` decent is that 364 days
   back is the *same weekday*: the distance is smuggling a grouping key in through
   the back door. Put that key in `by`, average the 27 comparable days, and the
   same information reads 0.95.

> So the rule is not "averages beat points". It is: **comparability belongs in the
> key, and once it is there, averaging the comparable periods beats reading one of
> them.** A lag is a key expressed as a distance — which is why removing it costs
> nothing that cannot be said better, and why the tool's refusal message says to
> move the season into `by` rather than simply to use a window.

The winner's own feature set is already this shape: `prevyear_dps_med` is a median
over a year keyed on store × weekday × promo, not a lag at "the same day last
year".

**The honest limit:** these are correlations on served rows, not probe differences
with intervals. The `_latest` finding is measured a level more strongly than the
offsets one, and nothing here rules out a lag helping a *model* on some dataset
where a point beats its own keyed window.

## What it costs

- **Recency dies as a signal.** `_ago` alone — *"this group has not been seen for
  40 days"* — is a distance, not a draw, and a window does not carry it. On a
  ragged panel that is real information. It is not rebuilt here: 0058's rule is
  that instruments get built when a run needs them, and no run has asked.
- **No single-observation escape hatch at all**, including for a horizon of one
  where persistence genuinely dominates. Accepted deliberately; the door is what
  produced two sessions of this.
- **A window is only an average while it holds several observations.** At exactly
  the forecast horizon it holds the freeze-point observation alone, which is the
  old `_latest` with extra arithmetic. The `_count` column says so and the horizon
  gate refuses lengths below it — but the rule has this soft edge and pretending
  otherwise would be the failure it exists to prevent.

## The compatibility guarantee, and where the rule lives

**The refusal is at the fitting entry points, not in the class.** `HistoryEncoder`
still knows how to compute the pair and the lags, and `from_dict` defaults
`latest` to `True` when the key is absent.

That asymmetry is load-bearing. Bundles written before today hold history steps
shaped exactly like this real one:

    {'kind': 'history', 'by': ['Store'], 'value': 'Sales', 'order': 'Date',
     'offsets': [], 'keep': 92, 'windows': [[91.0, 'median']], ...}

No `latest` key, and windows present. Re-deriving the flag from the arguments
would resolve it to `False` and hand the model **fewer columns than it was trained
on** — a `KeyError` at predict time on an artifact that used to work, which is
0055's failure repeated. Four tests pin it, and a mutation of that one default
fails three of them with exactly that `KeyError`.

The refusal itself is pinned as a **signature**, not a behaviour: `offsets` and
`latest` are not parameters of `encoders.fit_history` or `harness.tools.fit_history`
at all, so re-adding them has to be a decision someone makes on purpose (0018's
argument for pinning the tool set, one level down).

## What was not done

- **The notebook covers both halves** — `notebooks/phase5_the_column_nobody_asked_for.ipynb`
  sections 8 and 9 are where the offset table above was produced, and where a
  pre-0072 bundle is restored by hand to show it still serves its pair. What it
  does **not** do is re-run anything under the new rule.
- **No re-run of anything.** No table on disk is rebuilt, no score is re-measured
  under the new rule, and the two probe scripts that pass `offsets=` are left as
  the historical artifacts they are.
- **`StreamAggregateEncoder` is untouched**, though `STREAM_STATS` already refuses
  `min`/`max`/`median` on 0049's reasoning — a bucket of three records has no
  meaningful extreme. Whether the same rule should be stated there in these terms
  is open.
