# 0057 — The capabilities the agent does not have

- **Date:** 2026-08-14
- **Status:** a design inventory, **nothing implemented**. This is the list of what
  the agent cannot express, why each item is general rather than borrowed from one
  competition, and what tool would have to exist. Each item still needs its own
  record before it is built — this one exists so the next session argues about the
  right things instead of rediscovering them.
- **Related:** [0056](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
  and [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
  (the measurements that motivate most of this),
  [0035](../../../docs/decisions/0035-the-search-is-over-features.md) (the search is over features, which
  is what makes this list the right list),
  [0029](0029-the-harness-owns-the-exam.md),
  [0011](../../../docs/decisions/0011-stability-check.md) and [0010](0010-store-code-not-columns.md) (what
  any new tool has to satisfy),
  [0047](0047-the-forecast-horizon-is-a-column-not-a-constant.md),
  [0046](0046-the-mean-is-a-lossy-summary-of-a-group.md),
   `scripts/prep_restaurant.py`.

## Where this list comes from

Two sources, deliberately kept apart from the datasets they were found on.

**Read from a competition winner's own writeup**, distilled under the rule
CLAUDE.md sets: a capability counts only if it can be stated *without naming the
dataset*. "Per-group aggregates of a column, fitted once and replayed" passes;
"exclude rows where the store was closed" does not, and its general form — "some
rows should not be trained on" — is what got built instead.

**Found by running a second dataset.** `prep_restaurant.py` turned up a train/serve
skew no guard here detects, and its general form turned out to be *the same
capability* the winner's writeup asks for from the other direction. That
convergence is why item 1 is first.

## The agent's surface today

Ten tools: `inspect_data`, `fit_encoding`, `fit_group_stats`, `fit_history`,
`filter_rows`, `transform_features`, `explain_table`, `explore_table`,
`explore_journal`, `finalize_run`.

Everything below is something none of them can express. `transform_features` is
not the escape hatch it looks like: 0011 refuses any column whose value depends on
which other rows are in the frame, which rules out every aggregate, every lookup,
and every fit. That refusal is correct and is what makes the rest of this a list of
*tools* rather than a list of code snippets.

---

## 1. Features restricted to what was knowable at a given lead time

**The capability.** Summarise a related stream — one whose rows keep arriving over
time — using only the records that existed a stated number of periods before the
row being predicted.

**Why it is general, and not one dataset's quirk.** Two independent findings on two
datasets point at exactly this:

* A second dataset ships a reservations stream **cut at the moment the forecast is
  issued**. 30% of training rows can include a booking made the same day; **no row
  in the forecast window can, ever**, and the floor grows from 1 period at the near
  edge to 39 at the far edge. Every guard passes: no fold is crossed, no encoder is
  misfitted, the column exists at serving time, and validation *confirms* the
  column is excellent because validation is drawn from the same uncut stream.
* The competition winner built **"month ahead models"** that deliberately skip the
  most recent month when computing features, and used *only* those for the far half
  of the horizon — because at that lead time the recent month does not exist yet.

Both are the same statement: **a feature computed from a stream is only honest if
it is computed at the lead time it will be served at.** Any order book, booking
system, pre-registration, pipeline or claims feed has this shape.

**What the agent needs.** An aggregation tool taking a `lead` argument in the
ordering column's units, which excludes records observed later than
`row_time - lead` from that row's summary. The stream's own observation time has to
be a declared column, because nothing recovers it by inspection.

**What the harness must guarantee.** That the lead is applied identically at fit
and at serve; and — since 0047 established the horizon is a *column* and not a
constant — that a row deep in the forecast block gets a summary computed at *its*
lead, not at the block's floor.

**What it is not.** Not a lag. A lag reads one earlier value of the target
(0048); this restricts *which records may enter an aggregate of another column*.

---

## 2. Signed distance to a named event

**The capability.** Given a column marking occurrences of something, emit for each
row the distance to the previous occurrence, the distance to the next, and whether
the row is inside one.

**Why it is general.** The winner's writeup calls these "day counters" — days
before, after, or within promotion cycles, holiday periods, refurbishments,
competition openings — and the survey of six forecasting competitions names event
counters as broadly useful across all of them. The general fact is that **a flag
says an event is happening; it cannot say one is imminent.** Anticipation and
after-effects are ordinary in any process with scheduled interventions, and a
binary column represents neither.

**What the agent needs.** A tool taking an event column, an ordering column, and
producing the previous/next/within triple.

**What the harness must guarantee.** This is the subtle one. Distance to the *next*
occurrence is a lookahead, and whether it is legitimate depends on whether the
calendar is known in advance. For a published holiday schedule or a planned
promotion it is known and the feature is honest; for an unplanned event it is a
leak. **The tool must therefore require the caller to declare the event calendar as
known-ahead**, and refuse otherwise, rather than inferring it.

**What it is not.** Not derivable through `transform_features` — the distance to
the nearest occurrence depends on which other rows are present, so 0011 refuses it,
correctly.

---

## 3. Per-group trend extrapolation

**The capability.** Fit a small parametric model per group on the ordering column
plus a few covariates, store its coefficients, and emit its fitted or extrapolated
value as a feature.

**Why it is general, and this is the strongest argument on the list.** It is not
a trick; it repairs a structural limitation of the model class the harness ships.
**A gradient-boosted tree cannot extrapolate.** Its prediction is an average of
training rows, so beyond the range of a monotone covariate — which is exactly what
a forecast is — it is flat by construction. The survey states the winner's key
adaptation in precisely these terms: *"a trend adjustment using a ridge regression
model to deal with the fact that GBDT cannot extrapolate trends."*

That argument names no dataset. It applies wherever a boosted tree is asked to
predict outside the span it was trained on, which for this harness is every
ordered split.

**What the agent needs.** A tool taking `by`, an ordering column, and optional
covariates, fitting one regularised linear model per group and emitting its
prediction. The stored state is coefficients per group — bounded by group count,
which is what makes it storable in a lineage at all (the same argument
`fit_history` uses for its bounded tail).

**What the harness must guarantee.** Fitted per fold like every other fitted
encoder; groups unseen at serve time fall back to a global fit; and the number of
coefficients per group is fixed so the state cannot grow with the data.

---

## 4. Trailing-window aggregates keyed by column *combinations*, at several lengths

**The capability.** Per-group summaries over a window measured backwards from each
row's own position, where the group is a combination of categoricals and several
window lengths coexist as separate columns.

**Why it is general.** The winner computed recent-history summaries over the last
quarter, half-year, year and two years, **split by combinations** of the three
covariates that drove most of the variance. Two separable ideas:

* *Multiple horizons of recency are different features*, not competing settings for
  one feature. A short window tracks the current regime; a long one estimates the
  seasonal level. Making the agent choose one is making it discard the other.
* *The key is a combination.* A per-entity average and a per-entity-per-weekday
  average answer different questions.

**Why this is the item to be most careful with.** We already built a version of
this and it lost. 0052 measured the trailing window *worse* than 0046's unbounded
mean of the same key, and 0053 then showed a window's measured worth swings by 0.14
with the fold count while the unbounded aggregate is invariant — so **window length
cannot be searched over here and the number believed.** 0056 then refuted the whole
frozen-history family externally.

The honest reading is that our *implementation* of recency was refuted, not the
idea; the winner's version differs in ways we never tested (keyed by combinations,
several lengths at once, computed relative to each row rather than frozen at one
boundary date). But that is a hypothesis, and 0053 says the instrument for testing
it is unreliable. **Nothing here should be built until the ranking block from 0056
is fixed** — otherwise we will measure it with the ruler that just ranked four
models nearly backwards.

---

## 5. Several statistics per group, chosen by the agent

**The capability.** `fit_group_stats` takes one `stat`. The winner summarised each
group with measures of centre *and* spread — median, mean, harmonic mean, standard
deviation, skew, kurtosis, 10th/90th percentiles.

**Why it is general.** 0046 already made this argument and its title is the
statement: the mean is a lossy summary of a group. Spread is a different quantity
from level and is not recoverable from it.

**The caveat, from the same source.** The winner reports that *"the spread features
made overfitting easy"* and that only one of his log-transformed summaries survived
selection. So this capability arrives *with* its own hazard, and adding it without
the selection machinery to control it is how the vocabulary grows faster than the
ability to judge it — which is the pattern 0052/0053 record.

---

## 6. A budget the harness chooses

Covered by [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
and confirmed externally by 0056, and listed here only so the inventory is
complete. Not an agent capability: **0035 removed model parameters from the agent
deliberately and that stands.** The harness fits each family at a chosen budget;
the agent never sees the knob.

---

## Explicitly rejected, with reasons

* **External data** — weather, search-trend statistics, published population and
  income figures. The winner used weather and reports the state-level maximum
  temperature and precipitation helped. It stays out: the question this project
  asks is what the harness can express about *the data it was given*, and a
  capability that consists of finding a better file is not one of ours.

  **A gap in the record, noticed while writing this.** CLAUDE.md states this rule
  and cites `0001` for it. 0001 is *Own a constrained harness rather than serve
  other agents* and says nothing about external data; no record establishes the
  rule at all. It is a real project constraint held only in CLAUDE.md and in
  habit. Either it deserves its own record or CLAUDE.md's citation should be
  dropped — but it should not keep pointing somewhere it is not.
* **A constant multiplier on predictions.** The winner applied 0.995 and writes
  that the forum's 0.985 would have taken him "well below 0.10". It stays out
  (CLAUDE.md): it is a fitted correction to one leaderboard with no defence in its
  own terms.
* **Handing hyperparameters back to the agent.** See item 6.
* **More blending.** The survey reports ensembling bought the winner about 5% over
  his best single model, and his *individual* models already scored around 0.105
  private. 0056 found our three-way blend losing to a single model. Blending is not
  where the missing score is.

## The order to build them in, and the reason for it

1. **Fix the ranking block first (0056).** Items 1-5 are all features, every feature
   is judged by `val`, and `val` just ranked four models at Spearman +0.20 against
   the external truth. Building any of this before that is measuring with a ruler
   known to be wrong.
2. **Then item 1 (lead-time restriction)**, because it is the only item found
   independently twice, it fixes a skew that currently passes every guard, and one
   dataset already carries a natural experiment for it.
3. **Then item 3 (trend extrapolation)**, because its justification is a property of
   the model class rather than an observation about a competition, which makes it
   the most likely to generalise.
4. **Items 2, 4, 5 after that**, and item 4 not at all until 0053's measurement
   problem has an answer.
