# 0070 — What the winner had that we cannot express

- **Date:** 2026-08-16
- **Status:** accepted (an audit; nothing built)
- **Related:** [0057](0057-the-capabilities-the-agent-does-not-have.md) (the
  previous list of missing capabilities, which this revises),
  [0058](0058-the-ranking-failure-was-one-feature-not-the-block.md) (whose item 2
  this promotes to next), [0069](0069-feature-selection-pays-on-columns-that-mislead-not-on-columns-that-are-idle.md)
  (item 4 here, built and measured at zero),
  [0064](0064-month-ahead-is-the-same-idea-and-does-not-get-the-same-tool.md)
  (a refusal this audit vindicates),
  [0001](0001-own-the-harness.md) (external data stays out),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md) and `CLAUDE.md`
  (a capability must be statable without naming the dataset),
  `rossmann_store_sales/winner.md`, `rossmann_store_sales/Rossmann_nr1_doc.pdf`.

## Why this exists

`HANDOFF.md` has carried "distill the winner writeup" as sanctioned design input
for two weeks, with a guardrail: the distillation has to come out as **capabilities
stated without naming the dataset**, or it is leaderboard-fitting one level up.
Ulf added the documents; this is the audit, done against the tool vocabulary rather
than against the leaderboard.

## The number that should be read first

> *"Most of the individual XGBoost models already achieve around **0.105** on the
> private leaderboard — those individual models are simpler in the sense that no
> meta model ensembling is involved."*

| | private | rank of ~3,300 |
|---|---|---|
| his full ensemble | 0.10021 | ~1 |
| **one of his models, no ensembling** | **~0.105** | **~3** |
| private median | 0.12967 | ~1,648 |
| our v7 blend | 0.13813 | ~1,901 |

**His ensembling was worth 0.005.** The other 0.033 — the whole distance from our
blend to his simplest model — is in a single model. This project has spent 0044,
0045 and part of 0068 on the blend, which is the smallest lever in his account. His
Figure 2 makes the same point about features: xgboost without feature engineering
was "already quite good", and full feature engineering was "really about
finetuning".

## What we have

Everything in his `features` dict that is a raw store attribute, a date part or a
day counter — `day`, `dayofweek`, `dayofyear`, `month`, `week`, `year`,
`daysincompetition`, `daysinpromocycle`, `promointerval` — is expressible with
`transform_features`, and v7's `t13` built all of them.

His whole `shopavg_*` / `shopsales_*` family (average sales per customer, ratio of
sales during promotions, holidays, Saturdays) is `fit_group_stats` and
`fit_encoding`, and `t13`'s eight `_te` columns are the same idea with the periodic
keys 0046 added. **On that family we are ahead of him**, not behind.

His `prev{quarter,halfyear,year}_{ds,dps,dphs}_*` aggregates are
`fit_history(by=[...], windows=[[w, stat]])`, with his keys — day/store,
day/promo/store, day/promo/schoolholiday/store — all expressible as `by`.

## What we are missing

**1. Per-group trend extrapolation.** His `linmod_quarterly` and `linmod_yearly`:
a per-store Ridge on (day number, day of week, promotions), fitted on the last
quarter or year and **extrapolated into the forecast window**. There is no tool for
this and no combination of existing tools expresses it. It names no dataset — a
gradient-boosted tree cannot extrapolate a trend beyond the range it was trained
on, which is a property of the model family, not of retail sales. This is 0058's
item 2.

> **Corrected before this record was first committed.** The paragraph above
> originally ended "and it is the largest unbuilt item on this list", which his own
> feature dict contradicts. The four bits are
> `[salesmodel, customermodel, MA-models, pmd-variant of MA-models]`, and both
> entries read `[0,0,0,1]` — **the trend features are not in his main model at
> all.** They appear only in the previous-month-date variant of the month-ahead
> models, the branch he uses for the *second* month of the test window and
> describes as "almost as good" rather than better.
>
> The misreading is recorded rather than quietly fixed because it is the same
> failure as 0069's, twice in one day: taking a name from the writeup and ranking
> it without reading what the document says about where it sits. A four-bit vector
> per feature is the most precise information in the whole document and it was
> skimmed.

## What his main model actually contains

Filtering the dict on bit 0 is worth doing before ranking anything else. The
salesmodel is: store attributes, date parts, promotion-cycle counters, competition
features, holidays (including `holidays_last/this/nextweek`), weather,
`closurefeat` — and from the entire recent-data family **exactly two** columns,
`prevquarter_med` and `prevyear_dps_med`. Most of the `prev*` family sits in the
pmd variant, and the MA models carry none of it at all.

And one entry contradicts a mechanism this project has measured:

    'store':[0,0,1,0]

**His main model has no store identity.** He says so directly: *"at some point I
excluded store from my model, and was worried that daysincompetition and
competitiondistance would be used as proxies for store, therefore I rounded them on
the log scale."* The store reaches his model as *aggregates*, not as an identifier.

0054 measured per-entity identity as the mechanism that makes capacity pay at all —
restaurant's rows went from +0.0004 to +0.0283 once the model could tell one shop
from another. The winner went the other way. Both can be true (his aggregates carry
the same information in a form a small model can use), but it is a direct tension
with the direction this project has been taking, and it is not resolved here.

**2. Distribution shape beyond mean, standard deviation and median.**
`WINDOW_STATS` is `(mean, std, median)`. He used skewness, kurtosis, harmonic mean
and percentiles, and `prevyear_ds_p10` survived into his final model. Carries his
own warning — *"the spread features made overfitting easy"* — which is precisely
the column family 0069 identifies as the one feature selection exists to remove.
The two items belong together: adding unstable columns without the instrument to
drop them again is how the model "easily overfit the train set".

**3. Forward-looking calendar windows.** `holidays_lastweek`, `holidays_thisweek`,
`holidays_nextweek`. These need an exogenous calendar that is **known in advance**,
aggregated over a window that reaches *forward* from each row. `fit_history` reaches
backward only; `fit_stream_aggregate` restricts to what was knowable a lead time
ahead, which is the opposite restriction. The general form is "some second inputs
are known for the future at prediction time, and the harness has no way to say so".

**4. Feature subset selection** — built and measured at approximately zero in this
form; see 0069, which also explains why the obvious framing of it was wrong.

## What we can express and throw away

`fit_group_stats`'s docstring states that the value column *"does not have to exist
at predict time; only the `by` columns"*. His `shopavg_salespercustomer` is exactly
that: an aggregate of a train-only column, fitted once and replayed. **Every
Rossmann run this project has made drops `Customers` entirely.** The per-store
aggregate of it is serveable and no run has ever built it.

## What the document says to leave alone

- **Month-ahead models: 0064's refusal is vindicated.** He reports them "almost as
  good" as the models using recent data, never better, and reads that as the
  process being stationary. Not a lever.
- **External weather** stays out (0001), as it always has.
- **The 0.995 / 0.985 prediction multiplier** stays out, and note that his own text
  regrets not using 0.985 — a post-hoc constant fitted to the leaderboard is what
  0022 and `CLAUDE.md` exist to refuse, and being right about it does not change
  that.
- **His holdout discipline is weaker than ours, not stronger.** He selected among
  ~125,000 model pairs on his last six weeks with no sealed split, and reports being
  surprised it held. His method is not licence to loosen 0023.

## The decision

Nothing is built here, and after the correction above this record deliberately
**does not rank the four capabilities**. Two rankings were attempted today — mine
of feature selection (0069) and mine of trend extrapolation — and both were wrong,
both for the same reason: a lever's size was guessed from a description instead of
measured.

The gap to beat is 0.0204 of rmspe to the private top 10% and 0.0085 to the median.
Every lever this project has *measured* is 0.0037 to 0.0068. Guessing which
unbuilt capability is worth 0.02 has now failed twice, so the next step should be a
measurement rather than a third guess:

> **Rebuild his salesmodel feature set by hand in `scripts/`, fit one model at a
> real budget, and score it on our sealed holdout.** `scripts/prep_*.py` is where
> `CLAUDE.md` sanctions dataset knowledge, and the *output* of the probe is the
> general question — which capabilities `src/` actually needs.

It is decisive in both directions. If it lands near his ~0.105, the gap is
vocabulary and we learn exactly which columns carry it, measured rather than
ranked. If it lands near our 0.138, the gap is **not** vocabulary, and this entire
record — every capability listed above — is beside the point. That second outcome
would be the most valuable negative result available to this project, and nothing
currently planned would produce it.

What is settled: feature selection is not next (0069 measured this project's
version at zero), and capacity is not next (0068 measured +0.0037 on the serving
path and refuted the two obvious ways to choose a budget).

## What this audit cost, and what it is worth

It was cheap and it should have happened two weeks ago. `HANDOFF.md` listed
"distill a winner writeup" behind three sessions of feature vocabulary, and the
single most useful fact in the document — that one un-ensembled model reaches rank
3 — reframes which part of the pipeline is worth working on. **A competitor's
writeup is an instrument for finding out what the harness cannot say, and this
project treated it as optional reading.**
