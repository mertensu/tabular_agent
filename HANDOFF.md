# Handoff

> Session log for working across two machines. Newest entry on top.
> Update before ending a session, then commit & push. Pull and read this first when starting.
>
> **This file is pruned (~2 weeks) and is not the record of *why* anything was
> decided.** That lives in `docs/decisions/` — see its README for the index and
> the project vocabulary. When an entry here records a decision, write the
> decision record too, or the reasoning is deleted with the entry.

## 2026-08-15 (later) — Restaurant v1 ran and was scored; the seal is a lower bound

**Machine:** MacBook Air | **Branch:** master.
Written up as [0060](docs/decisions/0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md).
**Next session: 0057 — but follow [0058](docs/decisions/0058-the-ranking-failure-was-one-feature-not-the-block.md)'s
revised order, not 0057's own.** 0057 step 1 says "fix the ranking block first";
0058 withdrew that and 0059 built the interval instead. So: **lead-time-restricted
aggregates first, then per-group trend extrapolation.**

`scripts/run_restaurant_v1.py`, Rossmann v8's config verbatim. 17 tables, 5 terminal
fits, 21.9 min, agent finalized itself. Five predictions registered before the run.

### The measurement, and the ordering is the finding

| | rmsle |
|---|---|
| validation (lgbm, selected) | 0.52099 |
| **sealed holdout, blend — shipped** | **0.50176** |
| Kaggle public (nearer half) | 0.51301 |
| **Kaggle private (further half)** | **0.54398** |

Val→sealed optimism was **negative on all five models** (−0.012 to −0.026), which
reads as "the exam is honest" — the same signature v5 produced on Rossmann. It was
not. **sealed→private is +0.042.** Four windows in time order, each fair about
itself, each flattering the next.

> **A sealed score is a lower bound on the error at the horizon you will be graded
> at, not an estimate of it.**

Ranks, both boards (2,150 teams): public ~1544 (top 72%), **private ~1369 (top
64%), only +0.0118 above the median** (0.53223). For scale, 0054 measured
store-identity + capacity on this dataset at **+0.0283** — one already-measured
mechanism clears both median and top 10%.

### A mechanism refuted, and only the private board could do it

Prediction 2 said Kaggle would punish more at the far edge. Direction right (+0.031
public→private). **Mechanism wrong:** joining both boards on team, the field's mean
private−public is **+0.04464** against our **+0.03097**. Everyone decays with
horizon and **we decay less than average** — our rank *improves* 72%→64%. So the
decay says nothing about the cut reservation stream it was predicted to expose. A
control run with those columns withheld is one script, not yet written.

Read against the public board alone the story was available and wrong.

### The other predictions

- **1 confirmed** — 6 reservation features in the shipped model (67 total).
- **3 confirmed** — **6 of 17 comparisons `sign_stable: False`**, the block mode's
  first live firing. Every unstable table has |gain| < 0.003 against a block sd 2-5x
  larger; two negative. The four real steps (+0.190, +0.034, +0.032, +0.013) are all
  5/5 stable. It separated signal from tail cleanly.
- **4 untested, still open** — the agent built 90/180/365-*day* windows on a
  **ragged** spine where `fit_history` counts **rows**. Nothing caught it, nothing
  measured it. Real gap in the tool.
- **5 confirmed** — below median.

### Unpredicted: the agent used the new instrument

Stopped at 17 of 25 permitted tables, saying gains were *"below the probe's own
standard error and not cross-block-stable … so I stopped rather than overfit."*
0059 was eight hours old. **Patience cannot explain it** — on an ordered split
`probe_se` is `None`, so the bar floors at zero and every positive gain still counts
as progress. Weigh against 0027 (three advisory steers that changed nothing). n=1.

Also: the agent read the sealed score post-finalize and narrated it. Known, open
since 0042.

### Not done, deliberately

Submission **not clipped** — one prediction of 32,019 came out at 2868.5 against a
training max of 877 (~0.0015 of rmsle). Flooring it after seeing the score is the
correction 0022 exists to refuse. A general "predictions outside the training
range" diagnostic would be legitimate and names no dataset.

### Also this session

`docs/decisions/README.md` indexed; HANDOFF pruned to the ~2-week rule and the
08-05..08-13 entries compressed to summaries (full text at `801b52d:HANDOFF.md`).

---

## 2026-08-15 — the ruler is fixed, and it would not have caught the thing it was built for

**Machine:** MacBook Air | **Branch:** master | **479 tests pass** (was 456).
Written up as [0059](docs/decisions/0059-the-interval-belongs-to-the-difference-not-the-score.md).
0036's paired bootstrap is **built** — owed since 2026-08-13 and named the blocking
item in four entries.

**Priority call:** 0058's list is capabilities; this is the instrument that measures
them. Seven feature readings had been decided on differences with no interval, and
0052/0053/0058 are a trail of mechanisms withdrawn afterwards. Lead-time-restricted
aggregates (0058 item 1) is still next; it now has a yardstick to be measured with.

### What was built

`src/tabular_agent/compare.py`. Every table-vs-parent comparison now carries an
interval around the *difference*, not two point scores against a pooled `val_se`.

- **Random split → paired bootstrap over rows.** One resample, both candidates
  scored on it. That requires both prediction vectors, so `probe_table` now stores
  each table's validation predictions (`run_dir/probe_preds/<id>.npy`, one float64
  vector, 400KB on Rossmann, bounded by `max_tables`).
- **Ordered split → the difference recomputed across consecutive blocks.** A row
  bootstrap is refused there for the reason `val_se` already was: the rows are one
  *period*, and resampling inside it cannot see which period it is.
- **The mode is chosen by passing the ordering column, not a flag** — a flag lets a
  caller say "ordered" without supplying the order, the one case that cannot be
  served.
- `_score_on_held_out` returns a 4-tuple now (predictions alongside the score); six
  call sites unpack one more value.

### The bug the design was written around

**The validation half is not stored in its ordering column's order.** `RunSession`
picks the val positions in time order then writes them with `np.sort(val_pos)` —
which sorts the *positions*, so the file comes back in the input's row order.
Rossmann's raw file is **reverse** chronological (`ship_rossmann_median.py` was
already bitten by this). Blocking on file order would have cut five arbitrary
slices and reported a reassuring number. The ordering column is a required argument
and the sort happens inside the instrument.

### The measurement, and my prediction half failed

`scripts/probe_difference_interval.py`, against the history arm Kaggle already
refuted (t9 vs t8 in `_runs/ship_rossmann_median`; **−0.0301 private**).

| instrument | reading |
|---|---|
| pooled gain, t9 over t8 | **+0.02028** |
| paired bootstrap over rows | se **0.00074** → **gain/se 27.3** |
| pairing ratio | **0.371** |
| blocks, 44-day val window | sd **0.01214**, `sign_stable` **True**, 5/5 |
| blocks, 90-day window (val+holdout) | sd **0.01670**, `sign_stable` **False**, 8/10 |
| Kaggle, Aug–Sep | **−0.0301** |

1. **The row bootstrap is confidently wrong** — 27 standard errors from zero for a
   difference that reverses externally.
2. **0036's pairing ratio replicated**: 0.371 inside its 0.18–0.37, on a second
   dataset, a different metric, 50,175 val rows against 292. 0036 named that as the
   thing it could not establish.
3. **Blocking helped 16x and was still not enough.** `gain/se` 27.3 → 1.7, but all
   five blocks still favoured the history table. **A run would not have caught it.**
   Registered prediction 2 said the blocks would disagree; on the window the harness
   actually uses, they did not.
4. **The window was the problem, not the block count.** At 90 days the blocks
   disagree, the pooled gain nearly halves, and the decay is direct: first five
   blocks **+0.0200**, last five **+0.0022**. Kaggle's window starts two weeks later
   and reads −0.0301 — the same line continued.

So the limitation `compare.py` states in its own docstring *before* it was measured
is the one that bit: blocking samples the periods it has. More blocks over the same
44 days would not help; reaching further out would, and that is 0043's trade-off,
not this instrument's.

### Deliberately not done

**`min_gain_ratio` is unchanged at 0.25.** 0028 left the magnitude open and 0036
proposed this as a candidate, but that is one comparison on one dataset — picking a
multiple of the block sd from it is 0034's refusal verbatim. The interval is
reported; nothing is gated on it.

Also: `tests/test_compare.py` (15), `tests/test_compare_harness.py` (8),
`notebooks/phase5_the_interval_on_a_difference.ipynb` (35 cells, 14 code, ships
unexecuted, every code cell smoke-tested in order).

### Still owed

**Quizzes**: this chunk (asked, unanswered), slice 3, 0041/0042/0044, house-prices
capacity. `scripts/run_rossmann.py` still broken since 0035. The request timeout is
still only patched in one driver.

---

## 2026-08-14 (night, newest) — four submissions, and the ruler is the problem

**Machine:** MacBook Air | **Branch:** master.
[0056](docs/decisions/0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
(the measurements) and
[0057](docs/decisions/0057-the-capabilities-the-agent-does-not-have.md)
(what to build, nothing built yet).

| arm | val | holdout | Kaggle priv | Kaggle pub |
|---|---|---|---|---|
| bare block, 200 trees | 0.23331 | 0.24228 | 0.23048 | 0.23420 |
| bare block, 4,144 trees | 0.16000 | 0.15766 | **0.15166** | **0.14853** |
| full pipeline + history, blend | **0.13276** | 0.17678 | 0.18438 | 0.17444 |
| full pipeline no history, blend | 0.14793 | 0.15841 | 0.15429 | 0.16193 |

1. **Capacity confirmed externally**: +0.0788 private against a predicted +0.0846.
2. **`val` ranks at Spearman +0.20 against Kaggle; the sealed holdout ranks at
   +1.00.** The holdout is the six weeks adjacent to the test window. **We rank
   every table on the wrong block** — 0043's trade-off priced the other way round.
3. **History refuted externally**, −0.0301. Fifth negative, first external. The
   competition winner independently reports recent data barely mattering to him.
4. **The encoding pipeline is worth nothing once the budget is right** — one lgbm
   on a bare feature block beats the three-way blended pipeline.

**We did not beat v7 (0.13813 private / 0.12932 public) and are not near the
median (0.11677).** And an error worth not repeating: the rebuild targeted v8's
chain because `t24` had the best *probe* score, but v7 is the better run — the
rebuild target was chosen with the instrument 0056 shows is unreliable.

**Finding 2 was then withdrawn the same night — see
[0058](docs/decisions/0058-the-ranking-failure-was-one-feature-not-the-block.md).**
Ulf asked whether it was Rossmann-specific. It was: drop the one history arm and
`val` ranks fine (its only error is a pair separated by 0.0026); that arm decays
with distance from its 2015-05-01 freeze, which is 0049/0050 not a block problem;
`val`'s window is the odd one on *composition* (20.8% closed vs test's 14.6%, a
May-holidays accident) rather than on recency; and a perfect ordering of four
happens 1 time in 24 by chance. **Findings 1, 3 and 4 stand** — they rest on
differences 10-30x larger.

**Next, in this order** (0058's revision of 0057):
1. **Lead-time-restricted aggregates** — the only capability found independently
   twice, and `restaurant_visitor` already carries a natural experiment for it.
2. **Per-group trend extrapolation** — justified by GBDTs not extrapolating at all,
   which names no dataset.
3. Two *instruments*, when a run needs them, not pre-emptively: a composition
   diagnostic comparing the evaluation block against `serving_columns`, and
   ranking stability across several ordered blocks (which would have caught the
   history arm with no external seal).
4. `fit_history` should leave the vocabulary in its current form.

**Do not** swap which block ranks, and do not hardcode a block choice. 0043 stands.

Not done: 0036's paired bootstrap, still. Every number above is a difference with
no interval. Quizzes still owed.

---

## 2026-08-14 (evening) — the capacity plan was run, and it was wrong

**Machine:** MacBook Air | **Branch:** master.
Written up as [0054](docs/decisions/0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
and [0055](docs/decisions/0055-a-rename-makes-every-earlier-artifact-unreplayable.md).
**The START HERE plan below is now superseded on its main point — read this first.**

Ulf: *prepare Recruit Restaurant as a new dataset, then check the n_estimators
hypothesis everywhere, it is most probably wrong.* Both done, and he was right.

### New dataset: Recruit Restaurant Visitor Forecasting

`scripts/prep_restaurant.py`, `notebooks/phase5_restaurant_dataset.ipynb`,
public leaderboard in `restaurant_visitor/` with `leaderboard_rank()`.
252,108 store-days, 829 stores, rmsle, ordered on `visit_date`, 5%/5% like
Rossmann. Six files joined; two of them at a finer grain, which is the join the
harness has no tool for.

**It came with a trap worth the dataset.** The reservation stream is cut at the
moment the forecast is issued, so 30% of *training* reservations were booked
same-day while **no** forecast-window row can have one — and the floor grows to 39
days at the far edge. No guard we have sees it: no fold crossed, no encoder
misfitted, column present at serve time, and validation *confirms* it because
validation shares the uncut stream. Kept and documented, per the `Customers`
precedent (0029). The general capability it asks for — *aggregate a related stream
restricted to what was knowable a given lead time ahead* — does not exist.

Also: the spine is **ragged** (a store has no row on days it did not report), so
"last 364 rows" ≠ "last 364 days" here. Every window feature from 0052/0053
assumed they were the same, because on Rossmann they are.

### The capacity hypothesis is refuted; the mechanism is identity, not rows

`scripts/probe_capacity.py`, `_runs/capacity_probe.csv`,
`notebooks/phase5_model_capacity.ipynb`. Prediction registered before running.

| dataset | rows | best fixed budget | early stop |
|---|---|---|---|
| house-prices | 1,168 | −0.0034 (all lose) | 162 → **+0.0062** |
| restaurant | 227,139 | +0.0004 | 1,018 → +0.0003 |
| Porto Seguro | 476,170 | −0.0188 (all but one lose) | 181 → **+0.0049** |
| Rossmann | 915,744 | **+0.0573** | 16,926 → **+0.0606** |

Restaurant at 227k gains nothing, so it is not rows. The confound — nearly written
up as the conclusion — is that Rossmann's `Store` is an *integer* and survives
`_feature_frame` while restaurant's `air_store_id` is a *string* and is dropped.
Give restaurant an arbitrary integer store code and the same rows go from **+0.0004
to +0.0283**. It is per-entity effects: 200 trees cannot reach 829 shop effects.

### The thing nobody was looking for

On Rossmann, v8's own feature block measured against the same split:

| rmspe | baseline | + feature block | features worth |
|---|---|---|---|
| **200 trees (today)** | 0.21810 | 0.23331 | **−0.0152** |
| 5000 @ lr 0.03 | 0.16083 | 0.15213 | **+0.0087** |

**The sign reverses.** The ruler that ranks every table in every run has been
changing the sign of what it measures. `best_table` is that ruler's argmax.

### What to do next, in order

1. **Do not raise the constant.** Early stopping asked for 162 / 181 / 1018 / 3430
   / 16926 — two orders of magnitude. Any constant big enough for Rossmann wrecks
   house and Seguro; picking Rossmann's is a leaderboard-fit (0022, one level up).
2. **Design where the early-stopping block comes from.** This is the open decision.
   The arm here carves the tail 10% off the work half — fine for an ordered split,
   needs an argument otherwise, and it touches 0023/0029/0041/0043. **Not** an
   argument for giving params back to the agent (0035).
3. **Re-read `_runs/` with suspicion.** Every table comparison was made at 200
   trees. 0052 and 0053 are about features measured by this ruler.
4. **0055 first if anything is to be shipped:** v8's tables cannot be replayed at
   HEAD. Cheapest useful repair is a replay test against a checked-in artifact, so
   the *next* rename is loud at the commit rather than four hours later by luck.

### Owed, unchanged

0036's paired bootstrap still not wired in — every number above is still a
difference with no interval around it. Quizzes owed: slice 3, 0041/0042/0044, and
the one asked this session (house-prices capacity, unanswered).
`scripts/run_rossmann.py` still broken since 0035.

---

## SUPERSEDED — the old plan, kept only so its refutation is legible

> **Do not follow this section.** Its central claim — that model capacity matters
> in proportion to *rows* — was tested on 2026-08-14 and **refuted**
> ([0054](docs/decisions/0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)):
> a 227k-row dataset gained nothing while a 916k-row one gained 0.057, and the
> real mechanism is high-cardinality *identity*, not size. Its second guess — that
> capacity would dwarf the feature search — turned out to be right for a different
> reason than it gives.
>
> **The current plan is the newest entry at the top of this file.** This block is
> left intact rather than deleted because the decision records cite it as what we
> believed beforehand, and rewriting it would erase the prediction that was made
> before the measurement.

Written 2026-08-14 at Ulf's request, so a fresh context can begin without
re-deriving anything. Read this, then `docs/decisions/README.md`, then the entries
below for detail.

### Where we actually are

Sixteen runs, two datasets. **Rossmann's best is ~rank 2,200/3,303 — below
median.** House-prices reached top 9%, Porto Seguro top 57%. The harness is sound
(leakage-safe, sealed holdout, replayable bundles) and the *scores* are mediocre
on the forecasting task. Ulf: *"somehow we need to get better than the median, do
not know what we miss still."*

### My best guess at what we are missing, and it is not features

**Capacity, not vocabulary.** Every fit in this harness runs at a **fixed, modest
budget that nothing ever tunes**. Checked, so the next session does not have to:

    probe.py:110      PROBE_PARAMS = {"n_estimators": 200}
    models.py:419-432 lgbm / xgb / catboost defaults: n_estimators=200
    learning_rate     never set anywhere — whatever the library ships (lgbm: 0.1)

So it is 200 trees at lr 0.1, on 850k rows, for every model in every run. That is
not absurd, but competitive solutions on a dataset this size run thousands of
trees at a low learning rate, and **the harness cannot express that at all** —
there is no tuning step, no early stopping, and 0035 removed the agent's ability
to touch model params on purpose.

This is not a new idea — **0039 named it and left it open deliberately**: *"the
terminal phase enumerates library defaults, so it measures the libraries. A fair
comparison needs a common budget or early stopping — left open."* It has been
open since. Meanwhile three sessions went into feature vocabulary, and 0052/0053
just spent a whole slice finding that the newest feature is worth less than one
we already had.

**The cheapest decisive test, and it should be the first thing next session:**
take v8's best table and fit the probe estimator twice — the current 200 trees at
lr 0.1 versus a real budget (lr 0.03, a few thousand trees, early stopping on the
val block) — and read the difference. Register the prediction first. If capacity
is worth more than ~0.02 rmspe it dwarfs everything the feature search has ever
bought and the project's priorities are wrong.

It also explains the *pattern* of our results: house-prices (1,168 rows) reached
top 9% where 200 trees is roughly right, while the two large datasets (850k, 476k)
land mid-table — which is exactly where a fixed small budget would bite.

It is also **general** — "the harness never chooses model capacity" names no
dataset, which is the test CLAUDE.md sets.

Caveat to respect: 0035 removed the model tools deliberately, and 0029 says the
harness owns the exam. **This is not an argument for handing hyperparameters back
to the agent** — it is an argument for the terminal phase fitting each family at a
defensible budget instead of at whatever the library ships.

### `min_periods` — worth building, as an instrument

Ulf asked. The literature's standard answer to "rows that cannot fill a window"
is `min_periods` (skforecast's `RollingFeatures` drops such rows; a
wildfire-forecasting paper excludes its first three years for a 36-month window).

**Do not ship it as a guard.** 0053 showed `w=728` has identical count
distributions at `n_splits=3` and `5` with opposite outcomes, so a threshold on
counts cannot repair a difference that exists between two same-count
configurations. That would be the fourth guard this session built on an unverified
story.

**Do build it as a discriminating experiment.** Raggedness — the per-fold ramp
20/39/59/78/93 — is the last untested candidate mechanism, and `min_periods` is
exactly the knob that removes it. Set it high, re-run the n=3/n=5 pair:

- gap collapses ⇒ raggedness is the mechanism, and `min_periods` becomes a fix
  *on evidence*;
- gap survives ⇒ raggedness is dead too, and the cause is elsewhere.

Implementation is nearly free: NaN the statistic where `_count < min_periods`.
Blank rather than drop rows — a harness table's row count is load-bearing, and
NaN is already a first-class "no estimate" (0046/0047).

### A second forecasting dataset — yes, and pick one that breaks assumptions

Rossmann's private LB is **spent** for the blend-membership question (0045), and
every conclusion about ordered splits rests on one dataset. Candidates, with what
each would stress that Rossmann does not:

- **M5 / Walmart (hierarchical, intermittent).** Many series, lots of true zeros,
  hierarchical aggregation. Would stress everything this harness assumes about a
  single flat table. Heaviest lift, and its metric is complex — mind 0022, the
  yardstick must measure the ranked quantity.
- **Corporación Favorita.** Closest to Rossmann in shape, much bigger. Best for
  *comparability* — a second reading of the same claims — worst for finding new
  gaps.
- **Recruit Restaurant Visitor Forecasting.** Genuine multi-table join, different
  horizon, moderate size. A good middle option.

**My recommendation: Recruit Restaurant or Favorita**, because the immediate
question is "why are we below median on a task we understand", and answering it
needs a *comparable* second reading, not a harder problem. Save M5 for when the
capacity question is settled.

Whichever it is: `scripts/prep_*.py` holds the join, typing, metric and split —
nothing dataset-specific enters `src/` (CLAUDE.md).

### Distilling a winner writeup — the guardrail

Ulf wants to re-read the Rossmann winner (Gert Jacobusse) and distill. That is
sanctioned design input — it is where slice 3's argument came from — but the
distillation has to come out as **capabilities stated without naming the dataset**,
or it is leaderboard-fitting one level up (0022, CLAUDE.md).

The test that already worked once: *"per-group aggregates of a column, fitted once
and replayed"* passes; *"exclude rows where the store was closed"* does not — its
general form is *"some rows should not be trained on"*, and **that** is what gets
built.

Questions worth putting to a writeup, none of which name a dataset:

- What did they do about **model capacity and early stopping**? (Suspected biggest
  gap — see above.)
- Did they **weight or filter training rows** (recency weighting, dropping
  structurally-zero rows)? The harness has `filter_rows` but no weighting at all.
- How did they handle the **target transform** and any post-hoc correction?
  (0021 owns the transform; the 0.985 multiplier stays out — its own author reports
  the effect was inside CV noise.)
- Did they fit **one model or many** (per segment)? The harness can only fit one.
- What did their **validation** look like? Ours is one period with no error bar,
  which 0036 was supposed to fix.

### The blocking item behind all of it

**0036's paired bootstrap is still not wired in.** Ordered split ⇒ val is one
period, `val_se` is `None`, n=1 in the unit that varies. 0052 and 0053 are both
built on differences nobody can put an interval around, and the capacity
experiment above will produce another one. Fix the ruler before trusting the next
number.

### Also owed

- **Quizzes**: slice 3 (one question asked and answered), plus 0041/0042/0044 from
  earlier sessions.
- `scripts/run_rossmann.py` broken since 0035 (`StopPolicy(max_experiments=)`
  renamed to `max_tables`).
- Request timeout still only patched in one driver, not lifted into the library.
- `_latest` still emitted unconditionally by `fit_history`, so every history table
  carries the pair that destroyed v8's t7.
- `to_periods` turns every date into days; a second-resolution log is unusable.
  Fix is a *declared* unit (0051), not an inferred one.

---

## 2026-08-14 (night, last) — Ulf's question refuted 0052's mechanism

**Machine:** MacBook Air | **Branch:** master | suite green.
Written up as [0053](docs/decisions/0053-the-fold-count-moves-a-window-and-we-cannot-say-why.md).

Ulf, reading 0052: *"but the 136 for rossmann are only due to how we split the
data into folds no?"* Yes — fold 0's prefix is `span/(n_splits+1)`. Testing it
killed the explanation.

Same sweep at `n_splits=3` and `5`, everything else identical, both reproducing
to the digit:

| feature | n=5 | n=3 | swing |
|---|---|---|---|
| all-time mean of `Store_DOW` (0046) | +0.06048 | +0.06004 | **0.0004** |
| window 364 | +0.04458 | +0.05338 | 0.0088 |
| window 546 | −0.00030 | +0.04078 | 0.0411 |
| window 728 | **−0.09287** | **+0.04654** | **0.1394** |

**w=728's train/serve counts are identical in both** (median 59 against a served
101, 1.71x). The quantity 0052 blamed did not move while the outcome reversed.
Across pairs the ratio is *anti*-correlated with the score, and the two-level
guard I had just built fires in **both** configurations — 728 exceeds every prefix
at n=3 too (204/408/612). A guard that fires on the good case and the ruinous one
alike is uninformative.

**What survives.** 0052's measurements (all reproduced) and its conclusion: the
all-time mean beats every window under *both* fold counts, so slice 3 still buys
nothing over 0046. And a new statement about the feature class —

> a trailing window's measured worth is substantially an artifact of the fold
> count, sensitivity growing with length, while the unbounded aggregate it loses
> to is invariant.

So **window length cannot be searched over here and the number believed** (0022's
problem, arriving in a feature instead of a metric).

**No replacement mechanism is offered**, deliberately. Candidates considered and
untested: fold-dependent NaN share (25% at n=3 vs 17% at n=5), number of coverage
regimes in the trained column, interaction of the ramp with the target's time
trend. **Third failed story about these columns in one session** (0050's
`summarise_history` verdict, 0052's coverage count, 0052's prefix mechanism).

**Code:** prefixes still reported (the agent cannot get them otherwise), warnings
demoted to notes that describe the geometry and explicitly decline to predict a
cost.

**From the literature** (searched at Ulf's request): the two standard answers are
fixed-size sliding-window CV (`TimeSeriesSplit(max_train_size=...)`, skforecast's
"fixed training size" backtesting) and **`min_periods` — skforecast's
`RollingFeatures` drops training rows that cannot fill the window**, and a
wildfire-forecasting paper excludes its first three years for a 36-month window.
`min_periods` is the better fit here and is nearly free (NaN where `_count <
min`), but **not built**: 0053 says the mechanism is unknown, and building a fix
for an unidentified failure is how the last two guards got written. Switching CV
schemes stays out too — it changes the exam for every feature and makes all 16
prior runs incomparable, which is 0041's argument for opt-in.

**Next:** 0036's paired bootstrap. Two records now rest on differences nobody can
put an interval around; it is the blocking item for this whole area.

## 2026-08-14 (night, latest) — slice 3 built, and the measurement says it lost

**Machine:** MacBook Air | **Branch:** master | **suite green.**
Written up as [0052](docs/decisions/0052-a-window-is-a-worse-estimator-of-a-better-idea.md).

Windows are in `fit_history(windows=[[364, "mean"]])`. Row-anchored `[row - w, row)`,
strict right endpoint, `_count` per length, `WINDOW_STATS = (mean, std, median)`.
Then three hand-driven Rossmann probes said it was not worth building.

### The result, in the order it has to be read

**1. The key is the gain, not the averaging.** Same tool, same 364 days, same stat:

| key | probe | vs baseline |
|---|---|---|
| `Store` | 0.22072 | −0.00263 |
| `Store`+`DayOfWeek` | 0.17352 | **+0.04458** |

**2. And that +0.045 is mostly 0046, not slice 3.** The control nobody had run:

| | probe | vs baseline |
|---|---|---|
| **all-time target mean of `Store_DOW` (0046)** | **0.15762** | **+0.06048** |
| trailing 364 mean, `Store`+`DayOfWeek` (slice 3) | 0.17352 | +0.04458 |
| both | 0.16944 | +0.04866 |

**Slice 3 bought −0.01590 over what 0046 already had**, and stacking it on top of
the all-time mean is *worse than the all-time mean alone*. The winner's *"sales
for this store on a typical Monday over the last 6 months"* is two halves and
**the half we were missing was the periodic key** — built in 0046, paid for in v7.

`docs/slice3_windows.md` pre-registered exactly this: "if a window adds nothing,
that is a publishable result and the slice should stop."

### The new failure mode, and it is general

Length sweep at a fixed key. Registered prediction was **monotone**; wrong:

| window | probe | obs |
|---|---|---|
| 91 | 0.23365 | 3 |
| **364** | **0.17352** | 39 |
| 546 | 0.21840 | 58 |
| 728 | 0.31097 | **59** |

546 and 728 hold the *same* number of observations and are 0.093 apart, so
coverage is not the mechanism. It is:

> **A window longer than the shortest fold's training prefix cannot be filled in
> training, while a served row fills it completely.**

Work half 850 days, fold 0's prefix **136**. So w=728 trained at median 59 obs and
served at **101** (1.71x); w=364 ran 39 against 49 (1.26x). Train/serve skew
through a door 0048 did not cover — not *when* history stops but *how much* there
was to summarise.

### The gotcha, and it is mine again

**`window_observations`, added this session to catch a badly-chosen window, rated
the catastrophic one higher than the good one** (59 vs 39). The damage is in the
*ratio between the training and serving sides* and the count only sees one side.
0031's shape in the reporting layer for the second time in three records (0050
caught it in `summarise_history`).

Fixed generally: `fit_history` now returns **`shortest_training_prefix`** and warns
when a window exceeds it, saying outright that the count is no guide there.

**The guard over-fires and says so.** Rossmann's shortest prefix is 136, so the
warning flags **364 — the best window in the sweep** — alongside the ruinous 728.
The prefix marks where the mechanism starts, not where it starts costing. Warning
on the *ratio* instead needs a threshold, and there are two points on that curve
(1.26x fine, 1.71x ruinous); picking a number between them is 0034's refusal to
fit a constant to three observations. Conservative and honest about it, rather
than tuned and sharper-looking.

Also: the training-side clipping is **not uniform** — fold 0 holds ≤19 weekly obs
in a 728 window, fold 4 up to ~101, serving gives 97–104 to everyone. The trained
column is a *mixture* of differently-truncated windows under one name. Less data
everywhere would be a noisier feature; differing amounts across the seam is a
different feature.

### Decisions taken with Ulf

- **Keep the code, record the loss.** One dataset is one reading, and Rossmann is
  unusually stable per store-weekday — the feature was tested where it had least
  to offer. A negative result that stays retestable is worth more than one that
  does not.
- **Keep `median`, explicitly unvalidated.** 0.0003 from the mean on the good key,
  worse on `Store`. Dropping a stat because one dataset did not need it is the
  mirror of adding one because it did.

### Carry forward

- **None of these gaps has a yardstick.** Ordered split ⇒ val is one period,
  `val_se` is `None`, n=1 in the unit that varies (0022/0036/0041). Directions are
  consistent across three comparisons; the numbers have no error bars. **0036's
  paired bootstrap is still not wired in and is what would fix this.**
- **Offsets `[364,728]` scored −0.0229 here against +0.0217 in v8.** Likely because
  v8's agent had filtered `Open == 1` first, so a 364-day lookup could not land on
  a closed day. Plausible, untested, named so it is not later assumed.
- **Unexplained:** on the `Store` key, offsets alone −0.0229 and window alone
  −0.0026 combined to **+0.0458**. Recorded as an oddity, not narrated.
- Scripts: `probe_rossmann_windows.py`, `..._windows_control.py`,
  `..._window_length.py`. All register predictions in their docstrings.
- **Notebook** `phase5_history_and_the_horizon.ipynb` now 55 cells, sections 6–11,
  ships unexecuted, every code cell smoke-tested in order.
- **Quiz for slice 3 still owed** (0041/0042/0044's too).

## 2026-08-14 (night, later) — naming changed, flag replaces lag age, slice 3 planned

**Machine:** MacBook Air | **Branch:** master | **426 tests pass.**

Acting on 0050 plus Ulf's review of the column names.

### Columns renamed (breaking — the v8 bundle no longer replays)

| was | now |
|---|---|
| `<base>_last` | `<base>_latest` |
| `<base>_last_age` | `<base>_latest_ago` |
| `<base>_lag364` | `<base>_364_ago` |
| `<base>_lag364_age` | **`<base>_364_ago_exact`** |

**One word for one idea: `_ago`.** In `<base>_364_ago` the distance is a literal
in the name; in `<base>_latest_ago` it is unknown until you look, so it becomes a
column. Same word, same meaning — one fixed, one measured. ("history" was
considered and rejected: it suggests the column *contains* history, when it holds
a single distance.)

**No unit in the names, deliberately.** Ulf's `_days_old` read better but the
ordering column is not always days — an event log may be in seconds — and a name
saying "days" over a series counted in something else is precisely the
confidently-wrong label this area keeps producing.

`_runs/rossmann_v8/bundle` still stores `feature_names` with the old names, so it
will not replay. Accepted rather than versioned: nothing depends on it.

### The lag age became a flag

For an offset the age is near-constant — it *equals* the offset whenever the
lookup worked — so all its information is in the failures. `_exact` is 1/0/NaN and
carries the same signal legibly. **`_latest_ago` stays a real measured distance**, because there
it genuinely varies per row (1–45 days on Rossmann).

### `fit_history` now reports what the agent could not know

Ulf's catch, and it was a real gap: the warning said *"use offsets longer than the
period you are predicting"* — but the agent **has no way to learn that number**.
`inspect_data` reports `n_val` as a *row count*; nothing converts it to a span. So
the tool computes it and states it:

- **`block_span`** — how wide a scored block is, in the ordering column's units.
- **`offsets_answerable`** — `{"7": 0.112, "364": 0.805}` per offset, plus a
  WARNING in the note below 50%.

That is the line that would have made v8's `lag7` mistake obvious at a glance
instead of after an investigation.

### Known limitation, now documented in `to_periods`

**Dates always become days, whatever their granularity.** Fine for a daily panel,
unusable for a second-resolution log (age `0.00347`, an hour as offset `0.0417`).
Correct arithmetic, unusable numbers. Fix is a *declared* unit, not an inferred
one; a numeric ordering column already sidesteps it. Not built.

**Next: slice 3, in a fresh context — see `docs/slice3_windows.md`**, which carries
the full argument, the design constraints, and the two open questions
(should `_latest` still be unconditional; the time unit).

## 2026-08-14 (night) — Rossmann v8: offsets are the win, `_last` is the disaster

**Machine:** MacBook Air | **Branch:** master | first agent run with `fit_history`.
Written up as [0050](docs/decisions/0050-the-anchor-is-what-matters-not-the-history.md).
25 tables in 10.8 min, `_runs/rossmann_v8/`.

**The four best tables in the run are history tables. So is the worst.** Same tool,
same key, same value — the whole spread is one parameter:

| table | offsets | probe | vs parent |
|---|---|---|---|
| t24 | `[364, 728]` | **0.13669** | +0.0217 |
| t22 | `[364]` | 0.13915 | +0.0192 |
| t20 (best non-history) | | 0.15840 | |
| **t7** | **`[]` last-only** | **0.46396** | **−0.2639** |

**Why: the anchor.** `_last` anchors at the *freeze boundary* — one date shared by
every served row, and one date is never typical. 0049 caught it as zeros. v8 shows
the general form: the agent had already filtered `Open == 1`, so **no zeros at
all**, and `_last` was still ruinous — the last open day before a holiday is a
**pre-holiday spike, median 9,697 vs 6,337 overall (+53%)**. The filter fixed the
symptom, not the disease. An offset anchors at *the row*, so each row draws
independently.

**New rule 0047 lacked:** the declared floor makes an offset *legal*; the **span of
the serving block** makes it *useful*. Val block is 44 days, so `lag7` was actually
honoured for **11.2%** of val rows (median age delivered: 24) — the rest degraded
into `_last`. That is why `[7,30,364]` was the worst of the four good ones.

**Sealed:** blend (what ships, 0044) val 0.14138 → **0.17773**, beating the argmax
lgbm's 0.18595 — 0044 holds on a second dataset. Optimism +0.036, so v7's
registered "optimism ≤ 0" fails; drift and selection stay unseparated (0041).

### Two things to carry forward

- **`_age` measured to the observation *actually read* is what made this
  diagnosable.** `lag7` reported median age 24 and told the truth about failing.
  Had age been the offset *requested*, it would have read 7 everywhere and been
  silently wrong.
- **`forecast_horizon` was nearly left undeclared.** Without it offsets are refused
  and the run's best table cannot exist.

### The gotcha, and it was mine

`summarise_history` printed *"the feedback loop is decorative"* — a **hardcoded
verdict that assumed its own premise** (that the probe had called these tables
worse) while the deltas printed directly above it said 4 of 5 were better. Fixed
to read the deltas. 0031's shape in the reporting layer: a check that looks like
it checks and only asserts. **Do not trust a conclusion a script prints about its
own run without reading the numbers beside it.**

**Next:** slice 3's argument has changed — offsets already deliver recency *and*
periodicity (the agent picked 364/728 to keep the weekday aligned, which is the
winner's feature shape). A window would add only the averaging. Possibly worth
less than making `_last` optional and reporting each offset's answerable share.

## 2026-08-14 (evening) — Rossmann says no, and the reason is worth more than the number

**Machine:** MacBook Air | **Branch:** master | **425 tests still pass** (no code
changed as a result of this).

Ran slice 2 against Rossmann via `scripts/probe_rossmann_history.py` — not an
agent run, `fit_history` driven by hand against v7's split so the number moves for
one reason. **The feature made the probe substantially worse.** Written up as
[0049](docs/decisions/0049-the-last-observation-is-a-sample-of-size-one.md).

- baseline (Customers dropped) **0.21810** → + Store history of Sales **0.35054**
  (−0.13244). Every variant worse; `Store+DayOfWeek` −0.09720, `lag7` −0.13659,
  history of `Customers` −0.14111.
- The **naive** lag (val rows reading their own labels, built outside the harness
  because 0011 refuses to build it) scored **0.09999** — better than baseline by
  +0.11811.

**The cause:** the work half ends **2015-05-01**, a public holiday with **98.21%
of stores shut**. Training columns are healthy (median `_last` 6488, 0.1% zeros)
because forward chaining freezes at five different fold boundaries. But **every
validation row reads one freeze date**, and that one was a holiday — median
`_last` 0, **98.2% zeros**. Not this split's bad luck either: **16.8% of the 851
candidate work-end dates** have >90% of stores selling nothing.

**Control** (`probe_rossmann_history.py 0.06`, boundary moved to an ordinary
Wednesday, 0% shut): baseline 0.22460 → 0.23602, **−0.01142**. So the holiday is
~92% of the damage and the mechanism is not broken — **but the feature still hurts
on a clean boundary.**

**One key survives.** On that clean boundary `Store`+`DayOfWeek` history is
**+0.00272**, the only positive of the four. "What did this store sell the last
time it was this weekday" puts the closed days in **their own groups**, so a
Sunday's zero can only be read by another Sunday. Small and directional only, but
it establishes the mechanism can help at all. General form: **a group's last
observation is usable only if the group is homogeneous enough for one member to
stand for the rest** — the same requirement a group mean has (0046), from the
other side. Hence the fix is a narrower key or a wider window; both cut the
variance of what is read.

### The general statement, and the two things 0047/0048 missed

**The most recent observation is a sample of size one.** `_age` reports how *old* a
value is; **nothing reports how *typical* it is**. Honest about staleness, silent
about representativeness.

Second, quieter: **the age ranges don't match.** Training ages run 1–305 (median
74, fold blocks ~150 days), validation ages 1–45 (median 23). 0048's serving
symmetry holds in *shape* but not in *scale* — block width comes from `n_splits`,
the horizon from the task. Structural, not incidental.

### The gotcha for whoever picks this up

**Do not fix this by making Rossmann better.** Skipping closed days in history, or
taking the last *non-zero* observation, each improves the number and neither can
be defended without naming the dataset — 0022 one level up. 0049 says so
explicitly.

**Next:** slice 3 (rolling/expanding windows) is now **the fix, not a bonus**. A
7-day mean over the same stored tail gives **0%** of stores a zero (median 5771.7)
— robust to exactly this failure, because it is not a sample of size one.

### Input for slice 3, from outside: what the competition winner used

Ulf points out the Rossmann winner's writeup leans on features like *"sales for
this store on a typical Monday, averaged over the last 6 months / 1 year"*. That
is **a narrow periodic key plus a trailing window** — the two fixes 0049 named,
combined. Three things follow, and they change slice 3's framing:

- **The key half is already built and already paid.** `fit_encoding(method=
  "target")` on a composite `Store_DOW` key *is* "average sales for this store on
  a Monday", and it was v7's **t8, +0.0243, the largest gain of that run** (0046).
  Not a gap.
- **The missing half is recency.** That aggregate runs over *all* work-half
  history; the winner's is limited to a trailing period. A store whose trade level
  moved a year ago is described badly by an all-time mean and well by a trailing
  one. **That is slice 3, and it is a better argument for it than 0049's
  structural-zeros one.**
- **It argues `_last` is not the valuable part of history.** The winner used
  windowed aggregates, not a last-known value — which is what 0049's numbers say
  too. Worth stating before slice 3 rediscovers it.

**Where the line actually is — and it is not where a first reading suggests.** The
*agent* must be able to build this feature. `fit_history(by=["Store","DayOfWeek"],
value="Sales", window=180, stat="mean")` **is** the winner's feature, assembled by
the agent out of general parts, and that is the goal rather than the thing to
avoid. What stays out of `src/` is a *default*, a blessed list of window lengths,
or a docstring that names the dataset. The capability is general; the choice is
the agent's (0029).

Concretely, for slice 3:

- **`window` is a free number in the ordering column's units**, not a menu. A
  menu would be us picking the windows, which is the agent's call and would
  encode "180 and 365 are the good ones" — true of one dataset.
- **`keep` has to cover the longest window asked for.** `MAX_KEEP` is 1000, so a
  365-day trailing window is expressible; a multi-year one on daily data is not,
  and that ceiling should be a stated refusal rather than a silent truncation.
- **Watch the storage.** `Store`×`DayOfWeek` over a year is ~52 observations ×
  7,805 groups; `Store` over a year is 365 × 1,115. Both ~400k floats — bounded,
  which is the property that matters, but no longer small. Worth measuring before
  it goes in a bundle.

## 2026-08-14 (later still) — §3.4 settled, and slice 2 is built

**Machine:** MacBook Air | **Branch:** master | **425 tests pass** (was 387).

### 0048 — the last open question, and its premise was wrong

`docs/time_aware_features.md` §3.4 said a target lag "is target encoding's near
relative and needs the same care: out-of-fold construction". It is not. Two
failures were wearing one word:

- **Self-inclusion** (the row's own target in its own feature) is what OOF removes
  for `TargetEncoder`/`GroupStatsEncoder`. A strictly backward lag **cannot commit
  it** — the row at `t` only reads `t-k`, at any group size, under any folds. OOF
  buys nothing.
- **Availability skew** (reading a label that will not exist at serving time) is
  the real risk, and OOF does not fix it. The fix is 0047's frozen horizon,
  mirrored per fold.

So the fold loop looks like `fit_group_stats`'s and exists for an unrelated
reason. **This is the gotcha to carry forward:** fitting each fold's history on
"everything outside the validation fold" satisfies the older rule and is a
straight future leak here, because *last known* is defined relative to the fitting
set — put a later block in it and the newest observation comes from the future.
A strictly backward lag survives that by luck; the last-known lookup 0047 chose
does not.

Also decided: **target history is refused under random k-fold** (0013-style, not a
warning). Checked as a property of the index pairs, not a trusted flag.

### Slice 2 built

`HistoryEncoder` + `fit_history` in `encoders.py`, `history` in `ENCODER_CLASSES`
so `Pipeline.transform` replays it, `TaskSpec.forecast_horizon`, and the
`fit_history` harness tool. Emits **pairs**: `<by>_<value>_last` and `..._last_age`,
plus `_lag<k>`/`_lag<k>_age` per offset.

- **Age = row − the observation actually read**, not the offset asked for. So an
  unanswerable `lag7` degrades into `_last` and reports the real age rather than a
  stale number under a fresh name.
- **Offsets require `forecast_horizon`** and must be ≥ it. Undeclared → last-known
  pair only, which is honest without a declaration.
- Storage bounded by `keep` per group; encoder size never depends on table length.
- `value` **may be the target** here, unlike `fit_group_stats` — a target lag can
  only come from stored history, never from a served frame.

### Two things that bit, worth knowing

- **`pandas` hands back `datetime64` in s/ms/us/ns** depending on how the column
  was built. Scaling from the integer representation put every age out by 1000x.
  `to_periods` now divides by a `timedelta` instead.
- **Dates arrive as strings**, because harness tables come from CSV. Unit tests
  used an integer `day` and missed it entirely; caught only when the notebook ran
  the real tool. `to_periods` accepts **ISO 8601 only** — `01/02/2015` is refused
  rather than guessed, since guessing wrong silently transposes day and month.

**Notebook:** `notebooks/phase5_history_and_the_horizon.ipynb` (36 cells, ships
unexecuted, every cell smoke-tested in order). Its section 6 is the payload: the
naive lag scores 0.973 within-store correlation against the frozen one's 0.77 —
**that gap is the size of the lie, not the size of the loss**, and 0048 says so
explicitly so it is not later "fixed".

**Next:** slice 3 — rolling/expanding windows over the stored tail. `keep` already
carries the `L + W` bound. Still unvalidated by any real run; nothing here has been
scored on Rossmann.

## 2026-08-14 (later) — The two blocking questions are answered, and Gap A is closed

**Machine:** MacBook Air | **Branch:** master | **387 tests pass** (was 363).

Worked `docs/time_aware_features.md`. Both questions it named as blocking are
settled with Ulf, and slice 1 is built.

### 0047 — the horizon, settled (accepted, **not implemented**)

Ulf's answer to §3.3 collapsed both questions into one. The real forecasting
contract is: history to `T`, predict `T+1 … T+k`. For `T+1` you know yesterday's
value; for `T+2` you do not. **So availability is not a property of the served
set, it is each row's distance from the end of history** — which is what §3.2 was
asking. One mechanism answers both:

- **Frozen history + a per-row staleness column.** `T+1` gets the last known
  value at staleness 1 (that *is* yesterday's value), `T+5` the same value at
  staleness 5. Degrades visibly instead of lying quietly.
- **`TaskSpec.forecast_horizon` optional, and a *minimum*.** The refinement that
  is easy to get wrong: `h` varies per row inside one predict batch, so a single
  declared horizon cannot gate lags globally. It is the floor across the block;
  staleness stays per row.
- **No recursion** (predictions not fed back as lags) — sequential prediction,
  compounding error, and structural train/serve skew. Rejected-for-now, reopenable.
- **Predict rows never read their own batch's past.** This is the one that costs
  a guarantee: a lag reading the batch's earlier rows *is* 0011's batch-dependence.
  Frozen history keeps batch-independence by construction.

Two pre-work checks that changed the framing:

- **0030's serving schema does not already express §3.3.** It is column-level;
  §3.3 is temporal. And since the target is excluded from `serving_columns` by
  construction, a target lag can never come from a served frame at all — so §3.3
  is really "is stored history ever *extended*?", a new declaration.
- **0011's check passes a stored-history step as-is** (`unstable_columns` only
  ever runs on sandbox code), *because* history is frozen. That equivalence is
  exactly what the permissive answer would have spent.

### 0046 — slice 1 built: a group is more than its average

`fit_encoding(method="target", stats=[...])`, `TARGET_STATS = (mean, std,
median, count)`. `mean` keeps the bare `<col>_te` name; the rest get
`<col>_te_<stat>`. Default is `["mean"]`, so nothing changes unasked.

- **One tool, not two.** Lifting `fit_group_stats`'s target refusal for the
  non-mean stats is less code and produces an indefensible seam — a group's
  target *mean* from one tool, its target *median* from another.
- **Only the mean is smoothed.** The weighted-average formula is a statement
  about means; shrinking a median by a count weight invents an estimator and
  gives it a standard one's name. **`count` is emitted instead** — same
  information, learned rather than baked into an untuned constant.
- **`min`/`max`/`sum` off the menu.** An extreme of the target in a group is
  essentially one row's target and survives OOF only by luck of fold assignment;
  `sum` is `count * mean`.
- Everything is out-of-fold, including `count`.

### The gotcha worth carrying forward

**A leak test written for one statistic does not transfer to another statistic of
the same column.** The mean's OOF test correlates the encoding with `y`. Applied
to `std` it *fails* — and not because `std` is safe: a row inflates its group's
spread whether its target is high **or** low, so the leak lands on `|y - mean|`
and sits near zero against `y`. Replaced with a direct perturbation test (move
one row's target, watch that row's own feature) which covers all four with one
mechanism.

Also: **the serialized target encoder changed shape** — `mapping` is now nested
by statistic. `from_dict` detects the old flat shape by the absent `stats` key
and loads it mean-only, so bundles on disk keep predicting what they predicted.
One existing test read the flat shape and was updated.

### Also

- `tests/test_target_stats.py` (15) + `tests/test_target_stats_harness.py` (9).
- `notebooks/phase4_target_distribution.ipynb` ships unexecuted, all 11 code
  cells executed in order from the shipped file.
- `fit_group_stats`'s target refusal now names `stats=` instead of only saying
  the mean needs smoothing.

### Next

- **Slice 2 — the `history` lineage step.** Unblocked now; 0047 is the spec.
  Bounded storage (last `L + W` per group), replay at predict, and §3.4 (lagging
  the *target* specifically) still to settle.
- **No run has used 0046.** 0032 is the precedent: `fit_group_stats` was the tool
  the agent asked for and then never called once in 69 calls.
- Everything from the entry below is still open: fix the ruler, a non-Rossmann
  sealed run for 0044's claims, the request timeout, `scripts/run_rossmann.py`.

### Gotchas

- **Quiz for 0046/0047 done** (0041/0042/0044's are still owed). It surfaced a
  caveat 0046 does not state and should have: **`count`-instead-of-smoothing only
  works for a model that can express an interaction.** A GBDT splits on
  `cat_te_count < 5` and learns the discount; `elasticnet` fitted on both columns
  just adds them, so for the linear member of the terminal ballot `count` is
  strictly weaker than smoothing and smoothing is still doing the real work.
  0046 is accepted and stays as written — carry this into slice 2's record, or
  its own if slice 2 does not touch it.
- `count` under an ordered split is a *partial* count (folds preceding the row),
  while the stored encoder's count covers all rows — so the trained column and
  the served column measure subtly different things. Already true of the mean;
  noted in 0046, not fixed.

## 2026-08-14 — Kaggle confirmed the blend, so the blend is what ships now

**Machine:** MacBook Air | **Branch:** master | **363 tests pass** (was 353).

### 0042 claim 1 passed on an external seal

Rossmann v7's submissions, scored on Kaggle:

| | private | public |
|---|---|---|
| `submission.csv` — argmax winner (xgb) | 0.14356 | 0.13960 |
| `submission_blend_lgbm_xgb_catboost.csv` | **0.13813** | **0.12932** |

**+0.00543 private**, on rows nothing in this project has ever touched. Not a
fresh run — same models, same table, only the judging rows are new — but it rules
out the specific worry that stopped 0042 implementing: that the effect lived in
v7's sealed tenth alone.

### 0044 — implemented

`finalize_run` **ships an equal-weight average of every terminal fit that
trained**. No weights fitted, no candidate dropped, nothing chooses. Verified
against v7: the new library path scores the four-way blend at **0.13618** on v7's
sealed rows, matching 0042's recorded number exactly.

- **Every candidate, no filtering.** "Keep the trees, drop the linear model" is
  refuted by house v9, where elasticnet was the best model on the board. And on
  v7 four-way vs three-way differs by 0.00008.
- **Residual risk, named in the record:** Kaggle backed the *three-GBDT* blend;
  what ships is the four-way.
- **`selected` vs `shipped`** are now separate flags on the holdout rows.
  `selected` still means "what the search crowned" so 0039/0040's optimism series
  stays comparable; `shipped` is what the bundle predicts with. Their difference
  is 0042's A/B, **recomputed by every run from now on** instead of by a script.
- **The seal still picks nothing.** Members are fixed before any sealed row is
  read; `_score_on_holdout` receives the finished blend as an argument.

### A defect found on the way — and it would have shipped an unmeasured estimator

`predict_bundle` combined models **on the model's scale and inverted once at the
end**. Under `log1p` that is a *geometric* mean of the target. Every number in
0042, and both Kaggle submissions, were arithmetic means (`blend_submission.py`
inverts first). So passing k models into the old bundle would have shipped an
estimator never scored against anything, quoting a number belonging to a
different one. `predict_bundle` now inverts per model, then averages.

**Nothing on disk changed:** for a single-model bundle the two orders are
identical, and until now every library-written bundle had one model.

### Also

- `tests/test_blend_ships.py` — 10 tests. `notebooks/phase4_blend_ships.ipynb`
  ships unexecuted, all 10 code cells executed in order from the shipped file.
- Report drops the CV score line, the fold table, and the "gap to the CV score"
  sentence when a blend ships — it was never cross-validated.
- `scripts/blend_submission.py` docstring updated; it now enumerates *subsets*,
  which the harness still does not.

### 0045 — the four-way blend scored, and 0044's cost estimate was 44x wrong

Scored after 0044 was committed:

| submission | private | vs argmax |
|---|---|---|
| `submission.csv` — argmax (xgb) | 0.14356 | — |
| `submission_blend_all.csv` — **what ships** | **0.14172** | **+0.00184** |
| `submission_blend_lgbm_xgb_catboost.csv` | 0.13813 | +0.00543 |

**0044's decision holds** — the average beats the argmax on an external set. **Its
cost section does not:** including elasticnet costs **0.00359**, where v7's seal
said 0.00008. Two thirds of the benefit of blending, eaten by the weakest member.

**No code changed, deliberately.** Filtering on this number is choosing a rule by
consulting a holdout — 0042's refusal, and the private LB is now spent for this
question. House v9 still refutes the category rule ("drop linear models" deletes
that run's winner). 0045 registers a hypothesis with a mechanism instead: the
seal may understate a linear model's drag because it measures at six weeks where
Kaggle measures at four months (0043), **or** 0.00008 was never an effect at all,
just an unresolved difference quoted as if the smallness were the finding.

### The bigger finding this session: we are out of feature vocabulary

Went through v7's 15 tables. **Ulf's hypothesis — per-group target means like
"average sales for this store on a Monday" — is already there and was the
biggest win in the run:** `t7` built `Store_DOW`, `t8` target-encoded it, probe
0.1587 → 0.1344, **+0.0243**. Then `t9`–`t15` built nine more composite keys the
same way for **+0.0006 total**, three negative. Half the run re-drilling one hole.

Two gaps behind that, both general:

- **Only the mean.** `fit_group_stats` correctly refuses the target as its value
  column, and the target encoder only makes a smoothed mean — so per-group
  **std / median / count** of the target cannot be written at all.
- **No feature can look at another row in time.** No lag, no rolling, no
  days-since/until. `transform_features` *could* write `groupby().shift()` and
  0011's stability check would rightly kill it; and at predict time the test
  period is later than every training row. **A lag needs the harness to carry a
  time series across the fit/predict seam, and it can only carry a mapping.**

Stated generally: *"a feature whose value depends on other rows, ordered by a
known key, whose history is stored at fit time and replayed at predict time."*
No dataset in the sentence.

### Next

- **Time-aware features — start here, and start by reading
  `docs/time_aware_features.md`.** Written this session precisely so a fresh
  session does not rebuild the argument. It has the evidence, the general
  statement of the gap, a three-slice plan, and **two blocking questions that
  need Ulf's answer before any code**: (3.2) what a lag serves when the predict
  rows are past the end of stored history, and (3.3) whether validation rows may
  use their own past. Everything else follows from those two.
- **Fix the ruler** (blocked/rolling-origin CV, or paired-bootstrap-of-difference
  from 0036) — still necessary, but it is a multiplier on a search that currently
  has nothing new to search over. After lags, not before.
- 0044's registered claims need a **non-Rossmann** sealed run to test.
- Still open from last session: lift the request timeout into the library;
  `scripts/run_rossmann.py` still broken since 0035.

### Gotchas

- **No quiz done** for 0041/0042 yet, and none for 0044 — questions asked at the
  end of this session are unanswered.
- Rossmann private LB is 3,303 teams and 0.13813 is roughly **rank ~2,200** —
  interpolated from HANDOFF's own score→rank mapping, **not** from the real
  leaderboard file. Below median; worth pinning down properly.
- **Rossmann's private LB is now spent for the blend-membership question.** All
  three relevant submissions are scored; there is no unconsulted external set
  left to judge a filtering rule on this dataset. The next test has to be a
  different dataset.

> **Entries below are compressed summaries** (done 2026-08-15). Each one's full
> argument is in the decision records it cites; the long-form text is in git at
> `801b52d:HANDOFF.md` if a detail is ever needed. Only findings *not* carried by a
> record are stated at length here.

## 2026-08-13 (later) — the sealed split is in, and it caught us shipping the worst model

**MacBook Air | master | 353 tests.** Records [0041](docs/decisions/0041-the-sealed-split-scores-the-ballot-not-the-winner.md), [0042](docs/decisions/0042-the-terminal-argmax-shipped-the-worst-model.md), [0043](docs/decisions/0043-on-ordered-data-the-seal-costs-the-freshest-rows.md).

`RunSession(holdout_fraction=)` built, opt-in and off by default, a fraction of the
whole input, ordered splits cutting `work | val | holdout` in time order, every
terminal candidate scored. Rossmann v7 then **shipped the worst GBDT**: xgb won val
at 0.12980 and scored 0.15253 sealed, against catboost's 0.14113 — optimism +0.0227,
the worst ever measured. Every equal-weight blend beat every single model
(lgbm+catboost 0.13432), which became 0042 and later 0044. 0043: on ordered data the
seal takes the *freshest* tenth, so v7 forecasts 3–5 months out where the field
forecasts six weeks, and v5's Kaggle 0.135 is not a fair control.

**Live gotchas:** a hung request has **no timeout anywhere in the library** — v7
attempt 1 sat 105 minutes on one gateway POST; patched only in that driver.
`scripts/run_rossmann.py` broken since 0035. The agent can read the sealed number
after `finalize_run` — harmless by call ordering, not by structure.

## 2026-08-13 — 44% of the search budget went to a set we could have enumerated

**MacBook Air | master | 329 tests.** Records [0034](docs/decisions/0034-the-tiebreak-was-tested-and-rejected.md), [0035](docs/decisions/0035-the-search-is-over-features.md), [0036](docs/decisions/0036-the-pairing-asymmetry-was-not-there.md), [0037](docs/decisions/0037-residuals-need-folds-the-ranking-does-not.md), [0038](docs/decisions/0038-the-first-feature-only-run.md), [0039](docs/decisions/0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md), [0040](docs/decisions/0040-seguro-v4-a-near-copy-of-the-ruler-is-not-the-ruler.md).

Counted across 16 runs: 222 experiments, **97 (44%) varied only the model or its
params**, and the winner was a model variation in 10 of 16. So the agent's action
space became **tables only**, ranked by a fixed probe model (`probe.py`, one lgbm at
library defaults, one fit per table); the terminal phase enumerates families
afterwards. 0034 tested the cv tiebreak (1 win, 1 loss, 1 null) and rejected it.
0036 refuted 0035's pairing-asymmetry claim by paired bootstrap. 0037: ranking needs
one fit, residuals need folds — `explore_table` refits on request and reads the
**work half only**.

House v9 then ran feature-only: val *worse*, Kaggle *better* (0.12037, top 9% vs
v8's top 20%), and ablation showed the gain was the feature search, not the terminal
enumeration. Seguro v3/v4 found the same error at two grains — first the probe's
*family* missing from the terminal ballot (cost 0.0256 gini), then its
*configuration* (seed 42 vs 0) — hence 0040's rule: **an invariant about a specific
measurement cannot be enforced by reproducing the category it belonged to.**

**Live:** the lgbm-vs-selection confound is unresolved (every selected/unselected
pair so far is lgbm vs elasticnet). Terminal fairness open — enumerating library
defaults measures the libraries; this is what 0054 eventually cashed in.

## 2026-08-12 — first classification run; the diagnostic model was unrunnable at scale

**MacBook Air | master | 309 tests.** Record [0033](docs/decisions/0033-a-ceiling-in-the-wrong-unit-is-no-ceiling.md).

Porto Seguro, 595k rows, given to the agent **untouched** — it dropped all 20
`ps_calc_*` columns and target-encoded the `_cat` columns in two tool calls,
unprompted. Kaggle private gini 0.272, top 66%; optimism +0.0128.

The defect: `_LINEAR_MAX_ITER = 20000` served both coordinate descent and saga,
**which do not share a unit** — one fit projected at 77 min, the search at ~510
hours. Split by solver, plus `_PenaltySearchThenRefit` (search sees ≤20k rows, refit
sees all). 510 hours → 4.6 min, bit-identical below the cap. Convergence turned out
to be a property of the *penalty*: at the `C=0.01` the search selects, saga
converges in 118 epochs.

v2 then showed **`cv` ranked all three models correctly and `val` put the best one
last**, with the inflation specifically in the models the argmax crowned (`e8` and
`e11` differ on val by a twentieth of one `val_se`, and their val errors differ 3x).
That is the winner's curse with the confounds stripped out, and it is what 0023/0041
were built for.

**Live gotchas:** the wall-clock bound checks *between* folds, so a single runaway
fold outruns it entirely — refuted premise, guard unchanged. `python script.py > log`
buffers; use `python -u`. A killed run loses `transcript_N.json` but keeps the
journal.

## 2026-08-12 — thinking changed the behaviour and not the score; the ceiling is diversity

**MacBook Air | master | 306 tests.** **No decision record — `elasticnet` and
`fit_seconds` are still untaught and unrecorded, and the elasticnet-as-diagnostic
argument is the part a newcomer cannot reverse-engineer.**

First runs on a reasoning route (`fireworks/deepseek-v4-flash-0731`). **Behaviour
changed a lot, the score did not:** house v8 hit the same Kaggle 0.123 as run 3 from
a val score 0.0018 better — i.e. optimism went *up* on *fewer* experiments. Thinking
was cheap (fewer tokens, half the tool calls). What it did change was tool use:
`fit_group_stats` went from 0 calls to the second call of the run, `filter_rows` was
used for outlier removal, and the data-dictionary channel finally paid — v8 encoded
21 columns as `ordinal`, an order that can only have come from the dictionary.

**Blending is not our gap, measured twice** (Rossmann OOF-weighted: worse;
house 3-GBDT: +0.0002). The reason is GBDT-vs-GBDT residual correlation of
**0.86–0.88**. The gap is diversity, which is why `elasticnet` exists — on v8's
winning table it *matches* a tuned catboost, so by the TabArena rule the signal is
in the features. Its preprocessing lives **inside the estimator** so `train_cv`
refits per fold; in a feature table it would leak across the fold boundary in the
one place 0011 cannot see.

**Live:** a gateway's `supports_reasoning` flag describes the model *family*, not the
route — only a real call tells you. `ElasticNetCV` has no `scoring` parameter (noted
in code, not fixed). And an unacted finding worth keeping: **on ordered splits the
last fold ranks models far better than the mean of folds** (Spearman 0.960 vs 0.864),
a candidate correctness fix to what `cv_score` means under `split_on`.

## 2026-08-11 (latest+7) — both gaps closed; the run used the wrong one, and it did not help

**MacBook Air | master | 291 tests.** Record [0032](docs/decisions/0032-fitted-aggregates-and-a-step-that-is-not-replayed.md).

`fit_group_stats(by, value, stat)` — per-group statistics of a **non-target** column,
fitted once and looked up at predict, **always out-of-fold with no opt-out** (the
tool cannot tell whether the value column proxies the target). `filter_rows(code)` —
returns a mask, rows dropped once at table-write time, lineage step carries
`train_only: True` so `Pipeline.transform` skips it. Two exemptions follow from
"evaluated once, never replayed": the filter may read the target and may be
batch-dependent.

Rossmann v5 finished, finalized itself, wrote the first `submission.csv`, and
**refuted the premise the filter was prioritised on**: dropping the 17% closed-store
rows cost 0.0034 on a fold spread of 0.07 — because `Open` is a feature and the model
already learns "closed → 0". The capability is still right on generality. And
`fit_group_stats` — the tool the agent had *asked for* — got **zero calls in 69**,
the fourth time changing what the agent is told failed to change what it does.

Kaggle scored it 0.135 and **our exam was pessimistic for the first time**
(val 0.13760), which is 0029 paying off: an ordered split plus forward chaining asks
the competition's question, so there is no interpolation premium to hand back.

## 2026-08-11 (latest+6) — a decorative guard, a corrupted run, and the tool the agent keeps asking for

**MacBook Air | master | 255 tests.** Record [0031](docs/decisions/0031-the-guard-was-decorative.md).

v3 died holding the best servable model any run had produced, on a corrupt parquet:
two `fit_encoding` calls computed the same table id and wrote the same file at once.
**0008 set `max_concurrency=1` to prevent exactly this, and that parameter limits
concurrent agent *runs*, not tool calls — it never did anything**, and the test
asserted the constructor argument had been passed rather than the property it was
named for. Fixed in the primitives: `O_CREAT|O_EXCL` id claims, `os.mkdir`,
tmp→fsync→`os.replace`.

Also found: the agent reached unprompted for per-store aggregates of another column,
0011 refused them correctly, the refusal message pointed at a tool that **cannot do
this**, and three refusals exhausted the retry budget and killed the run. That gap is
what `fit_group_stats` closed the next session.

## 2026-08-11 (latest+5) — Rossmann ran three times; the leak is refused, and we audited what we cannot express

**MacBook Air | master | 254 tests.** Record [0030](docs/decisions/0030-a-model-that-cannot-be-served-is-not-trained.md).

v1's agent spotted the `Customers` leak in prose, dropped it unprompted, and built an
honest chain to 0.13173 — **which counted for nothing**, because the leaked model at
val 0.07912 (better than first place, the signature of a column nobody had) stayed
incumbent, ten genuine improvements were each recorded as no-improvement, and
finalize bundled a model that raises `KeyError` on the first real prediction. Fixed
by `serving_columns` + refusal at train time; v2's agent read
`columns_not_available_at_predict_time` off `inspect_data` and guarded it in its
*first* table.

**The capability audit against the winner's solution** named three gaps: no row
filtering, no time-windowed group aggregates, no blending — all three since built
(0032, slice 2/3, 0044). Deliberately not gaps: external data (0001) and the 0.985
prediction correction, whose own author reports it was inside the CV noise.

## 2026-08-11 (latest+4) — Rossmann runner written, NOT run; a third failure mode

**MacBook Air | master | 250 tests.** Dry-running the machinery without the LLM found
that dropping a serving-absent column has **two** failure modes, and the middle one
is the trap: `df.drop(columns=['Customers'])` gives an honest score and still cannot
be served, because per 0010 a transform is stored as *code* and replayed on raw rows
that never had the column. `errors='ignore'` is the only submittable form. **Nothing
in the harness can catch it** — `_score_on_val` replays over validation rows drawn
from the training file, which still have the column. There is no row anywhere in a
run shaped like a real serving row. Generalises: any transform referencing a
predict-time-absent column breaks the bundle silently.

## 2026-08-11 (latest+3) — the harness owns the exam; CV forecasts when the run does

**MacBook Air | master | 250 tests.** Record [0029](docs/decisions/0029-the-harness-owns-the-exam.md).

**The harness owns the evaluation protocol; the agent owns the modelling.** So
`Customers` went back into the Rossmann data — dropping it was the harness doing the
agent's work — and whether the agent uses it became a measured result. Three fixes to
an exam that was subtly wrong: `make_folds(order=)` builds forward-chaining folds
(random k-fold on a panel is interpolation, not forecasting); the split cuts on a
*value*, not a row count; a table dropping the ordering column is refused rather than
silently falling back to random folds.

**Live:** under forward chaining ~1/(n_splits+1) of training rows get no out-of-fold
prediction at all, and out-of-fold target encoding leaves that same block NaN with
the missingness time-correlated — a tree can learn it as "early period". Not fixed;
every alternative computes the value from the row's own future or from itself.

## 2026-08-11 (latest+2) — Rossmann: the data dictionary now has a channel

**MacBook Air | master | 236 tests.** `TaskSpec.column_notes={col: text}`, rendered
per-column inside `inspect_data`'s profile; run-level `notes` returns as
`about_this_data`. An undocumented dataset costs exactly nothing.

The finding that mattered more than the feature: **`Customers` is a leak this harness
cannot see** — 0.895 correlated with `Sales`, absent from `test.csv`, present in our
validation half because that is drawn from `train.csv`. No fold crossed, no encoder
misfitted, every guard holding. Nothing computable from the training file reveals it;
the only evidence is the competition's data dictionary. `scripts/prep_rossmann.py`
carries the join, typing and dictionary.

## 2026-08-11 (latest+1) — `val_se` shipped; the bar it gates is deliberately still open

**MacBook Air | master | 233 tests.** Record [0028](docs/decisions/0028-the-yardstick-is-fixed-the-bar-is-not.md), **which corrects 0022**.

`_score_on_val` bootstraps the validation score over its own rows (200 resamples,
seeded). It lands on the leaderboard row, `meta.json`, the payload, and the report.
**The correction is the important part:** 0022 claims the fix would have told run 3
its 0.008 gain sat inside the noise. It would not — the enforced bar is a *quarter*
of the yardstick, so rejecting that gain needs `min_gain_ratio > 0.66`. 0022 fixed
the category error and left the magnitude error untouched.

`min_gain_ratio` stays at 0.25, still open. An advisory note fires when a gain is
inside one SE — shipped as the experiment, since advisory guidance had already failed
three times. **`val_se` understates uncertainty under an ordered split** (the
bootstrap assumes i.i.d. rows); inherited from 0022, unaddressed, and it is why 0036's
paired bootstrap is still the blocking item.

## 2026-08-11 (latest) — v7 ran: the brief does not cut the orientation tax

**MacBook Air | master | 226 tests, no library code changed.** Record [0027](docs/decisions/0027-the-brief-did-not-cut-the-orientation-tax.md), **which supersedes 0026's central claim**.

Orientation per resumed episode: v6 without a brief **41%**, v7 with one **40%**. No
change. The *composition* changed completely while the size did not — `inspect_data`
calls fell exactly as designed and the agent spent the freed calls on
`explore_journal` instead. So **orientation is not an information deficit that can be
paid off; it is roughly a constant fraction of whatever episode it is given.** Third
advisory fix in a row to fail. Whatever solves it will be structural.

Not claimed: v7's better score (inside one `val_se`) or its lower call count (episode
1 is configured identically in both runs and differed by 8 calls — that is the noise
floor). General lesson from the same session: **a baseline not computed with the same
code is not a baseline.**

## 2026-08-11 (earlier) — the derived brief is built; it is not yet measured

**MacBook Air | master | 226 tests.** Record [0026](docs/decisions/0026-the-brief-is-rendered-not-remembered.md) (implementation stands, hypothesis refuted by 0027 above).

`harness/brief.py` renders a ~170-token `RUN STATE` block from the journal, wired as a
*dynamic* system prompt so it renders when the episode starts. Returns `None` on a
fresh run, deliberately, so a first episode is byte-identical to a run that never had
episodes — which is what made v6 and v7 comparable.

**The load-bearing test is the flatness one:** `test_the_brief_does_not_grow_with_the_run`
fakes 151 leaderboard rows and asserts the brief moves by <20 chars. It is the only
thing standing between this design and someone adding "recent experiments", which is
the natural next request and the thing that breaks it.

## 2026-08-10 (latest+1) — `explore_journal` is taught, and writing it found a bug

**MacBook Air | master | 215 tests.** No record — nothing decided.
`notebooks/phase3_harness_step6_journal.ipynb` (46 cells) teaches the tool and target
transforms together, since `target_transform` is a column in `lb` and "did log1p pay"
*is* a journal query.

**The bug:** `explore_journal`'s `tables` frame reported **zero columns added, for
every table, in every run since it was written** — it read `t.get("columns")` where
journal rows carry `new_columns`. Not a crash; a confident `0` to the one question
that frame exists to answer. Six unit tests covered the tool and all passed either
way, because none asserted a count they had independent reason to expect. **First time
the notebook convention paid in a bug rather than in prose.**

**Live gotcha:** the `notes` frame comes from `journal["log"]`, which no agent-facing
tool writes, and unlike `lb`/`tables` its columns are **not declared** — so it is a
`(0, 0)` frame and `notes["hypothesis"]` raises `KeyError`. Harmless while always
empty; a trap the day a note tool lands.

## 2026-08-10 (latest) — episodes work; the agent will not end one by itself

**MacBook Air | master | 214 tests.** Records [0024](docs/decisions/0024-the-episode-stop-is-structural.md), [0025](docs/decisions/0025-episodes-buy-feasibility-not-cost.md).

Asked in plain terms to follow one idea then stop, the agent ran all 25 experiments
in a single episode. **It may own the episode stop; it will not exercise it.** With a
structural cap, episodes work: peak context per episode 9k–12k against a single run's
climb to 43,691 — linear becomes sawtooth. A resumed agent orients itself unprompted,
opening `inspect_data → explore_journal` with no mention of the tool in its prompt.

0025 corrected two of 0024's claims: the orientation tax is ~1/3 of every episode at
**both** caps (given room the agent orients *more*, so a bigger cap is not the fix),
and episodes cost **more** per experiment at this scale — they buy **feasibility, not
cost**. Crossover ~35 experiments; single-run impossible ~60–80.

## 2026-08-10 (earlier) — three live Kaggle runs measured the leaderboard problem

**MacBook Air | master | 214 tests.** Records [0021](docs/decisions/0021-the-agent-picks-the-target-scale.md), [0022](docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md), [0023](docs/decisions/0023-the-third-split-has-evidence-now.md).

`explore_journal` (read-only code over the run's own records), target transforms over
a fixed menu (a per-experiment argument makes it an *experiment*, not a project
assumption — the agent immediately ran a controlled pair and switched), and
`harness/usage.py`. Three house-prices runs, top 20–22%.

**The big one: our val said the target transform bought 0.00808; Kaggle says
0.00070.** 91% was optimism, and optimism grew 4.6x for 1.4x the experiments.
Bootstrapping val over its own 292 rows gives sd ~0.010 — **the gain was smaller than
one standard error of the instrument that measured it.** That is 0022: 0019 moved
ranking onto `val_score` and left the yardstick on `fold_spread`, a bar four times
too small. Also: **`Id` was a feature in run 1** (train 1–1460, test 1461–2919,
entirely disjoint), inert in-sample and damaging only at serve time — no CV, no
holdout drawn our way, and no ablation can see it.

## 2026-08-10 (later) — the agent can see its own mistakes; the RLM kernel was declined

**MacBook Air | master | 197 tests.** Record [0020](docs/decisions/0020-the-sandbox-has-no-kernel.md).

`explore_experiment(exp_id, code)` — read-only code over one experiment's rows plus
`oof_pred`, `residual`, `fold` — closes the `score → hypothesise` arc. Free, no
leaderboard row, works after finalize. **The rejection is the real decision:** a
persistent kernel adds only names surviving between calls, which is run state not in
`run_dir`, dies at the episode seam anyway, and saves re-pickling ~100KB.
`test_nothing_survives_between_calls` holds the line.

**Live:** the leak that remains is **through the agent's reasoning** — an agent
picking features by staring at out-of-fold errors is doing selection informed by
held-out predictions. Every data-path guard holds; no guard catches this. That is the
price of 0020, and it is live, not hypothetical.

## 2026-08-10 — the input is split; the leaderboard is a public leaderboard

**MacBook Air | master.** Record [0019](docs/decisions/0019-the-leaderboard-is-a-public-leaderboard.md).

`RunSession` splits the input once, before any tool runs, into `work.parquet` +
`val.parquet` — time-ordered on `split_on`, seeded-random otherwise, idempotent and
re-read on resume. Every experiment is scored on the validation half by replaying the
trained table's lineage over raw validation rows, and `best`/`baseline`/`vs_baseline`/
`vs_source_table`/attribution/stop all moved onto `val_score`. `cv_score` stays on
every row: the gap between them is the run's record of a gain that lived only in the
folds. `resolve_table` became an allowlist (it used to accept **any absolute path on
disk**), and `finalize_run` became genuinely once.

**Live:** the reported number is the one the agent optimised against — 150
experiments ranked on the same rows is Kaggle's public leaderboard, and nothing in a
run measures how much the search gamed it. Deliberate; the fix is the third split,
which 0023 then argued for and 0041 built.

## 2026-08-09 — credit by ablation; the payload and the prompt cut hard

**MacBook Air | master | 170 tests.** Records [0017](docs/decisions/0017-credit-by-ablation.md), [0018](docs/decisions/0018-the-payload-is-a-budget.md); **0016 superseded** (its unit of credit survived, its measure did not).

`importance_share` is gone; `explain_table` fits once with every column and once per
contributing table held out. Importance decomposes one model's *attention*; the
question is a counterfactual, and the two come apart routinely — on the motivating
case the date columns score `+1.16` and the four original columns `-0.06`, and the
latter held **77% of the splitting**. One sign convention everywhere (`_gain`, positive
is better), after a `logloss` run reported the same improvement as `-0.19` and `+0.19`
in one result dict. Payload 21 → 13 fields, system prompt 924 → 229 words, under three
subtractive rules: say only what is not derivable, not tool-specific, not run-specific.

**Live:** `explain_table` understates every member of a redundant group — two copies
of a signal both read ~0, and it does not test subsets. The prompt guards are
keyword-based and did **not** fire on a real paraphrased duplication Ulf caught by
reading; that rule still needs a human.
`docs/feature_attribution_and_agent_iteration.md` is stale.

## 2026-08-06 — two journal fixes shipped; metadata-kernel + episodes parked

**MacBook Air | master.** `journal.save` became atomic (tmp → fsync → `os.replace`);
`vs_parent` and `feature_blocks` are now persisted rather than returned to the model
and dropped, so the two numbers that say whether an *idea* worked stop dying at the
next context reset.

Three parked designs were written up here at length and **none has a decision
record**: a metadata kernel over the journal (RLM-style, no path to `tables/`), the
episode loop, and a **SQLite schema for the journal**. The first two were overtaken —
`explore_journal` and episodes were built instead. The SQLite sketch was never
built and never recorded; if it is ever picked up, the full schema is at
`801b52d:HANDOFF.md` and **it needs a record of its own**.

The caveat that outlived all three: **target-conditional statistics must not reach a
metadata path.** Distributional facts about features are safe; a per-column
correlation with the target computed on the full dataset is feature selection on all
rows — selection leakage arriving through metadata while every data-path guard holds.

## 2026-08-05 — orientation notebook: the whole agent, one call at a time

**MacBook Air | master.** Nothing built. `notebooks/phase3_agent_map.ipynb` (109
cells) drives every tool by hand and prints a `tree()` of the run dir after each call,
so the folder filling up is the spine of the explanation. A wiki page was explicitly
**not** written — a notebook breaks loudly when it drifts.

Three findings, all from writing prose against real numbers rather than expected ones:
lgbm on the raw table beats the baseline by 20x under the noise; importance share and
marginal gain point **opposite ways in both directions**; and **the noise floor
accepted noise** — `city` is `rng.choice`, independent of the target by construction,
and it cleared the `min_gain_ratio` bar, was recorded as progress, and shipped as
`best`. The stop rule is a defence against *sustained* self-deception, not a
per-experiment lie detector. That finding is what put the holdout on the path to 0023.

---

*Entries before 2026-08-01 pruned on 2026-08-15 (~2-week retention). Their reasoning lives in `docs/decisions/` 0001–0033 and the commit log.*
