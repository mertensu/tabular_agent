# 0054 — The budget is not a constant, and at 200 trees a feature block scores backwards

- **Date:** 2026-08-14
- **Status:** a measurement, and it **closes the question
  [0039](0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md) left open**. No
  code changed yet: what the harness should *do* about this is a design decision
  and gets its own record. What is settled is that the current arrangement is
  indefensible and that the obvious repair — a bigger constant — is also
  indefensible.
- **Related:** [0039](0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md)
  (which named the missing common budget and left it open),
  [0035](0035-the-search-is-over-features.md) (which took model parameters away
  from the agent), [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md)
  and [0053](0053-the-fold-count-moves-a-window-and-we-cannot-say-why.md) (a
  ranked quantity that moves with the ruler — this is the same failure with a
  different knob), [0017](0017-credit-by-ablation.md) (a comparison must differ by
  one thing — the confound below nearly broke that rule),
  [0011](0011-stability-check.md) (which correctly refused the first control),
  [0055](0055-a-rename-makes-every-earlier-artifact-unreplayable.md) (why the
  reading on v8's own best table is missing), `scripts/probe_capacity.py`,
  `_runs/capacity_probe.csv`.

## The question

Every fit in this project runs at **200 trees, learning rate 0.1** — the probe
(`probe.py`), all three GBDT families (`models.py`), every run, every dataset.
Nothing tunes it, nothing stops early, and 0035 removed the agent's ability to
touch it deliberately. 0039 named this and left it open; the `HANDOFF.md` plan
promoted it to the leading explanation for why the two large datasets land
mid-table while house-prices (1,168 rows) reached the top 9%:

> capacity matters in proportion to rows, and 200 trees is roughly right for a
> small table and plainly not for 850,000.

Ulf's instruction was blunter and turned out to be right: *check the hypothesis
everywhere, it is most probably wrong.*

## What was run

`scripts/probe_capacity.py`. The table and the validation rows are held fixed and
**only capacity varies**. Scores come from `probe_table` — the harness's own
ruler, monkeypatched at the parameter dict, so every number is produced by the
code that produced every number in `_runs/`. The prediction was registered in the
script's docstring before the first run.

## The row-count hypothesis is false

| dataset | rows | best fixed budget | early stop |
|---|---|---|---|
| house-prices | 1,168 | **−0.0034** (every budget loses) | 162 → **+0.0062** |
| restaurant | 227,139 | +0.0004 (nothing) | 1,018 → +0.0003 |
| Porto Seguro | 476,170 | **−0.0188** (all but one lose) | 181 → **+0.0049** |
| Rossmann | 915,744 | **+0.0573** | 16,926 → **+0.0606** |

Restaurant at 227k rows gains nothing while Rossmann at 916k gains 0.057. If rows
were the mechanism, restaurant should sit between house-prices and Rossmann. It
sits with house-prices. **The hypothesis as written in `HANDOFF.md` is refuted.**

## What the mechanism actually is

The two datasets differed on a second axis, which is the confound 0017 forbids and
which the first pass nearly shipped as a conclusion. **Rossmann's baseline carries
`Store`** — 1115 identifiers that arrive already typed as an integer, so
`_feature_frame` keeps them. **Restaurant's `air_store_id` is a string**, so it is
dropped, and the restaurant baseline has no entity identity in it at all.

So: give restaurant the same *kind* of column and nothing else. Not
`fit_encoding` — a target encoding adds information as well as identity and puts
the confound straight back — just an arbitrary integer label, which is what
Rossmann's `Store` already is.

| restaurant + a store code | rmsle | vs current |
|---|---|---|
| current 200 @ lr 0.10 | 0.62897 | — |
| 1000 @ lr 0.10 | 0.60064 | **+0.02833** |
| 5000 @ lr 0.03 | 0.60146 | +0.02751 |
| early stop → 3430 | 0.60217 | +0.02680 |

The same rows that gained **+0.0004** from capacity gain **+0.0283** once the
model can tell one shop from another.

> **It is not rows. It is per-entity effects.** 200 trees cannot reach 829 shop
> effects, let alone 1115. Capacity pays where there is high-cardinality identity
> to learn, and that is a property of the *table*, not of its size.

Which is why house-prices and Porto Seguro do not gain: 1,168 rows of houses and
595k rows of anonymised driver features have no such structure. It is also why
every forecasting table we build does have it.

## The finding that matters most, and it was not the one being looked for

The plan wanted capacity swept on v8's own best table. That is impossible at HEAD
(0055). Rebuilt instead from the one part of that pipeline that is pure code — the
agent's own feature block, including its `Store × DayOfWeek` and `Store × Promo`
interactions — on a session with a bit-identical split.

| Rossmann, rmspe | baseline table | + v8's feature block | what the features did |
|---|---|---|---|
| **at 200 trees** (the harness today) | 0.21810 | 0.23331 | **−0.0152 — worse** |
| at 5000 @ lr 0.03 | 0.16083 | 0.15213 | **+0.0087 — better** |

**The sign reverses.** At the budget this harness actually uses, the agent's
feature engineering *scores as harmful*. At a real budget the same columns on the
same rows against the same validation block are worth having.

This is 0022's failure one level up, and worse than 0053's. 0053 found a feature's
measured worth moving with the fold count. This is a feature's measured worth
**changing sign** with a constant nobody chose, in the ruler that ranks every
table in every run. Three sessions of feature search have been ranked by an
instrument that cannot see what it is measuring, and `best_table` — which decides
what gets bundled and shipped — is that instrument's argmax.

Note also that capacity is worth *more* on the engineered table (+0.0812) than on
the baseline (+0.0573). Features and capacity are not competing explanations for
the mediocre scores; the features were being suppressed by the budget.

## What is settled, and what is not

**Settled: no constant is defensible.** Early stopping asked for 162, 181, 1018,
3430 and 16,926 rounds across five readings — two orders of magnitude. Any
constant large enough for Rossmann destroys house-prices and Porto Seguro, where
every raised budget loses. "Raise `n_estimators`" is not the fix and would have
been a leaderboard-fit to Rossmann (CLAUDE.md, and 0022 one level up).

**Settled: early stopping wins or ties on all five readings** and is never worse
than the incumbent — including on the two datasets where more capacity is
actively harmful, because it declines to spend it. It is the only arm here with
that property, and it needs no per-dataset knowledge.

**Not settled: what the harness does about it.** Early stopping needs a
validation block the fit is allowed to see, and this harness has strong opinions
about which rows a fit may look at (0023, 0029, 0041, 0043). The arm measured
here carves the tail 10% off the work half, which is defensible for an ordered
split and needs an argument for an unordered one. That is a design decision with
consequences for the sealed holdout and for what a bundle stores, and it does not
get made inside a measurement record.

**Not settled, and now urgent: what this does to sixteen runs of results.** Every
table comparison in `_runs/` was made at 200 trees. The Rossmann sign reversal
says at least some of those rankings are not merely noisy but inverted. 0052 and
0053's conclusions are about *features measured by this ruler* and inherit the
problem directly.

## What this cost

The confound was nearly shipped as the conclusion. "Restaurant gains nothing,
Rossmann gains a lot, therefore rows" was one table away from being written down,
and it was wrong. The control that killed it cost four minutes.

The harness also refused the first attempt at that control: `factorize` produces a
positional code that depends on which rows are in the frame, and 0011's stability
check caught it. The refusal was correct and the message said exactly what to do.
That is a guard working, recorded here because most of this file is about an
instrument that was not.
