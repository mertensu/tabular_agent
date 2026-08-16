# 0067 — Capacity is worth a tenth of that on a table that has features

- **Date:** 2026-08-16
- **Status:** accepted. **A measurement**, and it narrows
  [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
  sharply without contradicting any number in it. It also refutes a prediction I
  registered in `scripts/probe_capacity_v7_chain.py` and — more importantly — the
  priority argument I had just given Ulf on the strength of 0054's numbers.
- **Related:** [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
  (the readings this narrows), [0061](0061-we-were-measuring-against-the-wrong-board.md)
  (which named this as the untested cell, and fixed the reference line),
  [0056](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
  (capacity confirmed externally on a bare table),
  [0055](0055-a-rename-makes-every-earlier-artifact-unreplayable.md) (why this
  could not be measured in 0054), [0039](0039-seguro-v3-two-defects-and-a-falsified-hypothesis.md),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md).

## The question

0054 measured model capacity on two Rossmann tables and 0061 pointed out that
neither is the one that ships:

> The untested cell is still **v7's chain at a real budget**.

It could not be measured then. v8's `t24` had been broken by 0051's rename (0055),
so 0054 rebuilt only the pure-code part of it. v7's `t13` predates `fit_history`
entirely — eleven steps of transforms, a row filter and target/frequency encoders
— so it replays at HEAD, and that is what made this possible now.

The question it decides: **do capacity and feature work add up, or is the feature
work doing capacity's job by hand?**

## The measurement

`scripts/probe_capacity_v7_chain.py`, same grid, same monkeypatch and same
early-stopping arm as 0054's sweep — imported from it rather than restated, so the
rows are comparable. v7's split copied rather than recut.

| table | 200 trees | best budget | gain |
|---|---|---|---|
| baseline (raw columns, `Customers` dropped) | 0.21810 | 0.15749 | **+0.0606** |
| v8's engineered feature block (code only) | 0.23331 | 0.15213 | **+0.0812** |
| **v7's `t13` — the chain that shipped** | **0.13202** | **0.12520** | **+0.0068** |

Full sweep on `t13`: 1000@0.10 +0.0057, 1000@0.03 +0.0018, 3000@0.03 +0.0058,
5000@0.03 **+0.0068**, early stop (1195 rounds) +0.0028.

> **Capacity's payoff falls by about 90% once the table has the features.**

## What this settles

**They substitute, and strongly.** Both mechanisms reach the same structure: v7's
chain encodes per-store, per-store-weekday and per-store-promo effects
*explicitly*, and capacity lets a tree ensemble find them by splitting. Given one,
the other has little left to buy.

That was the direction of the registered prediction (+0.010 to +0.030) and the
magnitude was wrong — it came in at **+0.0068**, below the range and only just
above the +0.005 that would have falsified the line entirely.

**But they do not converge.** The three tables at their best budgets sit at
0.15749, 0.15213 and **0.12520**. Capacity on a bare table does *not* reach what
the feature chain reaches. The feature work is worth about 0.03 rmspe that
capacity cannot recover, and capacity is worth 0.0068 that the features do not.
Neither is a substitute for the other in the sense of "pick one"; the point is
that the *second* one you apply is worth far less than it was on its own.

## What this refutes, and it is mine, twice

**The prediction**, above: right in direction, wrong in size.

**And the priority argument built on 0054.** Asked an hour earlier how this
project gets better, I answered that capacity was measured at "three to seven
times the gap to the median" and should be priority one. That number came from the
**bare-table** readings. On the table that actually ships it is **0.8x** the gap —
+0.0068 against v7's +0.00846 from the private median.

The arithmetic that follows is the part worth carrying: if the whole +0.0068
transferred to Kaggle — and 0056 showed capacity gains do transfer — v7's private
0.13813 would become **0.13133** against a private median of **0.12967**. Still
short.

> **Features and capacity are both measured now, and together they do not close
> the gap to the median on Rossmann.**

## What follows

Not a new priority list. What follows is that **the two mechanisms this project
has spent sixteen runs on are now both bounded**, and the remaining gap is made of
something not yet identified. Candidates, none tested and none endorsed here:

- **one model, not many.** The harness fits a single model over all rows; the
  competition winner fit per-segment models. Nothing here can express that.
- **no row weighting of any kind.** Not recency, not anything.
- **every hyperparameter except `n_estimators` and `learning_rate` is a library
  default** — depth, leaves, feature fraction, all untouched, in every run. 0017
  forbids moving four at once, which is an argument for how to test it, not for
  leaving it.
- **the blend** (0044/0045) has never been compared against a tuned single model.

Choosing among these needs an argument, not a guess, and it should not be made in
this record.

**Do not raise the constant.** 0054's rule is untouched by any of this and the new
row strengthens it: early stopping on `t13` chose 1195 rounds and delivered
+0.0028, well short of the fixed 5000@0.03's +0.0068. So the inner-tail
early-stopping arm **under-shoots on a well-featured table**, which is direct
evidence about 0054's open item 2 — *where the early-stopping block comes from* —
and evidence that the obvious answer is not good enough.

## The caveat that keeps this honest

This is a **probe** score on v7's validation half: one lgbm, one table, one
dataset. What ships is a three-way blend, and 0056 established that the probe
ranks unreliably at 200 trees. The direction is trustworthy — the same sweep on
three tables shows a clean ordering — and the exact 0.0068 is not a Kaggle
prediction. No submission was made; the Rossmann private board is spent (0045).
