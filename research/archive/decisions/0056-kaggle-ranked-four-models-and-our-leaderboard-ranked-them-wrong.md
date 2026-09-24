# 0056 — Kaggle ranked four models, and the block we rank on got it nearly backwards

- **Date:** 2026-08-14
- **Status:** a measurement, on an external seal. **Confirms
  [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md)
  externally** and adds a finding 0054 could not have produced: `val` — the number
  that ranks every table in every run — ranks candidates almost independently of
  how Kaggle ranks them, while the sealed holdout ranks them perfectly. No code
  changed; what to do about it is
  [0057](0057-the-capabilities-the-agent-does-not-have.md) and a revision of 0043
  that is not written yet.
- **Related:** [0054](0054-the-budget-is-not-a-constant-and-at-200-trees-features-score-backwards.md),
  [0043](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md) (which named
  this cost and priced it the other way),
  [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md) /
  [0042](0042-the-terminal-argmax-shipped-the-worst-model.md) (why both arms of
  every pair were submitted rather than the better one),
  [0049](0049-the-last-observation-is-a-sample-of-size-one.md),
  [0050](0050-the-anchor-is-what-matters-not-the-history.md),
  [0052](0052-a-window-is-a-worse-estimator-of-a-better-idea.md),
  [0053](0053-the-fold-count-moves-a-window-and-we-cannot-say-why.md) (four earlier
  negatives on history; this is the fifth and the first external one),
  [0055](0055-a-rename-makes-every-earlier-artifact-unreplayable.md),
  [0022](../../../docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md),
  `scripts/ship_rossmann_capacity.py`, `scripts/ship_rossmann_median.py`.

## What was run

Four models, all on the same source, the same split fractions, the same seed,
scored on Kaggle's own rows. Two pairs, each pair submitted **in full** — never the
better one — so that the difference is measured externally rather than chosen
internally, which is what 0041 and 0042 forbid.

| arm | val | sealed holdout | Kaggle private | Kaggle public |
|---|---|---|---|---|
| A — bare feature block, 200 trees | 0.23331 | 0.24228 | 0.23048 | 0.23420 |
| B — bare feature block, 4,144 trees | 0.16000 | 0.15766 | **0.15166** | **0.14853** |
| full pipeline + history, 3-way blend | **0.13276** | 0.17678 | 0.18438 | 0.17444 |
| full pipeline, no history, 3-way blend | 0.14793 | 0.15841 | 0.15429 | 0.16193 |

## 1. Capacity is confirmed on rows nothing here has touched

**A against B: +0.07882 private.** 0054 measured +0.08462 on the sealed holdout and
predicted it before the submission; the external number came in 0.0058 optimistic.
On a dataset where 0043 warned that a four-month gap between our validation period
and Kaggle's test window would erode internal gains, it eroded 7% of one.

That is the first time this project has converted an internal finding into an
external measurement of the same size.

## 2. The block we rank on ranks badly; the block we seal ranks perfectly

| exam | Spearman against Kaggle private |
|---|---|
| `val` | **+0.20** |
| sealed holdout | **+1.00** |

The holdout put all four arms in exactly Kaggle's order. `val` put its own best arm
— the full pipeline with history — **third of four**, and put Kaggle's winner third
from last.

n = 4, so the coefficient itself is weak evidence. The **mechanism** is not weak:

    work      2013-01-01 -> 2015-05-01
    val       2015-05-02 -> 2015-06-15     <- ranks every table in every run
    holdout   2015-06-16 -> 2015-07-31     <- read once, ranked Kaggle perfectly
    Kaggle    2015-08-01 -> 2015-09-17

The holdout is adjacent to the test window and `val` is six weeks further back.
0043 recorded that sealing the last block "costs the freshest rows" and treated
that as a price paid for an unbiased final read. This says the freshest rows were
not a price — they were the only block that ranks the way the world does, and we
spent them on a single confirmatory number while ranking on the stale ones.

**This is 0022 at the level of the split rather than the metric.** The ranked
quantity is not what we are graded on, and the difference is not noise: it changes
the winner.

## 3. History is refuted externally, on the fifth attempt

**−0.03009 private** for adding `fit_history` to an otherwise identical pipeline,
having *gained* 0.015 on `val`. 0049, 0050, 0052 and 0053 each found this feature
class wanting on internal evidence and each stopped short of a verdict. This is
the external one.

The competition winner's own documentation says the same thing from the other
direction: his "month-ahead models", which discard the most recent month of data
entirely when computing features, were *"almost as good"* as the models that used
it, and he concluded the series is *"a fairly stationary process"* where *"recent
changes are not very important"*. The feature class we spent three slices on is
the one the winner reports getting almost nothing from.

## 4. The encoding pipeline is worth nothing once the budget is right

Arm B — no group statistics, no frequency encoding, no target encoding, no row
filter, **one** model — beats the full three-way blended pipeline (0.15166 against
0.15429). At 200 trees that same pipeline was worth about 0.095.

So 0054's sign reversal is not a curiosity at the margin. The fixed budget was not
merely mis-ranking feature work; **it was manufacturing its apparent value.** Every
table comparison in `_runs/` was made at 200 trees.

## What this does not show, and an error of judgement

**Nothing here beats what we already had.** Our best submission remains v7's blend
at **0.13813 private / 0.12932 public**, fitted at 200 trees, and every arm above
is worse. The public median is 0.11677 and we are not close to it.

Part of that is a mistake worth recording. The rebuild targeted **v8's** chain,
because `t24` had the best *probe* score in the runs directory (0.136692). But v7
is the better run — v7's best val was 0.12980 against v8's 0.13669 — and v7 is what
scored 0.13813 externally. **The rebuild target was chosen with the instrument this
record shows is unreliable**, which is the same error the record is about, made
while writing it.

## What follows

* The sealed holdout should be ranking candidates, not confirming one. That
  contradicts 0041's design and prices 0043's trade-off the other way; it needs its
  own record and probably a third block.
* The capacity fix is external and settled. Where its early-stopping block comes
  from is still open (0054).
* `fit_history` should not be in the vocabulary in its current form. Five readings,
  the last external, and the winner's own report agree.
* What the agent is actually missing is a different list, and it is
  [0057](0057-the-capabilities-the-agent-does-not-have.md).
