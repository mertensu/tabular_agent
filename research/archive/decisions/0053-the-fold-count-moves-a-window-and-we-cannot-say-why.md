# 0053 — The fold count moves a window by 0.14, and we cannot say why

- **Date:** 2026-08-14
- **Status:** a measurement. **Revises the mechanism in
  [0052](0052-a-window-is-a-worse-estimator-of-a-better-idea.md); leaves its
  measurements and its conclusion standing.** Code changed: the warnings built on
  the refuted mechanism are demoted to factual reporting.
- **Related:** [0052](0052-a-window-is-a-worse-estimator-of-a-better-idea.md),
  [0046](0046-the-mean-is-a-lossy-summary-of-a-group.md) (the all-time aggregate,
  which turns out to be the *stable* one), [0022](../../../docs/decisions/0022-the-yardstick-must-measure-the-ranked-quantity.md)
  and [0036](0036-the-pairing-asymmetry-was-not-there.md) (a ranked quantity
  that moves with the ruler), [0029](0029-the-harness-owns-the-exam.md) (the exam
  is ours, and here it is part of the answer),
  [0031](0031-the-guard-was-decorative.md), `scripts/probe_rossmann_window_length.py`.

## The question

Ulf, reading 0052: *"but the 136 for rossmann are only due to how we split the
data into folds no?"* Yes. Forward chaining gives fold 0 a prefix of roughly
`span / (n_splits + 1)`, so 0052's "shortest training prefix = 136 days" is an
artifact of `n_splits=5`, not a fact about Rossmann. If the mechanism 0052 named
is real, **changing that knob should move the ceiling.**

## What happened

The same length sweep at `n_splits=3` and `n_splits=5`, everything else identical.
Both runs reproduce to the digit, so nothing here is fitting noise.

| feature | `n_splits=5` | `n_splits=3` | swing |
|---|---|---|---|
| all-time target mean of `Store_DOW` (0046) | +0.06048 | +0.06004 | **0.0004** |
| trailing window 364 | +0.04458 | +0.05338 | 0.0088 |
| trailing window 546 | −0.00030 | +0.04078 | 0.0411 |
| trailing window 728 | **−0.09287** | **+0.04654** | **0.1394** |

**The 728-day window is a disaster under one fold count and a good feature under
another.** Its train/serve observation counts are *identical* in both — a median
of 59 against a served 101, a ratio of 1.71x. So the quantity 0052 blamed did not
move at all while the outcome reversed.

## 0052's mechanism is refuted

0052 said: a window longer than the shortest fold's training prefix is unfillable
in training and full at serving, and *that* is what costs the score. Three
independent checks kill the causal half of it.

1. **The ratio does not move while the score swings 0.139.** 59 against 101 in
   both configurations.
2. **The ratio is anti-correlated with the score across pairs.** `w=546` at
   `n_splits=5` has a ratio of 1.29x and gains nothing (−0.00030); the same window
   at `n_splits=3` has a *worse* ratio of 1.39x and gains +0.04078. Likewise
   `w=364` (1.26x, +0.045) against `w=546` (1.29x, −0.000) — near-identical ratios
   either side of a 0.045 gap.
3. **The two-level guard cannot discriminate.** At `n_splits=3` the prefixes are
   204/408/612, so `w=728` exceeds *every* prefix there too. The condition
   "no fold can fill this window" is true in both configurations — the one that
   scores +0.047 and the one that scores −0.093. A guard that fires identically on
   both is not conservative, it is uninformative.

The **arithmetic** in 0052 stands: the fold prefixes are what they are, the
per-fold counts really do ramp 20/39/59/78/93 against a served 101, and the
trained and served columns really do summarise different numbers of observations.
What does not follow — and what 0052 asserted — is that this is what damages a
model.

## What 0052 got right, and it is the important part

**The conclusion is unaffected and is now better supported.** The all-time target
mean of the same key beats *every* window under *both* fold counts:

    all-time  +0.06048 (n=5)  +0.06004 (n=3)
    best window       +0.04458        +0.05338

So slice 3 still buys nothing over 0046, and that holds under a fold count it was
never measured at. 0052's central finding survives its own explanation being
wrong.

## What replaces it: a property of the feature class, not a mechanism

The one thing this run establishes cleanly:

> **A trailing window's measured value is substantially an artifact of the fold
> count, and its sensitivity grows with the window's length. The unbounded
> aggregate it competes with is invariant.**

0.0004 for the all-time mean against 0.1394 for a 728-day window, monotone in
length across 364 / 546 / 728. That is a statement about the feature class and it
is actionable without knowing the route: **window length cannot be searched over
in this harness and the number believed**, because a large part of the score
belongs to a CV setting rather than to the data. It is 0022's problem — a ranked
quantity that moves with the ruler — arriving in a feature rather than in a
metric.

**No replacement mechanism is offered.** Several were considered and none was
tested: fold-count-dependent NaN share (`n_splits=3` blanks 25% of rows against
`n_splits=5`'s 17%), the number of distinct coverage regimes in the trained
column, and interaction between the ramp and the target's own time trend. Naming
one of these as the cause after the ratio story failed would be replacing a
refuted explanation with an unrefuted one, which is worse than saying we do not
know. This is the third time in one session that a plausible story about these
columns has failed a check — 0050's `summarise_history` verdict, 0052's coverage
count, and now 0052's prefix mechanism — and the pattern is more instructive than
any of the individual stories.

## What changed in the code

The `shortest_training_prefix` / `longest_training_prefix` numbers **stay**, for
the reason `block_span` exists: the agent cannot obtain them otherwise, and they
describe real geometry. What goes is the **prediction attached to them.** The note
no longer says a window past a prefix will hurt; it says the columns summarise
different numbers of observations, that whether this costs anything is unsettled,
and that the probe score is the thing to trust over the diagnostics.

That is a tool reporting what it knows and declining to advise on what it does
not, which is the honest shape for a number this session has twice been wrong
about.

## The cost of finding out, and the thing to fix first

Every number in 0052 and 0053 is a point estimate on a single ordered validation
period where `val_se` is `None` — 0022, 0036 and 0041 all say this is n=1 in the
unit that varies. The measurements reproduce exactly, so they are not *noisy* in
the seed sense, but reproducibility is not the same as a confidence interval: a
0.139 swing and a 0.009 swing are being read off the same instrument with no way
to say which is inside the ruler's own error.

**0036's paired bootstrap remains unwired, and it is now the blocking item for
this whole area.** Two records have been written on differences nobody can put an
interval around.
