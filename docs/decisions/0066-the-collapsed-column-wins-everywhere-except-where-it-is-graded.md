# 0066 — The collapsed column wins everywhere except where it is graded

- **Date:** 2026-08-16
- **Status:** accepted. **A measurement**, and the one this dataset was kept for
  since 0054. It refutes a prediction registered in
  `scripts/probe_restaurant_collapsed_columns.py` before either submission was
  scored — mine — and confirms
  [0065](0065-the-agent-found-the-leak-and-the-ruler-overruled-it.md)'s central
  claim, which 0065 itself had marked as reopened.
- **Related:** [0065](0065-the-agent-found-the-leak-and-the-ruler-overruled-it.md)
  (the claim, and the run that produced both tables),
  [0060](0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md) (the sealed
  score as a lower bound — now shown to be a lower bound on a *difference* too),
  [0061](0061-we-were-measuring-against-the-wrong-board.md) (which board is which),
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (the
  interval that called this stable), [0035](0035-the-search-is-over-features.md)
  and [0029](0029-the-harness-owns-the-exam.md) (the argmax that shipped the
  wrong table), [0030](0030-a-model-that-cannot-be-served-is-not-trained.md),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md),
  [0045](0045-the-cost-of-the-unfiltered-blend-was-underestimated-44x.md).

## What was run

`scripts/probe_restaurant_collapsed_columns.py`. Restaurant v2 produced two tables
one variable apart: **t7**, which carries the six precomputed
`air_reserve_*` / `hpg_reserve_*` columns, and **t8**, which is the same chain with
those six dropped and nothing else changed. The agent built t8 itself, having
diagnosed the cut off `column_notes`; the probe scored it worse and `best_table` is
the argmax, so t7 shipped (0065).

Each arm got a fresh run directory holding **a copy of v2's own split** — the same
work/val/holdout parquets, so both arms are scored on identical rows — plus its one
table and lineage. `finalize_run` did the rest: the enumerated terminal set, the
0044 equal-weight blend, the sealed score, the bundle. Nothing was reimplemented.

**Two controls, both passed.** The t7 arm reproduced v2's journalled blend val
(0.519707) and sealed score (0.505713) to six decimals, and its submission came
back from Kaggle at **0.55124 / 0.51197** — digit-identical to v2's. The path is
verified end to end, so the t8 arm is a controlled comparison rather than a
rebuild that happens to look plausible.

## The measurement

| | probe | val | sealed | public | private |
|---|---|---|---|---|---|
| t7 — six collapsed columns present | 0.52135 | 0.51971 | 0.50571 | 0.51197 | 0.55124 |
| t8 — those columns dropped | 0.52984 | 0.53093 | 0.51694 | 0.51425 | **0.54975** |
| **t8 − t7** | +0.00849 | +0.01122 | +0.01123 | +0.00228 | **−0.00149** |

Read the bottom row against how far each window sits from the end of training —
public is the nearer half of the forecast block and private the further half
(0061):

    in-sample block   +0.00849
    validation        +0.01122
    sealed, adjacent  +0.01123
    nearer half       +0.00228
    further half      -0.00149

> **The collapsed column's advantage decays monotonically with distance from
> training and reverses sign at the far half of the horizon.**

That gradient is the evidence, not the sign flip on its own. A single reversal
would be one noisy comparison; five readings at increasing distance, ordered, is
the mechanism's own prediction reproduced.

## What is refuted, and it is mine

The prediction registered in that script's docstring was that **t7 would win the
private board too**, and that the trap would turn out to be a null on this
dataset. It was argued from two prior refutations — 0060's horizon-decay mechanism
and v2's failure to decay less than v1 — and it was wrong.

The error worth naming is not the guess. It is that **the two refutations it rested
on were both uncontrolled comparisons**, and 0065 said so about the second one in
the same breath as registering the prediction. v1-against-v2 is two agent
trajectories, 17 tables against 13; it could not separate "the honest columns did
not help" from "v2 built a slightly worse model". Having written that down, I then
used it as evidence. The confound was named and then leaned on anyway.

## What is confirmed

**0065's central claim, externally, in the controlled form.** The collapsed columns
are a genuine train/serve skew on this dataset, they are worth about +0.011 rmsle
on every block that shares the cut, and they cost about 0.0015 where the
competition actually grades. The agent's diagnosis was correct and the harness's
was not.

**And every internal instrument was confidently wrong about it.** Three, all in the
same direction:

- the **probe**, by +0.00849 — and it is the argmax that decides what ships;
- the **block interval** (0059), reporting `sign_stable: True`;
- the **sealed holdout** (0023), by +0.01123 — the largest error of the three.

The seal is the project's honesty instrument, and here it was not merely
uninformative but the most misleading of the set. 0060 established that a sealed
score is a lower bound on the *error* at the horizon you are graded at. This adds
the sharper form:

> **A sealed score is also a lower bound on a *difference* — and where the defect
> is in the data's provenance rather than in the model, the seal can order two
> candidates backwards.**

Because the sealed block is adjacent to the work half and drawn from the same uncut
file, it shares the very property under test. No block of these rows could have
caught this. That is not a flaw in how the split was cut; it is what "the file was
cut once, at the forecast moment" means.

## The consequence, which is now measured rather than argued

0065 named the argmax-shipping defect and could only call it theoretical. It is not.

    the agent identified the leak, built the clean table, and the harness
    shipped the leaky one — at a measured cost of 0.00149 rmsle on the
    private board

Ulf stated the governing principle **before this measurement existed**: *"der agent
ist die Intelligenz, die harness sollte ihn nicht beschneiden. ihn nur in richtige
bahnen leiten."* That was a design conviction; it now has a number behind it.

It also settles the direction of 0065's open decision, though not its detail. Of
the three shapes named there:

- **a channel for the agent's judgement to survive** — consistent with the
  principle, and with what actually happened: the agent was right and could not
  act;
- **extending the serving declaration from presence to provenance** — the harness
  deciding instead, which is the shape the principle rules out as a *first* answer,
  though it remains useful for facts only the data's owner has;
- **an evaluation block that does not share the cut** — 0041/0043, and this dataset
  shows there is no such block available inside the file.

## The limits, stated so the number is not over-read

- **The magnitude is tiny.** 0.00149 rmsle moves the private rank from ~1446 to
  ~1434 of 2150 — twelve places, both top 67%. Both arms remain well below the
  private median (0.53223). This is a finding about **direction**, not about score.
- **n = 1**, and there is no interval on the Kaggle difference. What raises it above
  a single reading is the ordered gradient across five windows, not the final
  comparison alone.
- **It does not license dropping such columns as a rule.** That would be tuning
  against a board (0022, one level up) and would also throw away the natural
  experiment. The general statement is about the *instruments*: no block sharing a
  provenance defect can rank candidates that differ in it.
- **The Restaurant private board is now spent for this question** — 0045's
  argument, applied to a second competition. Three
  submissions have been scored against it; a fourth reading of the same claim is
  not available.

## What would change this

A second dataset with a bitemporal cut, where the same controlled pair is built and
the gradient either reproduces or does not. Until then this is one dataset's
mechanism, measured carefully, and the general claim it supports is the one about
instruments rather than the one about reservations.
