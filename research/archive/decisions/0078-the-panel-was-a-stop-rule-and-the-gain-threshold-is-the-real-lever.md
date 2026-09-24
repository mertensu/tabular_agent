# 0078 — The panel was a stop rule wearing a ruler's clothes, and the threshold is the real lever

- **Date:** 2026-08-18
- **Status:** accepted (a measurement). **Supersedes
  [0077](0077-several-disagreeing-rulers-rather-than-one-better-one.md)'s gate**,
  which is reverted the same day it landed; 0077's reasoning stays put because the
  way it was wrong is the useful part. `panel.py` and its tests survive, unwired.
- **Decided by:** Ulf — *"unfortunately, i think we should revert"*, and then the
  three replacements this record measures
- **Related:**
  [0077](0077-several-disagreeing-rulers-rather-than-one-better-one.md) (what was
  built, and the registered prediction it failed),
  [0073](0073-the-ruler-is-mispriced-on-one-dataset-and-three-mechanisms-failed-to-say-why.md)
  (whose saved sweeps are the ground truth used here),
  [0017](0017-credit-by-ablation.md) (why importance is refused, and the
  counterexample that settled it),
  [0028](0028-the-yardstick-is-fixed-the-bar-is-not.md) (the open bar — this
  records that on ordered runs there is no bar at all),
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (`block_sd`,
  the yardstick that *is* computed on ordered splits and gates nothing),
  [0034](0034-the-tiebreak-was-tested-and-rejected.md) (no constant fitted to a
  handful of observations),
  `fraud_detection/winner.md` cells 12 and 17.

## What was measured, and it refutes 0077

Every one of the 24 IEEE-CIS steps was replayed through 0077's four schemes.

| | |
|---|---|
| steps accepted by the panel | **1 of 24** |
| positive-gain steps accepted | 1 of 10 |
| registered prediction | 60-90% rejection |
| **actual** | **96% rejection** |

The distribution is the finding, not the rate:

    n_agree   0/4  ######    6
              1/4  ####      4
              2/4  ######### 9
              3/4  ####      4
              4/4  #         1

    mean per-scheme agreement with the headline: 0.396   (a coin flip is 0.500)

> **On this data the schemes behave close to independent coin flips about a step's
> sign.** An all-or-nothing vote over four of them rejects nearly everything
> whether or not the step is real, so the gate was not discriminating — it was
> rejecting uniformly. With it in place, patience would have been hit after **8**
> tables instead of 24.

That the mean sits *below* 0.5 is worth keeping: schemes that train on less data
and forecast across gaps disagree with the headline more often than chance, which
is 0073's inflation showing up in a second instrument without a capacity sweep.
The panel measures something real; the binary vote was the wrong summary of it.

## The porting error, which is the transferable part

Re-reading the winners' notebook after the measurement shows the mistake, and it
was legible before the build:

> *"The procedure for engineering features is as follows. First you think of an
> idea and create a new feature. Then you add it to your model and evaluate whether
> local validation AUC increases or decreases."* (cell 12)
>
> *"I evaluated features by training on the first 75% of the data and predicting
> the last 25%."* (cell 17)

**One scheme, trained on 75% of the data**, for the exact question 0077 pointed
four schemes at — three of which train on a sixth to two thirds. They ran three
instruments at three questions:

| instrument | question | bar | their rejection rate |
|---|---|---|---|
| one local validation, train 75% | is this step worth keeping? | any improvement | — |
| "time consistency", **one column at a time** | is this column actively harmful? | AUC > 0.5 | **8%** (19 of 242) |
| several geometries, GroupKFold, the LB | is the whole approach sound? | human judgement | n/a |

0077 collapsed all three into one automated gate and aimed it at the hardest. Their
8% against its 96% was the tell.

> **A technique's value lives in the pairing of instrument to question, and a
> writeup describes the instrument while leaving the pairing implicit.** This is the
> third time this project has ported a lever from a description and lost — 0069 and
> 0070 are the other two — and the cause has been the same each time.

## The three replacements Ulf proposed, measured

Ground truth is 0073's saved sweeps: `_runs/ruler_deflation_across_runs.csv`, every
step of every run in `_runs/` scored at the ruler's budget and at capacity. A step
"really helped" if its gain is still positive at capacity. Porto Seguro is excluded
because 0073 established its gains are all under its own noise; that leaves **42
steps across four runs, 24 of which helped**.

### Importance / SHAP share — refused, and 0017 already had the argument

0017 rejected importance for *credit* with a case that decides this too: four
columns held **77% of the splitting and contributed nothing**, while three date
columns held 23% and carried the entire improvement. Its framing generalises —
importance describes **one model's attention**, and the question is a
**counterfactual**: what would the score be if these columns had never been built.
SHAP attributes output rather than splits and is the better descriptive measure,
but it answers the same descriptive question, and two redundant columns split its
mass while neither adds anything.

What is new is the price of the swap. The ablation gain's *sign* is 73-92% accurate
against capacity (below), so replacing a counterfactual with a proxy is a step down
from a measured baseline rather than a lateral move.

**Where it would fit is a different job**: the winners' per-column time-consistency
test asks whether a *column* transfers across time, not whether a *step* helped.
That instrument is still unbuilt.

### A gain threshold — supported, and it is the strongest lever here

Gain size does predict survival:

    median |gain| among steps that helped at capacity   0.00704
    median |gain| among steps that did not              0.00221
    steps scored > 0.010   ->  91% helped  (10 of 11)
    steps scored 0 - 0.002 ->  40% helped  ( 2 of  5)

| rule | accepts | precision | recall |
|---|---|---|---|
| current — `gain > 0` | 30 | 0.73 | **0.92** |
| **0.25 x best gain so far** | 16 | **0.88** | 0.58 |
| 0.50 x best gain so far | 12 | **0.92** | 0.46 |
| 0.5 x median positive so far | 15 | 0.80 | 0.50 |

So *"we want a meaningful gain, not fishing for 0.0001"* is measurably right, it
costs **no extra fits**, and it lifts precision from 0.73 to 0.88.

**And it breaks on one dataset, structurally.** Per-run accepts under the quarter
rule:

    house_prices_v9   7/9    (7 really helped)   ok
    rossmann_v8       7/15   (8 really helped)   ok
    restaurant_v1     1/16   (7 really helped)   NO

Restaurant had one transformative early step which **ratcheted the bar for the rest
of the run** and killed six real gains. The median anchor does not fix it (2/16).

> **Anchoring on the best-so-far encodes an assumption — that later gains should be
> comparable to early ones — and diminishing returns are legitimate rather than
> evidence of fishing.** That is why this is recommended as a *report*, not a gate:
> the agent knows it is twelve steps into a chain and a constant does not.

A methodological note, because it nearly became a finding: an earlier pass computed
the anchor across the concatenated runs rather than per run, which leaked one
dataset's best gain into another's bar and produced a spurious precision of 1.00.
The corrected figures are the ones above.

### And the hole nobody proposed, found on the way

`stop._noise` returns `probe_se`, or **0.0** when it is absent. `probe_se` is a
bootstrap over validation rows, and it is deliberately refused on ordered splits
(0059: resampling rows inside one period says nothing about *which* period it is).
So on every ordered run:

    ieee_v1        probe_se None on 24/24 tables | block_sd present on 24/24
    restaurant_v2  probe_se None on 13/13 tables | block_sd present on 13/13

`min_gain_ratio` multiplies zero. **There is no bar at all on Rossmann, Restaurant
or IEEE-CIS** — every positive gain counts as progress, which is exactly the
fishing Ulf's question was about. The docstring says so and calls it "the honest
failure direction", which it was when nothing better existed.

Something better now exists and is already computed on those runs: `block_sd`, from
0059's interval — a noise estimate for **the difference**, which is the quantity
being gated, not for the score. 0028 left "the bar it is a quarter of" open; this
records that on the datasets this project actually cares about, the bar has been
zero the whole time.

## What is decided here, and what is not

**Decided:** 0077's gate is reverted. `panel.py` and its tests stay, unwired,
because the machinery is what the two justified follow-ups need. Importance and
SHAP are refused as a step filter, on 0017's argument plus the accuracy baseline
above.

**Not decided, and deliberately left to its own record:** whether the gain
threshold is reported or enforced, and whether `block_sd` becomes the yardstick in
`_noise`. Both change what every run does, both are cheap, and neither should be
chosen on four datasets without registering a prediction first — which is the
mistake this record exists to document.
