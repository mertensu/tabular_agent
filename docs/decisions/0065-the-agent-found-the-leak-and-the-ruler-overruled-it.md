# 0065 — The agent found the leak, built the honest table, and the ruler overruled it

- **Date:** 2026-08-16
- **Status:** accepted. **A measurement**, from the first run with a declared
  stream. No capability is added or removed here; what it settles is what the
  capability was worth and where the remaining defect lives.
- **Related:** [0062](0062-a-stream-is-a-second-input-and-its-observation-time-is-a-column.md)
  / [0063](0063-the-grain-is-measured-from-the-table-not-required-of-the-caller.md)
  (the capability this run exercises),
  [0029](0029-the-harness-owns-the-exam.md) and
  [0035](0035-the-search-is-over-features.md) (the division of labour this run
  puts pressure on), [0030](0030-a-model-that-cannot-be-served-is-not-trained.md)
  (presence, not provenance — the gap this lands in),
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) (the
  instrument that was supposed to catch this and did not),
  [0060](0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md) (why the
  sealed score cannot settle it), [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md),
  `scripts/run_restaurant_v2.py` (the run, with its predictions registered in the
  docstring before it started).

## The run

`restaurant_v2` is `restaurant_v1` with **one change**: the two reservation files
are declared as streams, so `fit_stream_aggregate` can reach them. Model, thinking
budget, both split fractions, fit budget, stop policy, prompts and serving schema
are v1's verbatim. 13 tables, 5 terminal fits, 8.2 minutes, the agent finalized
itself.

Measured by hand before the run, on rows carrying any reservation at all: the
collapsed `air_reserve_visitors` averages **13.668** against the lead-39 column's
**0.676**. At the far edge of the forecast window the honest column is 5% of the
naive one.

## What happened

The agent found the streams, used both, and **identified the trap on its own** —
then built the honest table and was overruled by the harness.

| table | | probe | vs parent | block sd | sign_stable |
|---|---|---|---|---|---|
| t4 | stream aggregate, air, leads 7/14/30 | 0.55340 | **+0.00497** | 0.00287 | True |
| t5 | stream aggregate, hpg, leads 7/14 | 0.55546 | −0.00206 | 0.00091 | True |
| **t7** | target encoding — **this is what shipped** | **0.52135** | +0.03205 | 0.00458 | True |
| **t8** | t7 with the six collapsed reservation columns **dropped** | 0.52984 | **−0.00849** | 0.00297 | True |
| t9 | the same idea, a slightly different drop list | 0.52901 | −0.00766 | 0.00313 | True |

Work table under the same ruler: 0.76728. Baseline fit: 0.84512.

In its own closing words, unprompted:

> *"I deliberately built and tested a version without them (t8, 0.5298) to avoid
> that leak, but the harness's automatic selection of the best table chose the
> version that includes them."*

## The registered predictions

**1. Confirmed — the agent reached the stream.** Two stream tables, both files.
The surfacing (`inspect_data`'s stream profile, the brief's line) did its job;
this was not guaranteed and is the cheapest thing that could have failed.

**2. Refuted — the lead it chose.** The prediction was a lead at or near the
declared horizon of 1, because that is the number the refusal names. It asked for
**7, 14 and 30**: neither the floor nor the far edge, but a *ladder* across the
middle. That is the shape the tool invites — "several leads are separate features,
not competing settings" — and the prediction failed to consider the design's own
intended use.

**3. Confirmed, and in a sharper form than it was registered.** The prediction was
"the honest column will probe worse than the collapsed one". What the run shows is
more specific and worse:

- *adding* honest columns **helped** (+0.00497, stable across every block);
- *removing* the collapsed columns **cost 0.00849** — **2.9 block standard
  deviations, every block agreeing**.

So the ruler is not undecided. It is **confident and pointing the wrong way**. It
rewards keeping both, which is to say it rewards keeping the dishonest one.

**4. Confirmed.** The shipped blend carries **9 honest and 6 collapsed**
reservation columns out of 58.

**5. Open.** `_runs/restaurant_v2/submission.csv` is written; Kaggle has not
scored it.

## The finding

**Supplying the capability is necessary and not sufficient, and the remaining
defect is not in the tool.**

Three things had to happen for the honest column to ship, and only two did:

1. the agent has to be *able* to express it — 0062 built that, and it worked;
2. the agent has to *choose* to — it did, twice, unprompted, off `column_notes`
   alone;
3. the harness has to let that choice survive — **it does not.** `best_table` is
   the argmax of the probe, and the probe scores against a validation half drawn
   from the same uncut file. An agent that correctly diagnoses a leak has no way
   to act on the diagnosis.

This is the failure mode 0029 always risked and never met: the harness owning the
exam is right, and it means the harness's judgement is *binding* even where the
agent's is better. The natural reading of this run is not "the ruler is bad" — the
ruler is measuring exactly what it was asked to, on the rows it has. It is that
**no block of these rows can measure the thing, because every one of them shares
the cut.** 0060 said a sealed score is a lower bound on the error at the horizon
you are graded at; this is the same statement about a *comparison* rather than a
score.

### 0059's instrument did not catch it, for the second time

The whole point of the block interval was to stop a difference being believed on
one period's noise. On t8 it reports `sign_stable: True` at 2.9 block sd — the
leaky table wins in **every** block. That is not the instrument failing; blocking
samples the periods it has, and every period here has the same defect. 0059's own
docstring says so before it was measured. But it is now the second recorded case
(after Rossmann's history arm, −0.0301 externally) where the interval was
confident and externally wrong, and two is enough to stop citing `sign_stable` as
if it bore on honesty.

### What the sealed split says, and why it cannot settle this

| | val | sealed | optimism |
|---|---|---|---|
| lgbm (selected) | 0.52135 | 0.50670 | −0.01465 |
| catboost | 0.52247 | 0.50520 | −0.01727 |
| **blend — shipped** | **0.51971** | **0.50571** | **−0.01399** |

Negative on all five, exactly as v1 read before Kaggle punished it by +0.042. The
sealed block is adjacent to the work half and drawn from the same uncut stream, so
it flatters the collapsed column too. **A run cannot detect this defect from
inside itself.**

**Registered here, before the submission is scored:** if the mechanism is what
this record claims, v2 should decay *less* from sealed to private than v1's
+0.042, even though v2's sealed number (0.50571) is marginally worse than v1's
(0.50176). If it decays the same or more, the mechanism is wrong and this record
is the thing to reopen.

## What is not concluded

- **Not that the collapsed columns should be dropped.** That would be tuning
  against a leaderboard we have not read yet, and 0022 is one level up. What the
  run establishes is that *the agent's own attempt to drop them could not survive*,
  which is a statement about the harness.
- **Not that the agent "understands" the trap.** It read a `column_notes` entry
  that states the cut in plain prose, written for exactly this purpose. n=1, and
  [0027](0027-the-brief-did-not-cut-the-orientation-tax.md) is the record of three
  advisory steers that changed nothing.
- **Not which fix to build.** Three shapes are visible and none is decided here:
  an explicit channel for the agent to mark a table as not-to-ship (narrower than
  giving back the terminal choice, which
  [0035](0035-the-search-is-over-features.md) removed deliberately); extending the
  serving declaration from *presence* to *provenance*, which is the gap 0030
  leaves and is the same class of unrecoverable fact `serving_columns` already
  carries; or an evaluation block that does not share the cut, which is 0041/0043.
  Choosing among them needs its own record and probably a second dataset.

## A gotcha, and it is mine again

`run_restaurant_v2.summarise_streams` classified shipped columns as honest by
testing `"_lead" in name`. That caught `air_reserve_lead_days` — a **collapsed**
column from the prep script holding the mean booking notice, carrying the same
timing caveat as the rest of the collapse. It printed **11 honest and 4 collapsed
where the truth is 9 and 6**, in the direction that flattered the run.

Fixed to match against the columns the stream tables actually produced. 0050's
shape in the reporting layer for the third time in this project: *a script's
summary of its own run is not evidence.*
