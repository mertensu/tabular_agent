# 0060 — The sealed block is optimistic at a horizon it cannot reach

- **Date:** 2026-08-15
- **Status:** a measurement, on a second dataset, against an external seal. No
  code changed.
- **Confirms externally:** the limitation
  [0059](0059-the-interval-belongs-to-the-difference-not-the-score.md) states in
  its own module docstring and measures on Rossmann — that an evaluation window
  cannot see an effect which changes beyond its far edge.
- **Related:** [0043](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md)
  (whose trade-off this prices on a second dataset),
  [0041](0041-the-sealed-split-scores-the-ballot-not-the-winner.md),
  [0057](0057-the-capabilities-the-agent-does-not-have.md) (item 1, which this
  supports but does **not** establish),
  [0058](0058-the-ranking-failure-was-one-feature-not-the-block.md),
  [0029](0029-the-harness-owns-the-exam.md).

## The run

`scripts/run_restaurant_v1.py`, Recruit Restaurant Visitor Forecasting, the first
agent run on the second forecasting dataset. Rossmann v8's configuration verbatim
— model, route, thinking budget, both split fractions, fit budget, stop policy,
prompts — so only the data differs. 17 tables, 5 terminal fits, 21.9 minutes, the
agent finalized itself at 17 of 25 allowed tables.

Five predictions were registered in the script's docstring before it ran.

## The measurement

| | rmsle |
|---|---|
| validation (lgbm, selected) | 0.52099 |
| **sealed holdout, blend — what shipped** | **0.50176** |
| **Kaggle public** (nearer half of the forecast window) | **0.51301** |
| **Kaggle private** (further half) | **0.54398** |

The competition splits its test set **on time, public fold first, private fold
following**. So those last two rows are the same model at two horizons, and the
ordering of all four is the whole finding:

    sealed  ->  public   +0.01125     sealed was optimistic
    sealed  ->  private  +0.04222     sealed was optimistic
    public  ->  private  +0.03097     this model decays with horizon

Both boards, 2,150 teams:

| | best | top 10% | median | our rank | our gap to median |
|---|---|---|---|---|---|
| public | 0.46535 | 0.47800 | 0.48921 | ~1544 (top 72%) | +0.02380 |
| **private** | 0.50128 | 0.51817 | **0.53223** | **~1369 (top 64%)** | **+0.01175** |

**We are 0.0118 from the median on the board that counts**, not the 0.0238 the
public board suggested — and 0.0258 from the top 10%. For scale, 0054 measured the
store-identity-plus-capacity mechanism on this exact dataset at **+0.0283**. One
already-measured mechanism is arithmetically enough to clear both.

## Why this matters more than the score

**Our validation half was pessimistic and our sealed block was optimistic, at the
same time, about the same model.** Optimism from val to sealed was *negative* for
all five candidates (−0.012 to −0.026): the sealed rows looked *easier* than the
validation rows. Read on its own, that says the exam is honest — it is the same
reassuring signature Rossmann v5 produced, and 0032 recorded it as 0029 paying off.

It was not honest. It was measuring a horizon nobody was going to be scored at.

The four windows sit in strict time order — validation, sealed, public, private —
and performance degrades monotonically across all four. Each window is a fair
reading *of itself*, and each one flatters the next. There is no defect here to fix
in the split: 0041's seal did exactly what it promises, and 0043 already priced the
fact that it costs the freshest rows. What is new is the **size** of the residual
error at the horizon that counts — **+0.042**, on a board whose entire span from
median to first place is 0.031.

The practical consequence, and it applies to every ordered run this project makes:
**a sealed score is a lower bound on the error at the horizon you will be graded
at, not an estimate of it.** Rossmann v5 read its own negative optimism as evidence
the exam was honest (0032); this run reproduces that signature and then shows what
sits beyond it.

## What this confirms, and it was predicted in writing

0059 shipped a block instrument with a limitation stated in its docstring before
any of it was measured:

> blocking samples the periods it has, and if an effect changes sign beyond the
> far edge of the evaluation window every block can agree and the effect can still
> fail on the rows that matter.

On Rossmann that cost us a feature: five of five blocks favoured a history table
that lost 0.0301 externally, and only a 90-day window exposed the decay. This run
is the same failure on a second dataset, a different metric and a different
feature set — and this time the degradation is measured *directly*, as the gap
between two halves of one test block scored on the same model.

**Registered prediction 2 is confirmed in direction and refuted in mechanism**, and
having both boards is what separates them.

Its direction was right: the far half of the window is 0.031 worse than the near
half. But joining the two boards on team gives the field's own answer to the same
question —

    mean(private - public) across 2,099 teams:  +0.04464
    ours:                                       +0.03097
    Spearman, public vs private rank:            0.928

**Every team degrades with horizon, and we degrade less than average.** Our rank
*improves* from top 72% public to top 64% private, precisely because the field
lost more than we did going into the further half. So the decay is a property of
the task — the far half is simply harder to forecast — and carries **no evidence
whatsoever** for the mechanism the prediction named.

That mechanism was the cut reservation stream (30% of training reservations booked
same-day; no forecast row can carry one, and the floor grows to 39 days at the far
edge). Six such columns did reach the shipped model, so the exposure is real. What
is now clear is that this measurement cannot speak to it: a decay we share with
2,099 other teams, most of whom never touched those columns, is not our skew
showing through. Establishing it needs the same run with the reservation columns
withheld — one script, not yet written.

Recorded this way deliberately. Read against the public board alone, the direction
fit and the story was available; the private board turned a "confirmed" into a
"confirmed, but not for the reason I gave". 0052 and 0053 are what happens when
that check is not available and the plausible mechanism gets adopted anyway.

## The other registered predictions

**1 — confirmed.** Six reservation features in the shipped model, 67 features
total. Nothing refused them, correctly: they genuinely exist at predict time, and
`serving_columns` is about presence, not provenance (0030).

**3 — confirmed. 6 of 17 comparisons came back `sign_stable: False`**, the first
time the block mode has fired anywhere. Every unstable table has |gain| below
0.003 against a block sd two to five times larger; two are outright negative. The
instrument separated the four real steps — target encoding at **+0.190**, then
+0.034, +0.032, and history at +0.013, all 5/5 stable — from the tail cleanly.

**4 — untested, still open.** The agent built 90/180/365-*day* windows on a
**ragged** spine, where `fit_history` counts rows. Nothing caught it and nothing
measured it. This is a real gap in the tool, not in the run.

**5 — confirmed.** Below median, as registered.

## The thing nobody predicted: the agent used the new instrument

The run stopped at 17 of 25 permitted tables, and its own closing summary gives
the reason:

> *"Gains toward the end were below the probe's own standard error and not
> cross-block-stable — competition-density and extra holiday-interaction
> encodings did not robustly help, so I stopped rather than overfit."*

0059's interval was eight hours old and had never been seen by an agent. Patience
cannot explain the stop: on an ordered split `probe_se` is `None`, so `stop._noise`
floors the bar at zero and every positive gain still counts as progress — t10, t13,
t14 and t17 were all positive. The run was not forced to end and ended anyway.

Worth weighing against 0027, which recorded three consecutive failures of advisory
text to change behaviour and concluded that whatever solves it would be
structural. This is advisory information that did change behaviour. **n=1**, and it
should be read as one observation, not a refutation.

Also present: the agent read the sealed score after `finalize_run` and narrated it
("*Holdout RMSLE 0.502 … no optimism*"). Harmless by call ordering, not by
structure, and named as open since 0042.

## What was deliberately not done

**The submission was not clipped.** One prediction of 32,019 came out at 2868.5
where the training maximum is 877 and p99.9 is 115 — worth roughly 0.0015 of
rmsle on a board whose median-to-top-10% gap is 0.011. Flooring it after seeing
the score is the post-hoc correction CLAUDE.md and 0022 exist to refuse. A general
"predictions outside the training range" diagnostic names no dataset and would be
legitimate; doing it to this file would not.

**No conclusion is drawn about which block should rank.** 0058 settled that and
0043 stands. Nothing here argues for moving the seal, shrinking it, or ranking on
it — the finding is that *every* window we own sits nearer than the one we are
scored at, which is a property of forecasting and not a bug in the split.
