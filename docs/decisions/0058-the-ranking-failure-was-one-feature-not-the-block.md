# 0058 — The ranking failure was one feature, not the block

- **Date:** 2026-08-14
- **Status:** a measurement. **Revises the mechanism in
  [0056](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
  and the build order in
  [0057](0057-the-capabilities-the-agent-does-not-have.md); leaves 0056's
  submissions and 0057's inventory standing.** No code changed. What it mainly
  does is *withdraw* a recommendation before anything was built on it.
- **Related:** [0056](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md),
  [0057](0057-the-capabilities-the-agent-does-not-have.md),
  [0049](0049-the-last-observation-is-a-sample-of-size-one.md) and
  [0050](0050-the-anchor-is-what-matters-not-the-history.md) (which already
  contained the actual explanation),
  [0043](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md) (whose pricing
  0056 tried to overturn and which stands),
  [0029](0029-the-harness-owns-the-exam.md),
  [0053](0053-the-fold-count-moves-a-window-and-we-cannot-say-why.md) (the same
  shape of correction, on the same kind of over-read).

## The question

0056 measured that across four submitted models, `val` ranked at Spearman **+0.20**
against Kaggle while the sealed holdout ranked at **+1.00**, and concluded that the
block we rank every table on is the wrong one. 0057 then put "fix the ranking
block" ahead of every capability on its list.

Ulf, reading both: *"but this is Rossmann specific no? so we have to be careful
hardcoding anything to our harness/agent or am I missing anything."*

He was not missing anything. Three checks, and the conclusion does not survive any
of them.

## 1. The failure is one arm, not a property of `val`

| | Spearman vs Kaggle private |
|---|---|
| all four arms — `val` | +0.20 |
| all four arms — holdout | +1.00 |
| **history arm removed — `val`** | **+0.50** |
| history arm removed — holdout | +1.00 |

With the history model dropped, `val`'s only remaining disagreement with Kaggle is
a swap of two models separated by **0.00263** on Kaggle — and by **0.00075** on the
holdout, which is no better determined. `val` is not ranking badly. It ranked one
model badly, and that model dominated a four-point correlation.

## 2. That model had a documented reason to behave exactly that way

`fit_history` freezes each group's past at a single boundary date — here
**2015-05-01** — and emits the age of that frozen value. The three exams sit at
increasing distances past the freeze:

    val       starts 2015-05-02   —  1 day past the freeze
    holdout   starts 2015-06-16   —  6 to 13 weeks past
    Kaggle    starts 2015-08-01   — 13 to 20 weeks past

So the feature is at its freshest exactly where `val` measures it and staler at
every subsequent exam. **A feature whose value decays with distance from its anchor
will be over-rated by the nearest block, whichever block that is.** That is 0049
and 0050, already written, and it is a statement about the *feature*, not about the
evaluation block.

Swapping which block ranks would not fix this. It would move the over-rating from
one date to another.

## 3. The recency story does not survive the composition check

0056's mechanism was that the holdout wins because it is temporally adjacent to the
test window. The blocks say otherwise:

| block | from | to | closed | promo |
|---|---|---|---|---|
| work | 2013-01-01 | 2015-05-01 | 17.0% | 38.1% |
| **val** | 2015-05-02 | 2015-06-15 | **20.8%** | **35.6%** |
| holdout | 2015-06-16 | 2015-07-31 | 12.8% | 41.3% |
| Kaggle test | 2015-08-01 | 2015-09-17 | 14.6% | 39.6% |

`val` is the outlier on both margins — the **highest** closure rate and the
**lowest** promotion rate of any block — because May 2015 in Germany carries an
unusual run of public holidays. The holdout resembles the test window far more
closely in composition, not merely in date.

And drift is not doing the work: mean sales over successive 45-day blocks of the
work half run 7240, 6831, 7274, 8022, 6645, 6748, 6858, 7076 — no trend for
recency to exploit.

So the holdout's advantage here is substantially **a calendar accident of one
six-week window**, which is as dataset-specific as a finding gets.

## 4. And the statistics never supported it

Four models, and not four independent ones — two pairs, each differing by a single
change. A random ordering of four items is exactly correct **1 in 24** times, or
4.2%. A single perfect ordering at n=4 is not evidence for changing how every run
in the project is ranked.

## What is retracted

* **0056's section 2 conclusion** — "we rank every table on the wrong block", and
  its claim that 0043 priced the seal the wrong way round. 0043 stands.
* **0057's build order**, which put "fix the ranking block" first and everything
  else behind it. That ordering is withdrawn.

0056's four submissions and its other three findings are untouched: capacity
confirmed externally at +0.0788, history refuted at −0.0301, and the encoding
pipeline worth nothing at a proper budget. Those rest on differences ten to thirty
times larger than the ones this record is about.

## What survives, and it is smaller and genuinely general

**A composition diagnostic.** Nothing in the harness checks whether the block it
ranks on resembles the rows the model will serve. `serving_columns` already
declares the serving schema, so the marginals of shared columns can be compared
with no external label and no modelling judgement. "Your evaluation block is 21%
closed where the serving frame is 15%" would have flagged this in one line, on any
dataset, naming none. Per 0029 it reports and decides nothing.

**Ranking stability across blocks.** On an ordered split, several evaluation blocks
can be cut at increasing distance and the candidate ranking compared across them. A
ranking that reorders as the block moves is a finding in itself — and it would have
caught the history arm here **without Kaggle**, since that is precisely the arm
whose position moves. This is the internal version of what 0056 needed an external
seal to see.

Neither is a change to which block ranks. Both are instruments, which is what this
project is for.

## The revised order

1. **Lead-time-restricted aggregates** (0057 item 1) — the only capability found
   independently twice, from a second dataset and from a competition winner's
   writeup, and the one with a natural experiment already sitting in
   `restaurant_visitor`.
2. **Per-group trend extrapolation** (0057 item 3) — justified by a property of the
   model class rather than an observation about any dataset.
3. **The two instruments above**, when a run needs them rather than pre-emptively.
4. Everything else as 0057 has it, with its item 4 still gated on 0053.

## The lesson worth keeping

The over-read was made **in a record whose subject is over-reading from a bad
instrument**, and its author then chose a rebuild target with the same instrument
(0056's own closing section). Two errors of the same kind, in one session, one of
them inside the write-up of the other.

What caught it was not a test. It was being asked whether the finding was specific
to the dataset it came from — which is the question `CLAUDE.md` already tells us to
ask of every capability, and which is evidently just as necessary for every
*finding*.
