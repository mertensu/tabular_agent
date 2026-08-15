# 0061 — We were measuring against the wrong board, and the gap is a third of what we thought

- **Date:** 2026-08-15
- **Status:** a measurement and a small fix. `scripts/prep_rossmann.py` gains
  `leaderboard()` / `leaderboard_rank()` reading both boards from disk.
- **Corrects the framing of:** [0056](0056-kaggle-ranked-four-models-and-our-leaderboard-ranked-them-wrong.md)
  and every Rossmann runner's hardcoded `LEADERBOARD` constant. **Corrects no
  measurement** — every score in 0056 is right; what was wrong is the reference
  line drawn beside them. 0056 stays as written.
- **Related:** [0060](0060-the-seal-is-optimistic-at-a-horizon-it-cannot-reach.md)
  (the same error caught one dataset earlier, which is why this was checked),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md),
  [0045](0045-the-cost-of-the-unfiltered-blend-was-underestimated-44x.md).

## The error

Four Rossmann runners carried

    LEADERBOARD = {"best": 0.08932, "top 10%": 0.10294, "median": 0.11677}

taken from `rossmann-store-sales-**public**leaderboard.csv`, the only board in the
repo. The scores quoted beside it are **private**: v7's 0.13813, and all four arms
of 0056's table. The HANDOFF headline reads

> We did not beat v7 (0.13813 private / 0.12932 public) and are not near the
> median (0.11677).

which invites subtracting a private score from a public median. Nothing in the
repo stopped that, and this project has been describing itself as far below median
on the strength of it for several sessions.

**This was found because 0060 made the identical mistake on Restaurant one hour
earlier** — comparing a private score to a public median there was worth 0.043 —
and the obvious next question was whether Rossmann had the same hole. It did.

## Both boards, 3,300 teams

| | best | top 10% | median | median → best |
|---|---|---|---|---|
| public | 0.08932 | 0.10294 | 0.11678 | 0.02746 (23.5%) |
| **private** | 0.10021 | 0.11774 | **0.12967** | 0.02946 (22.7%) |

Our submissions, each ranked on its own board:

| submission | public | rank | private | rank |
|---|---|---|---|---|
| **v7 blend, 3-GBDT** | 0.12932 | top 62% | **0.13813** | **top 58%** |
| v7 blend, all four (0044) | — | — | 0.14172 | top 61% |
| v7 argmax xgb | 0.13960 | top 70% | 0.14356 | top 61% |
| pipeline, no history | 0.16193 | top 78% | 0.15429 | top 71% |
| bare block, 4,144 trees | 0.14853 | top 74% | 0.15166 | top 69% |
| pipeline + history | 0.17444 | top 79% | 0.18438 | top 79% |
| bare block, 200 trees | 0.23420 | top 86% | 0.23048 | top 84% |

## What changes

**The gap to median is +0.00846, not +0.0214.** Our best submission is 0.0085
from the private median and 0.0204 from the private top 10%. The project has been
quoting a distance two and a half times too large.

That does not make the work better than it was — the rank is what it always was,
**top 58%** — but it changes what "close" means when deciding what to build. A
mechanism worth 0.01 now clears the median. Under the old reference line, nothing
short of 0.021 did, which made every incremental capability look pointless.

**What does not change:** we are below median, sixteen-plus runs in, and 0054's
capacity finding is still the largest unspent lever. Note the trap in reading the
table above optimistically: capacity *was* spent, on a bare feature block, and
that arm sits at top 69% — well behind v7's chain at 200 trees. The untested cell
is still v7's chain at a real budget, and 0056's own error was choosing the rebuild
target with the probe that 0054 showed scores features backwards.

## The finding nobody was looking for: we are more horizon-robust than the field

Joining both boards on team id:

| | field mean, private − public | ours | Spearman, public vs private |
|---|---|---|---|
| **Rossmann** | **+0.01473** | **+0.00881** (v7 blend) | 0.969 |
| **Restaurant** | **+0.04464** | **+0.03097** | 0.928 |

**On both datasets the field degrades more than we do** going from the public half
of the test block to the private half — by 40% more on Rossmann, 44% more on
Restaurant. Our rank improves on both (62%→58%, 72%→64%).

Two readings, and this record does not choose between them:

* the harness's ordered split and forward-chaining folds ask the forecasting
  question honestly (0029), so our models are less tuned to the near horizon than
  a field that could iterate against the public board;
* or a mid-table model has less to lose, and regression toward the field's mean is
  the whole effect.

The second is the boring explanation and it is not ruled out here. What would
separate them is whether the effect survives among teams at similar public scores,
which is one groupby and has not been run.

Also worth recording: **Rossmann's Spearman between boards is 0.969.** The
competition's reputation for a large shake-up is about *rank* movement at the very
top, not about the score distribution being noisy. That matters for 0045, which
treated the private board as a clean seal for the blend question — on this
evidence it was cleaner than we credited.

## The fix, and what it deliberately is not

`prep_rossmann.leaderboard(private=True)` and `leaderboard_rank(score,
private=True)` read the CSVs from disk, defaulting to private because that is what
decides the competition. `prep_restaurant` already had this shape; Rossmann did
not, which is exactly why the constant could drift from the board it described.

**The runner constants are not rewritten in this record.** They are inputs to runs
that have already happened, and changing them now would make old console logs
disagree with the scripts that produced them. New runs should call the helper.

**No conclusion is revised.** 0056's four measurements, 0058's withdrawal, 0059's
instrument and 0060's finding are all unaffected: every one of them compares
like with like, or compares our own numbers to each other. The damage was confined
to a single reference line, and its only victim was our sense of how far away the
median was.
