# 0059 — The interval belongs to the difference, and on ordered data it comes from blocks

- **Date:** 2026-08-15
- **Status:** accepted and implemented. `src/tabular_agent/compare.py`,
  wired into `probe_table` / `_probe_and_compare`, `tests/test_compare.py` (15)
  and `tests/test_compare_harness.py` (8).
- **Implements:** the instrument
  [0036](0036-the-pairing-asymmetry-was-not-there.md) proposed and left unbuilt,
  and the ordered-block diagnostic
  [0058](0058-the-ranking-failure-was-one-feature-not-the-block.md) put third on
  its list.
- **Related:** [0028](0028-the-yardstick-is-fixed-the-bar-is-not.md) (whose open
  `min_gain_ratio` this supplies a candidate for and **does not close**),
  [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md),
  [0034](0034-the-tiebreak-was-tested-and-rejected.md) (why no threshold is
  fitted here), [0043](0043-on-ordered-data-the-seal-costs-the-freshest-rows.md),
  [0049](0049-the-last-observation-is-a-sample-of-size-one.md) /
  [0050](0050-the-anchor-is-what-matters-not-the-history.md) (the feature this
  was tested against).

## The problem

Every decision a run makes is about a **difference** — is this table better than
the one it came from — and until now the only thing the harness could put an
interval around was a **score**. `val_se` bootstraps the validation rows behind
one number. Nothing bootstrapped the gap between two.

Those are different quantities and the gap is large, because

    Var(A - B) = Var(A) + Var(B) - 2*Cov(A, B)

and two candidates scored on the same rows miss the same rows for mostly the same
reasons, so the covariance term dominates. 0036 measured a difference at **3-5x**
better determined than either score behind it, and then said, in as many words,
that this "should be built and tested" as a candidate answer to 0028's open
question. It was not built. Since then this project has recorded five negative
findings about history features and two about windows, **every one of them a
difference with no interval around it**, and 0052, 0053 and 0058 are a trail of
mechanisms proposed on such differences and later withdrawn.

## The decision

Two instruments, chosen by the split, in one module.

**A random split gets a paired bootstrap over rows.** Resample the validation rows
with replacement, rescore *both* candidates on the same resample, take the
difference; its spread is the standard error of the difference. The pairing is the
whole mechanism, and it is why the harness now stores each table's probe
predictions on the validation half rather than only its score — a paired
comparison needs both prediction vectors, and the alternative is refitting the
parent every time a child is scored.

**An ordered split gets consecutive blocks instead.** Under `split_on` the
validation half is one later *period*, deliberately unlike the training rows.
Resampling rows inside it measures wobble within that period and says nothing
about which period it is — the reason `val_se` is refused outright there, and a
too-small yardstick is worse than none. So the difference is recomputed on each of
several consecutive blocks and what is reported is how it varies across them.

**The mode is selected by passing the ordering column, not a boolean.** A flag
would let a caller declare "ordered" without supplying the order, which is the one
combination that cannot be served.

### The bug this design was written around

The validation half is **not stored in its ordering column's order**.
`RunSession` selects the validation positions in time order and then writes them
with `np.sort(val_pos)`, which sorts the *positions* — so the file comes back in
the input file's original row order. Rossmann's raw file is in **reverse**
chronological order (`ship_rossmann_median.py` records this, having been bitten by
it). Blocking on file order would have cut five arbitrary slices, reported a
reassuring spread and meant nothing, on the dataset this was built for. The
ordering column is therefore a required argument and the sort happens inside the
instrument, where a caller cannot skip it.

## The measurement, and a prediction that half failed

`scripts/probe_difference_interval.py`, run against the one case whose answer is
known from outside. `_runs/ship_rossmann_median` ends with `t9`, a `fit_history`
table on `t8`; 0056 scored that arm on Kaggle at **0.18438** private against
**0.15429** without it — a real loss of **0.0301** — while our validation block
ranked it best of four.

Registered before running: (1) the row bootstrap would report a small error around
a positive gain; (2) the blocks would disagree.

| instrument | reading |
|---|---|
| pooled gain, t9 over t8 | **+0.02028** |
| paired bootstrap over rows | se **0.00074** → gain/se **27.3** |
| pairing ratio | **0.371** |
| blocks over the 44-day val window | sd **0.01214**, `sign_stable` **True**, 5/5 |
| blocks over val + holdout, 90 days | sd **0.01670**, `sign_stable` **False**, 8/10 |
| Kaggle, Aug-Sep | **−0.0301** |

**Prediction 1 passed, and brought a bonus.** The pairing ratio of 0.371 lands
exactly in 0036's measured 0.18-0.37 — reproduced on a second dataset, a different
metric, and 50,175 validation rows instead of 292. 0036 named that replication as
the thing it could not establish. It is established.

**Prediction 2 failed on the window the harness actually uses.** Over the
validation half all five blocks favoured the history table. A run would have
reported a stable positive difference for a feature that went on to lose 0.03
externally. What the blocks *did* do is widen the interval **16-fold**: `gain/se`
falls from 27.3 to 1.7. The row bootstrap called this twenty-seven standard errors
from zero. It is not, and nothing about the difference was ever wrong except the
yardstick measuring it.

**Prediction 2 passed once the window was long enough.** Extended over validation
plus the sealed holdout — 90 days instead of 44 — the blocks disagree, the pooled
gain nearly halves, and the decay is legible directly: the first five blocks
average **+0.0200**, the last five **+0.0022**, a ninefold fall with distance from
the 2015-05-01 freeze. Kaggle's test window begins two weeks after the last block
and reads −0.0301, the same line continued.

## What this establishes, and what it does not

**Establishes.** Blocking is a large and cheap improvement over a row bootstrap on
ordered data, and the row bootstrap on ordered data is actively misleading — 27
standard errors of confidence in a difference that reverses. 0036's pairing effect
replicates outside house-prices.

**Does not establish.** That the block instrument would have caught this feature
inside a run. **It would not have**, on this run's own validation half, and that is
the honest headline. The limitation is structural and was stated in `compare.py`
before it was measured: blocking samples the periods it has, and if an effect
changes sign beyond the far edge of the evaluation window every block can agree and
the effect can still fail on the rows that matter.

So the finding is about **window length, not block count.** More blocks over the
same 44 days would not have helped; reaching further out would. That is a fact
about the exam and it belongs to 0043's trade-off, not to this instrument.

## What was rejected

**Changing `min_gain_ratio`, or gating the stop rule on the new number.** The
enforced bar stays at 0.25 of `probe_se`. 0028 left the magnitude open, 0036
proposed this instrument as a candidate, and the measurement above is *one*
comparison on *one* dataset — picking a multiple of the block sd from it is fitting
a constant to a single observation, which is 0034's refusal verbatim. The interval
is reported; nothing is gated on it yet.

**Making the block count adaptive, or tuning it.** `DEFAULT_BLOCKS = 5` is a
compromise with no data behind it and says so in its own comment. It is an
argument of the function.

**Reporting one number for both modes.** `probe_vs_source_se` deliberately carries
a paired standard error on one split type and a between-block standard deviation on
the other, with `probe_vs_source_mode` beside it. They are not the same quantity
and are not reconcilable; naming the mode is honest, averaging them would not be.

**Blocking the holdout inside a run.** The 90-day reading above used the sealed
rows, which is legitimate *here* because the seal is already spent for this
question (0045) and nothing was selected on it. Wiring that into a run would let
the search see the seal, which is 0023's whole point.

## What it costs

- **One prediction vector per table on disk**, `float64` over the validation half.
  400KB on Rossmann's 50,175 rows, bounded by `max_tables`. Written after the
  score, so a table always has its number even if the sidecar fails.
- **A four-tuple.** `_score_on_held_out` now returns predictions alongside the
  score, and six call sites unpack one more value.
- **Nothing on old runs.** Tables probed before this existed have no sidecar; the
  interval is simply absent, and an absent key reads as "not measured" where a null
  would invite "measured, and zero".
