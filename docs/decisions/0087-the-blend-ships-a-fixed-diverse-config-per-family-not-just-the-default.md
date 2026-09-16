# 0087 — The blend ships a fixed diverse config per family, not just the default

- **Date:** 2026-09-16
- **Status:** accepted (2026-09-16)
- **Related:** [0044](0044-what-ships-is-the-average-not-the-argmax.md) (ship the
  average of the terminal fits, not the argmax), [0080](0080-a-blend-weight-a-member-cannot-inflate-decided-before-the-seal.md)
  (weight members by inverse cross-fitting fold-spread), [0035](0035-the-search-is-over-features.md)
  (the *agent* searches over features, never tunes models), [0022](0022-the-yardstick-must-measure-the-ranked-quantity.md)
  (validation must not be fitted or selected on), [0068](0068-capacity-reaches-the-shipped-blend-at-half-strength-and-validation-picks-the-wrong-budget.md)
  (validation picks a model budget the seal punishes)

## What happened

The terminal blend fitted exactly one configuration per family — the default
lgbm / xgb / catboost / elasticnet — and averaged them (0044/0080). A blend only
gains from members whose errors *decorrelate*, and four defaults are less diverse
than the same four families can be. The literature is clear that on tabular data,
ensembling several **hyperparameter configurations** of a family is a leading
source of that decorrelation — TabArena (2025) finds post-hoc ensembling of tuned
configs is what separates the top tabular models; Caruana's ensemble selection
and Wenzel's hyper-deep ensembles say the same. The question was how to get that
diversity without the part those systems rely on and this project forbids:
*tuning and selecting configurations on validation.*

## The decision

**Add one fixed, pre-declared, deliberately different configuration per boosted
family to the terminal blend, and average it in with the rest.** The extra
configs (`EXTRA_TERMINAL_CONFIGS` in `harness/tools.py`) sit at the opposite
corner of the bias/variance trade from their defaults — shallow, slow-learning,
row/column-subsampled, more regularized — so their errors decorrelate. `elasticnet`
gets no partner: it is the linear diagnostic (0035), and a second penalised linear
fit adds cost without a new error shape.

The load-bearing distinction from what 0035/0022 forbid:

- **No search.** The configs are constants in the source, not points a search
  visited. There is no continuous space explored, so there is nothing to
  overfit and nothing that fails to reproduce.
- **No selection.** Every config that fits is *averaged in* by the same 0080
  fold-spread weighting (0044); none is picked because it scored well on
  validation. A member that is fold-unstable is down-weighted, exactly as before
  — the mechanism that 0068's "validation picks the wrong budget" finding
  demands, applied to the new members for free.

So the agent still searches only over features (0035 intact); the harness simply
ships a *larger fixed member set*. A config whose fit exceeds the wall-clock
budget is skipped like any other terminal fit (0007), so a slower ensemble can
never turn a finishing run into a stalled one.

## What was rejected

- **Tuning hyperparameters on validation and selecting the best.** The direct
  0022/0068 failure — it fits the validation sample rather than finding structure
  that transfers, and it is unbounded and non-reproducible (0035). The whole point
  is to get the diversity benefit *without* this.
- **A second elasticnet.** No new error shape for real cost; the linear model is
  already the diagnostic, not a diversity source.
- **Many configs per family (a Caruana-style library).** Cost grows with each
  member for a shrinking decorrelation return; one well-chosen opposite corner is
  the cheap majority of the benefit. If the A/B shows room, more can be added the
  same way — as fixed configs, never as a search.

## What it cost

Roughly doubles the boosted-family fit time at finalize (three extra fits). The
time budget already bounds that, and a skipped config just leaves the blend the
size it was. Whether the added diversity actually improves the sealed score is an
empirical question, measured by an A/B (current four-default blend vs this
config-diverse blend, same table, same folds, same seed) on House Prices and
Rossmann.
