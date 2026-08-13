# Handoff

> Session log for working across two machines. Newest entry on top.
> Update before ending a session, then commit & push. Pull and read this first when starting.
>
> **This file is pruned (~2 weeks) and is not the record of *why* anything was
> decided.** That lives in `docs/decisions/` — see its README for the index and
> the project vocabulary. When an entry here records a decision, write the
> decision record too, or the reasoning is deleted with the entry.

## 2026-08-13 (later) — The sealed split is in, and it immediately caught us shipping the worst model

**Machine:** MacBook Air | **Branch:** master | **353 tests pass** (was 329).

0023 implemented (**0041**), Rossmann v7 run against it, and the run produced
**0042** — a finding that should change the terminal phase and deliberately has
not yet.

### 0041 — the third split, built

`RunSession(holdout_fraction=...)`, **opt-in and off by default**. Four calls 0023
left open, all argued in the record:

- **Opt-in**, because turning it on takes rows out of the work half and makes the
  first sealed run incomparable to every run before it. That belongs in a driver,
  not in a library default.
- **A fraction of the whole input**, not a slice of the val half — otherwise
  `val_se` grows ~1.4x on the number that ranks every table. Under a random split
  the validation rows come out **byte-identical** with the seal on or off; only
  work shrinks.
- **Ordered splits cut `work | val | holdout` in time order**, so the seal is the
  latest period. Price: val moves earlier, and the gap mixes selection with drift.
- **Every terminal candidate is scored, not just the winner** — the fits already
  exist, and it is 0039's selected-vs-unselected pair without a Kaggle submission.

The seal is held by ordering (`_score_on_holdout` takes the winner as an
*argument*), by `journal.set_holdout` being **write-once**, and by
`resolve_table`'s existing allowlist. `tests/test_holdout.py` is 17 tests.
`notebooks/phase4_sealed_holdout.ipynb` ships unexecuted, all 12 code cells
smoke-tested in order. **0023's "`build_report` already takes a `holdout_score`
parameter" was stale** — it did not; it does now.

### Rossmann v7, and the finding: **val picked the worst GBDT**

| model | val | **sealed** | optimism | |
|---|---|---|---|---|
| **xgb** | **0.12980** | **0.15253** | **+0.02273** | shipped |
| lgbm | 0.13202 | 0.14145 | +0.00943 | |
| catboost | 0.14329 | 0.14113 | −0.00217 | |
| elasticnet | 0.20556 | 0.19493 | −0.01063 | |

Seguro v2's e8/e11 failure again — except v2 needed Kaggle to find out and **v7
found it itself**. 15 tables, 5 terminal fits, 40 min, agent finalized on its own.
`submission.csv` written, unscored.

**Not noise: val is 50,175 rows.** Under `split_on` it is *one period*, so ranking
by it is n=1 in the unit that varies. `_score_on_val`'s own comment says exactly
this — it is why `val_se` is `None` on ordered splits — and the argmax then ranks
by that number with no yardstick at all, one function away.

**Registered claims: 1 passed, 2 failed (+0.0227, our worst optimism ever), 3
passed with the confound finally broken** (probe is lgbm, selected was xgb, and
optimism still ordered by selection), **4 passed** (`Customers` guarded, bundle
predicts).

**Drift is bounded and smaller than 0041 feared:** all four are scored on the same
sealed rows, so drift is a common shift — and two of four gaps are *negative*. The
spread across the ballot is selection, not the calendar.

### 0042 — and why nothing was implemented

Equal-weight blends of the same four fits, on the same sealed rows:

    xgb (shipped)              0.15253
    catboost (best single)     0.14113
    lgbm + catboost            0.13432
    lgbm + xgb + catboost      0.13610

**Every blend beat every single model**; the 3-family average is 0.0164 better
than what shipped, ~7x what the feature search bought from t8 on. An argmax over
k costs ~`sqrt(2*ln k)` SEs by construction; an average costs none.

0035 rejected blending on two measurements that **do not apply to this blend** —
both were OOF-*weighted* and scored on **val**. Equal weights against sealed rows
had never been tried.

**Implementing this from these numbers would fit the harness to v7's holdout**,
which is 0023's own failure one level up. So 0042 registers it for the next run
instead. Also rejected, with reasons: agent-chooses (would choose on the broken
signal, and reopens what 0035 closed), cv-always (house v9 refutes it), cv-above-
a-row-threshold (a constant fitted to four points), probe-always.

Replay of `val` vs `cv` across every run with a known truth is in 0042. Neither
wins outright and **the line is not split type** — seguro v2 is a random split
where cv was right.

### 0043 — a cost of the seal that 0041 missed

On **ordered** data the sealed slice is not a random tenth, it is the **most
recent** tenth, and on a forecasting task recency is most of what a row is worth.

    work (all the shipped model ever saw)   2013-01-01 -> 2015-05-01
    validation                              2015-05-02 -> 2015-06-15
    sealed                                  2015-06-16 -> 2015-07-31
    Kaggle's test set                       2015-08-01 -> 2015-09-17

v7's model forecasts **3-5 months past the end of its own data** where the field
forecasts six weeks. So **v5's Kaggle 0.135 is not a fair control for v7's
submission** — different forecast horizons, not two harnesses. Rejected as fixes:
shrinking the seal on ordered data (makes the honesty number noisy exactly where
0042 showed it was needed) and refitting the winner on work+val before packaging
(the sealed score would then describe a different model than the one that ships).

### Blend submissions are written and unscored

`scripts/blend_submission.py` packages equal-weight blends of a finished run's
terminal fits. Five CSVs in `_runs/rossmann_v7/`, 41,088 rows each, no nulls, no
negatives, alongside the run's own `submission.csv` (the argmax winner).

**Kaggle is a seal we have not burned.** 0042's claim 1 can be tested on it now,
on rows nothing here has touched, rather than waiting for a fresh sealed run.
Implied ranks *if Kaggle agreed with the sealed numbers* (it will not — see 0043):
shipped xgb ~2497/3300, 3-GBDT blend ~2153, lgbm+catboost ~2133.

### Gotchas

- **A hung request has no timeout anywhere in the library.** v7 attempt 1 sat
  **105 minutes on one gateway POST** — socket ESTABLISHED, 0% CPU, no bytes, 6
  tables on disk. Drivers retry provider *errors*; a hang is not an error.
  Patched only in `run_rossmann_v7_sealed.py` (`MODEL_SETTINGS["timeout"]`, and
  `drive()` now classifies timeouts as provider failures so they retry). **Every
  other driver is still exposed and the library is unguarded.**
- **`scripts/run_rossmann.py` has been broken since 0035** — it passes
  `StopPolicy(max_experiments=...)`, renamed to `max_tables`. Dies on
  construction. Not fixed; v7 is a fresh driver built from `run_seguro_v4.py`.
- **The agent reads the sealed number.** `finalize_run` returns `holdout_score`
  and `optimism`, and v7's agent reasoned about them in its summary. Harmless
  today because the run is over — but that is call ordering, not a structural
  bar. Named as open in 0042.
- **The work table is unservable on Rossmann** (`Customers`), so `work_probe` has
  no score. My summary code formatted it with `:.5f` and crashed *after* a
  successful run. Fixed; nothing was lost.
- v7's val is **not comparable to v5/v6's** — with a seal, val is the 45 days
  *before* the last 45, not the last 45.
- **No quiz done** for 0041/0042 yet. One question was asked and is unanswered.

### Next

- **The blend A/B**, pre-registered in 0042, on a fresh sealed run. This is the
  highest-value open item and it needs a seal that has not been consulted.
- Score `_runs/rossmann_v7/submission.csv` on Kaggle — gives the sealed number an
  external check, and v5 (0.135) is the comparison.
- Lift the request timeout out of the driver into the library, with a record.

## 2026-08-13 — 44% of the search budget went to a set we could have enumerated

**Machine:** MacBook Air | **Branch:** master | **329 tests pass** (was 309).

Three records — **0034**, **0035** (accepted, not yet implemented) and **0036** —
plus `scripts/rebuild_submission.py` and `src/tabular_agent/probe.py`.

### 0036 — and a claim of 0035's that did not survive contact

0035 argued model-vs-model is the noisiest comparison we make, because different
estimators have less correlated errors. Paired bootstrap on v8 (292 val rows,
B=400) says pairing is worth **3-5x** and is **not asymmetric**: tables 0.18-0.30,
models 0.29-0.37. What drives the covariance is how similar the two *predictors*
are — and our own GBDT residual correlation of 0.86-0.88 already said so.

0035 stands as written (never edit an accepted record); **0036 corrects it**. Its
decision is unaffected: the waste of spending 44% of a budget on four enumerable
options never depended on the pairing argument.

**`probe.py` is in.** One fixed estimator (lgbm, library defaults, seed a module
constant not `TaskSpec.seed`), justified by coverage rather than pairing: scoring
v8's tables with one ruler gave numbers for **16 tables where the run had 8**.
**One fit per table, not k+1** — the val score comes from a model fitted on all
the table's rows, so folds never enter the ranking number. That is what makes
automatic scoring affordable. Target scale now derives from the metric
(`rmsle`/`rmspe` → `log1p`, everything else `none`, caller overrides).

Loose finding, in 0036, not acted on: v8's `val_se` is 0.01116 but the paired sd
of a *difference* is 0.0024-0.0041. Comparing a table to its parent by paired
bootstrap is ~3x tighter than two point scores against a pooled `val_se`, with no
fitted constant — a candidate for **0028's open `min_gain_ratio`**.

### 0034 — the cv tiebreak: tested, rejected

`scripts/rebuild_submission.py` is committed (`ed4b486`). It packages any
experiment from a finished run, verified by replaying against the run's own val
half and insisting on the journalled score. It exists because seguro_v2 shipped
e8 (Kaggle **0.26522**) over e11 (**0.27724**) on a `val_score` margin of
0.00020 = **0.05 `val_se`**.

Breaking such ties with `cv_score` scored **1 win, 1 loss, 1 null** (seguro_v2
+0.0120, seguro_v1 −0.0031, house_v8 +0.00004). `cv` *is* the better estimator —
error spread 0.0043 vs val's 0.0124, rank correlation **+0.8 vs −0.4** — and is
still useless at tie margins, because at a tie the quantity being resolved is
~zero. **No margin threshold**: that is fitting a constant to three points.

Measured optimism now spans 5 readings on 2 datasets, one direction, growing
with experiment count: house 17exp +0.00206 → 24exp +0.00944; seguro 5exp
+0.0129 → 11exp +0.0183. **The conclusion is 0023.**

### 0035 — feature-only search

Counted across all 16 runs: **222 experiments, 97 (44%) varied only the model or
its params**, and the winner was a model variation in **10 of 16**. Winner-over-
runner-up margins where `val_se` exists: **0.03 / 0.05 / 0.18 `val_se`**.

Model-vs-model on one table is also the *least* reliable comparison we make —
different estimators, uncorrelated errors, so the covariance term that makes
paired comparisons cheap is absent. We were spending half the budget on the
noisy axis and taking the winner from it.

Decision: **search where the space is unbounded, enumerate where it is small.**
Agent's action space becomes tables only; a fixed **probe model** (fixed params,
folds, seed) ranks them so every comparison is paired; terminal phase fits
xgb/catboost/elasticnet on the winner and ships the best; **target scale derived
from the metric, overridable in `TaskSpec`** — which supersedes 0021 on placement
only (its menu-not-code decision stands).

Also found, and it argues *for* the feature axis: the agent already recombines
abandoned branches **24 times across 10 runs**, always as "current table + one
step first seen on a dropped branch".

Cross-check: `wandb_fc/tabular-nomad` does exactly this — LightGBM screening,
minimal tuning, selection by OOF only, 677 columns. Their held-out labels are
**not on the machine the agent runs on**, which is 0023 enforced physically
rather than by an allowlist. Worth stealing.

### 0035 is implemented, and 0037 covers what it nearly broke

**The agent's menu is now:** `inspect_data`, the four table tools, `explain_table`,
`explore_table`, `explore_journal`, `finalize_run`. Nothing that picks a model.
`finalize_run` fits xgb/catboost/elasticnet on the best table by `probe_score`
and packages the winner. Stop rule counts **tables**, patience 6, `table_patience`
gone with the move it existed to refuse.

**0037** — removing the model tools silently removed `explore_experiment` (it
needs an `exp_id` that no longer exists mid-run), which is the only arc turning a
score back into a hypothesis. `explore_table` replaces it and refits across folds
**on request**. The asymmetry is the record's point: ranking needs one fit (folds
never enter a val score), residuals need `n_splits`. The shortcut that looks free
— reuse the ranking fit's residuals — is wrong because they are in-sample:
measured in the notebook, mean |error| **1711 in-sample vs 6172 out-of-fold**.
And `explore_table` reads the **work half only**; val-half residuals would let an
agent fit features to the ranking set by hand.

**The no-op loophole closed itself.** Verified end to end: a `df.copy()`-style
table scores a gain of exactly **0.0** and counts against patience. No check
needed — with one counter there is nothing to reset.

`notebooks/phase4_probe_and_feature_search.ipynb` ships unexecuted, every cell
smoke-tested in order. Writing it caught a bug no test would have: four tool
results and a prompt line still said "pass this table to `train_tabular_model`".

### v9 ran, and both pre-registered claims passed

| | val | Kaggle | optimism | rank /3765 |
|---|---|---|---|---|
| v8 catboost | 0.11171 | 0.12298 | +0.01127 | 756 (top 20%) |
| **v9 elasticnet** | 0.11579 | **0.12037** | **+0.00458** | **336 (top 9%)** |
| v9 runner-up catboost | 0.12078 | 0.12553 | +0.00475 | 1149 (top 31%) |

**v9's val was worse and its Kaggle was better** — which is what less
self-deception looks like. 10 tables, all scored, **2.3 min and 17 tool calls**
against v8's 14 experiments. Full record: **0038**.

**A prediction of mine failed and it is the useful part.** Mid-run I read
elasticnet's val-cv gap (0.11579 vs 0.13015) as a winner's-curse signature and
called catboost the safer pick. Catboost scored **0.12553** — worse by 0.005.
The gap was cv's *structural pessimism*: fold models train on 80% of 1168 rows,
and a linear model at ~930 rows is on a steep learning curve. Val ranked the two
correctly; **cv ranked them backwards** — a fourth data point against preferring
cv, and the first at a margin that was not a tie.

**Ablation: the features did it.** v9's terminal set on v8's own winning table
t11 — catboost val 0.11207, elasticnet 0.11217, xgb 0.13131. Elasticnet on v8's
features scored Kaggle **0.12934**; the same family on v9's features scored
**0.12037**. Nine thousandths apart, four times the gap between the two runs'
headlines. So the gain is the **feature search**, not the terminal enumeration.
`scripts/ablate_terminal_set.py` reproduces it.

**And the finding nobody was looking for: optimism belongs to the table.** Four
models with both numbers, two per table:

    v8 t11  catboost   +0.01127      v9 t10  elasticnet +0.00458
    v8 t11  elasticnet +0.01717      v9 t10  catboost   +0.00475

They cluster **by table, not by model** — the two pairs do not overlap. Swapping
the model family moves optimism by <0.006; swapping the table moves it 3x. The
winner's curse attaches to the *feature set* a search selected, and is handed on
to whatever is fitted afterwards. Prediction for 0023 to test: optimism should
track how hard the table search worked and barely track the model.

### Seguro v3 — two defects, and 0038's hypothesis dead

Full record: **0039**. Kaggle private gini:

    v3 as shipped   (elasticnet)  0.24907   <- FAILED claim 1
    v3 with the fix (lgbm)        0.27470   <- passes, +0.0095 over v2
    v2 shipped e8                 0.26522
    v2 discarded e11              0.27724   <- still 0.0025 better than v3

**Defect 1: the ruler was not on the terminal ballot.** TERMINAL_MODEL_TYPES was
(xgb, catboost, elasticnet); the probe is lgbm. The run ranked tables with lgbm,
picked t3, then shipped the best of a set excluding lgbm. **Cost 0.02563 gini** —
twice the tiebreak effect in 0034. Fixed by putting PROBE_MODEL_TYPE in the set,
which makes "never ship worse than the number you selected by" hold by
construction.

Not evidence lgbm is better: at comparable capacity the families are close (xgb
0.27409, catboost 0.28351, lgbm 0.29066). The terminal phase enumerates *library
defaults*, so it measures the libraries. A fair comparison needs a common budget
or early stopping — **left open, deliberately**, rather than copying a learning
rate off one table.

**Defect 2: `_refuse_if_unservable` sits behind `if enforce_stop:`** and the
terminal phase passes `enforce_stop=False`, so 0030's guarantee was bypassed —
and 0035 had deleted the only place the agent heard about it. Latent since 0035
with a green suite; surfaced only because adding lgbm changed which model won.
The check moved into the probe; an unservable table is left **unscored** so it
cannot win `best_table`.

**0038's "optimism belongs to the table" is rejected.** Same table, same rows:
lgbm +0.01596 vs elasticnet +0.00482 — 3.3x, against a registered 0.002.

What replaces it, fitting 3 of 4 pairs: **optimism attaches to the *selected*
measurement.** Both seguro runs show the selected model at ~3x an unselected one
(v2: e8 +0.01828 vs e11 +0.00587; v3: lgbm +0.01596 vs elasticnet +0.00482), and
house v9 fits because *neither* shipped model was the probe. House v8's t11 does
not fit and is recorded as such.

**Also:** 0033's elasticnet fix carried its first real run (3.0 min on 476k rows
vs v1's projected 510 hours). And **the probe's seed is load-bearing at scale** —
lgbm moves 0.003 auc between seeds on 476k rows, a third of everything the search
gained. The phase-4 notebook's "seed variation measures nothing" is true only at
house-prices' 1168 rows.

### Seguro v4 — the re-run, and the same mistake one grain finer

Record: **0040**. Kaggle private gini:

    v4 shipped (lgbm)    0.27670   optimism +0.00838   top 57%
    v2 shipped (e8)      0.26522   optimism +0.01828   top 72%
    v2 discarded (e11)   0.27724   optimism +0.00587   top 57%

**Claims 1, 3 and 4 passed; claim 2 failed.** v4 beat v2's shipped model unaided
(+0.0115) and lands within 0.0005 of v2's *discarded* best — the new harness now
delivers by construction what the old one only had by luck. Optimism less than
half of v2's, and with house v9 that is two datasets, two task types, two metrics
moving the same way.

**The terminal-ballot fix is worth 0.0248 gini on val:** under v3's set this run
would have shipped elasticnet at 0.26026; it shipped lgbm at 0.28508.

**Claim 2 failed by 0.00279 — and 0039's "holds by construction" was wrong.**
`train_tabular_model` fitted with the run's seed (42), the probe with its own
(0), and lgbm moves 0.003 auc between seeds here. A near-copy of the ruler does
not carry the guarantee. Fixed: `_terminal_fits` now passes `PROBE_PARAMS` and
`PROBE_SEED`, `train_tabular_model` takes an explicit model seed (folds still
come from `spec.seed` — the exam is 0029's and must not move), and the test
asserts **equality** with the probe's score rather than membership in a list.

Same error twice in two days at two grains: first the family missing from the
ballot, then the configuration. **An invariant about a specific measurement
cannot be enforced by reproducing the category it belonged to.**

**0039's hypothesis held a second time.** On t8: lgbm (which chose it) +0.00838
vs elasticnet +0.00377 — 2.2x, predicted direction. Tally now 3 direct pairs
(3.1x, 3.3x, 2.2x), house v9 consistent, house v8 still contradicting. **The
confound is unresolved and was named before the run**: in every pair the selected
model is lgbm and the unselected is elasticnet, so "selection inflates" and "lgbm
inflates more than a linear model" predict the same thing. Needs a run where the
probe is not the selected family.

**Also:** v3 and v4 are identical setups and searched differently (8 tables vs 11,
best table +0.0087 vs +0.0060 gini). Run-to-run search variation confounds every
A/B here — another argument for 0023, which would let us run a config three times
and report a spread.

### Gotchas / owed

- **Order settled: 0035 before 0023.** Building the third split first would take
  rows out of training and make v9 incomparable to v8. Kaggle is already the
  honest number for house-prices and seguro, so the A/B needs a submission, not
  new machinery. Pre-register the comparison before submitting: v8 was val
  0.11354 → Kaggle **0.12298**.
- **Separate the lgbm/selection confound** — every selected-measurement pair so
  far is lgbm-vs-elasticnet. Needs a probe of a different family, or a run where
  the probe's family does not win.
- **Terminal fairness is open.** Enumerating library defaults measures the
  libraries. Needs a common budget or early stopping, and a record.
- 0035 collapses 0014's two-level stop, which also closes the `df.copy()`
  no-op-table loophole that reset `table_patience` — the loophole that produced
  e11, the best model this harness has made. Closing it is not a pure win.
- **0033's elasticnet fix is still untested by a real run** — seguro_v2 never
  called elasticnet.
- No notebook or quiz for this session's analysis; it produced no machinery yet.
- Analysis scripts were scratchpad-only and are not in the repo.

## 2026-08-12 — First classification run; the diagnostic model was unrunnable at scale

**Machine:** MacBook Air | **Branch:** master | **309 tests pass** (was 306).

Porto Seguro (`_runs/seguro_v1`), the project's first non-regression dataset, and
the run found a library defect worth more than its score.

### The run, and its Kaggle number

595,212 rows, 3.6% positive, metric Normalized Gini. `scripts/run_seguro.py` reads
`train.csv` **untouched** — no prep script, `id` left in, `-1` left as `-1`, `_cat`
columns left as integers. Everything an agent could do for itself was left to it.

**Metric is `auc`, not a new `gini` entry.** `gini.c` in the data folder confirms
`gini = 2*auc - 1` for binary labels, a strictly increasing function, so every
comparison the harness makes is identical under either. The driver prints gini.

| | gini | rank /5157 |
|---|---|---|
| our val (e4) | 0.28480 | top 27% |
| **public** | **0.268** | top 70% |
| **private** | **0.272** | top 66% |

Optimism **+0.0128 vs private = 1.5 `val_se`** on **5 experiments**. And a detail
for 0023: `cv_score` was 0.25998, `val_score` 0.28480 — **the truth landed between
them and nearer cv.** Random folds, so unlike Rossmann's forward chaining, val is
the flatterer here and cv was nearly honest.

### What the agent got right unprompted, in two tool calls

`t1` was one `transform_features`: `-1 -> NaN` everywhere, **and all 20
`ps_calc_*` columns dropped** — it was never told they are noise, only the naming
convention. `t2` target-encoded all 14 `_cat` columns. Three of the run's four
questions answered correctly and immediately. It did *not* drop `id`, which is in
the winning model's 52 features.

### The defect: elasticnet could not finish, and no guard noticed

Experiment 6 sat at 100% of **one** core for 63 minutes with no journal write. A
stack sample found `_sag_fast.sag64`. The agent was following the tool description,
which says elasticnet is "worth one experiment on any table you care about".

Measured: **0.23s per saga epoch** at 476k x 52, against `max_iter=20000`, so **one
fit ~77 min** and the 400-fit `LogisticRegressionCV` search **~510 hours**. Killed;
`e4` packaged by hand from the run dir; submission written and scored.

**`_LINEAR_MAX_ITER = 20000` served both branches, and the two solvers do not share
a unit.** Coordinate descent counts coordinate sweeps (cheap); saga counts full
epochs over every row. sklearn's own defaults differ 10x for exactly this reason.
The constant looked justified because a *correct* rationale sat above it — one
with a scope the constant did not have.

### The fix — see [0033](docs/decisions/0033-a-ceiling-in-the-wrong-unit-is-no-ceiling.md)

- Ceiling split by solver: `_COORDINATE_DESCENT_MAX_ITER=20000`, `_SAGA_MAX_ITER=200`.
- `_PenaltySearchThenRefit`: the **search** sees at most `PENALTY_SEARCH_MAX_ROWS
  = 20_000` rows (stratified when classifying), the **refit** sees all of them.
  Sampled inside the fold from training rows only, seeded.
- Classification grid 400 -> 45 fits, on the same solver argument (saga cannot
  warm-start along the path).

**510 hours -> 4.6 min, and it answers**: elasticnet gini 0.25111 vs lgbm 0.28480,
so trees win by ~4 `val_se` and the signal is not purely linear. It also beat
catboost's 0.24644, i.e. `e5` was the misconfigured experiment.

**Below the cap nothing changed** — verified bit-identical against the pre-change
code on house v8's t11: 0.11356705 both, to 1e-12.

**The under-fitting worry was checked, not assumed.** At the penalty the search
selects (`C=0.01`) saga converges in **118 epochs**, same gini at ceilings of 200 /
1000 / 3000. The earlier non-convergence was at `C=1.0` — weak regularisation the
search never picks. Convergence is a property of the penalty. A warning now fires
if any linear fit does stop on its ceiling, because under-fitting *this* model
biases the verdict towards flattering the trees.

### v2 on the fixed library — and the bigger finding: **`val` selection picked the wrong model**

v2 ran clean: 11 experiments, 23 tool calls, agent called `finalize_run` itself,
18.2 min (13.0 in fits). **It never called elasticnet.** Same prompt, model,
settings and seed as v1, which called it at experiment 6 — so whether the
diagnostic gets used is run-to-run variance, and 0033's fix is still untested by a
real run.

v2 also *tuned* rather than engineered: best was `e8`, lgbm on the **untouched**
58-column table with hand-tuned params. Every feature table it built scored worse.

Kaggle, all three models scored (**the two runs' val halves are byte-identical**,
so these are directly comparable):

| model | val | cv | **truth (private)** | val error | cv error | selected? |
|---|---|---|---|---|---|---|
| v1 `e4` | 0.28480 | 0.25998 | 0.27191 | +0.01289 | −0.01193 | yes |
| v2 `e8` | 0.28350 | 0.25425 | 0.26522 | **+0.01828** | −0.01097 | yes |
| **v2 `e11`** | 0.28311 | **0.26800** | **0.27724** | **+0.00587** | −0.00924 | **no** |

**`cv` ranked all three correctly; `val` put the best one last.** cv is biased
(−0.009 to −0.012, pessimistic — its fold models train on 80% of the rows) but the
bias is near-constant, so order survives. val's bias is not constant.

**And the inflation is specifically in the models that were selected.** `e8` and
`e11` differ on val by 0.0004 — a twentieth of one `val_se` — yet their val errors
differ 3x. Same run, same split, near-identical val: the only difference is which
one the argmax crowned. That is the winner's curse with the confounds stripped out.

So the harness **shipped the worse model**: `e11` beats `e8` by 0.0120 and v1's
`e4` by 0.0053, and `best` is `argmax(val_score)` with no noise term.
`submission_e11.csv` is in `_runs/seguro_v2/` (rebuilt from `experiments/e11/` +
`tables/t6.pipeline.json`; replaying it reproduces the journalled val exactly).

**Before changing the selection rule**, the proposed cv-tiebreak-within-`val_se`
was replayed over every run on disk:

- **It cannot fire on any Rossmann run** — ordered splits withhold `val_se`, so it
  is inert exactly where cv and val diverge most structurally.
- **On house-prices v8 it sweeps 8 of 14 experiments into the tie set**
  (`val_se` 0.01116 against a score of 0.11171), i.e. it degenerates into
  cv-selection on small data. Any real version wants a tighter window —
  `StopPolicy.min_gain_ratio`'s existing quarter is the convention already here.
- It changes the winner on all three runs that have a `val_se`, and only one of
  those changes has been scored.

**Do not implement it on one confirmed point.** The two cheap tests are
`seguro_v1 e3` (val 0.28324 / cv 0.26123 vs shipped `e4`) and `house_prices_v8 e14`
(val 0.11255 / cv 0.11217 vs shipped `e9`, Kaggle 0.12300) — both reconstructible
the same way `e11` was, both against a known truth, and together they span two
datasets, two task types and both metric directions.

Leaderboard context, since the absolute numbers flatter: `e11` at 0.27359 public
beats **37%** of 5157 teams and sits 0.0064 *below* the median. The field is
extremely compressed — median 0.28001, winner 0.29154 — and an untuned lgbm on raw
columns already gets ~0.28.

### Gotchas

- **A no-op table resets `table_patience`.** v2's `e11` exists because the agent
  built `t6` = `def transform_features(df): return df.copy()` and got a fresh
  tuning budget on identical features. 0014's redirect is circumventable, and the
  circumvention produced the best model of the run.
- **The wall-clock bound still cannot bound a single fold.** `train_cv` checks
  between folds; here one fold outran the whole budget, so it never fired — and
  would have discarded 510 hours when it finally did. `train_cv`'s docstring
  rejects a subprocess timeout as buying little; that premise is now refuted but
  **the guard is unchanged and the fix is unwritten** (0033, "What is left open").
- **`python script.py > log` buffers.** `_runs/seguro_v1_console.log` is empty
  because the process was killed before flushing. Use `python -u`. The journal is
  the live view; a `Monitor` on the log sees nothing until exit.
- **`params` can restore the pathology** — `{"max_iter": 20000}` still does exactly
  that, by design (0029).
- A killed run loses `transcript_N.json`: it is only written when `drive()` returns.
  The journal, experiments and bundle survive, and `finalize_run` on the same run
  dir packages the winner afterwards.

### Owed, and next

- **No notebook and no quiz for this chunk** — `_PenaltySearchThenRefit` and the
  two-units argument are untaught. Still owed from the previous entry too:
  `fit_seconds`, the time budget, `elasticnet` itself.
- **Next, in order:** score `seguro_v1 e3` and `house_prices_v8 e14` to decide
  whether the cv tiebreak is real; if it is, implement it (tighter than one
  `val_se`) with its own record. Then the single-fold timeout (0033's open
  question), then the `table_patience` no-op loophole.
- **Not done, deliberately:** no selection-rule change, no elasticnet re-test.
  Both are waiting on measurements rather than on time.

## 2026-08-12 — Thinking changed the behaviour and not the score; the ceiling is diversity, not blending

**Machine:** MacBook Air | **Branch:** master | **306 tests pass** (was 291).

Two runs (`_runs/rossmann_v6`, `_runs/house_prices_v8`), one new model type, and a
measurement that killed the plan I was about to recommend.

### Read this first

**Every behavioural finding before yesterday was measured on a model that was
never allowed to think** — see the previous entry. v6 and v8 are the first runs on
the reasoning route (`fireworks/deepseek-v4-flash-0731`, `reasoning_effort: high`).

**Result, both datasets: behaviour changed a lot, the score did not.**

| run | route | exp | best val | Kaggle |
|---|---|---|---|---|
| rossmann v5 | tensorx | 14 | 0.13760 | 0.135 |
| rossmann v6 | **fireworks+think** | 15 | **0.12741** | unscored |
| house run 3 | tensorx | 24 | 0.11354 | 0.12298 |
| house v8 | **fireworks+think** | 14 | **0.11171** | **0.123 — identical** |

**v8 is the important row.** Same Kaggle score as run 3 from a val score 0.0018
better, i.e. **optimism went UP (+0.0113 vs +0.00944) on FEWER experiments (14 vs
24)**. All the apparent val gain was self-flattery. 0023's "optimism grows with
experiment count" is not the whole story.

Thinking was **cheap**: v6 22.9k output tokens vs v5's 29.0k, and v6 used 38 tool
calls where v5 used 69. v8: 18.3k output. It does more with less context.

### What thinking did change — the tools

`fit_group_stats` went from **0 calls in v5 to the second call of v6**, on
`Customers`, the exact column it was designed for. v8 used it 3x
(`Neighborhood_*_mean`) and `filter_rows` **4x for outlier removal**
(`GrLivArea <= 4000`) — a second legitimate use, distinct from Rossmann's regime
exclusion, and the one where the target being visible to the predicate is the point.

**The dictionary channel finally paid.** `house_prices/data_description.txt` now
reaches the agent via `TaskSpec.column_notes` (79 columns, ~2.8k tokens), and v8
encoded **21 columns as `method="ordinal"`** — run 3 never used ordinal once.
`ordinal` *requires* a level order that cannot be inferred, so those orders came
from the dictionary and nowhere else. It also filled `BsmtQual` NaNs with
`"NoBsmt"` *before* encoding, which is reading the dictionary properly.

`MAX_REPORTED_COLUMNS` **60 -> 100**: with 80 columns the profile truncated and took
a quarter of the dictionary with it, including four columns carrying a level order.

**Filter measured cleanly on Rossmann, and it is a null.** v6 `e10` vs `e8`:
identical 29 features, same catboost, same log1p, only 166,265 dropped rows.
`vs_source_table -0.00336` on a `fold_spread` of 0.07. See 0032 — the 17%-of-rows
premise this was prioritised on is refuted, because `Open` is a *feature* and the
model already learns "closed -> 0".

### `fit_seconds` and a wall-clock bound

v6 lost **113 of its 132 minutes to one fit** whose parameters were *identical* to a
3.8-minute one (30 vs 32 features, same 799,654 rows). So cost is **not** a function
of what was asked for, and the blowup was environmental. Consequences:

- `fit_seconds` is now on the leaderboard row, in `meta.json`, in the training
  payload and in `explore_journal`'s `lb`. Nothing measured it before — v6's
  timings had to be reconstructed from file mtimes.
- `train_cv(time_budget_s=)`, checked between folds, raising `TrainingTooSlow` (a
  `ValueError`, so the agent gets a correction and the run survives). Off by
  default; Rossmann sets 20 min.
- **Deliberately not a cap on hyper-parameters** — that would have caught the wrong
  thing. The features-over-tuning preference is a **prompt line** instead, which is
  worth trying now precisely because all four prior advisory failures were measured
  on the non-reasoning route.

### The wandb blogpost, and a correction to my own reading

`wandb_fc/tabular-nomad` — their agent got rank 1/685 agent submissions on NOMAD.
I first read it as "generate wide, then prune". **Re-read: the post describes no
pruning at all.** It says only *"The feature matrix grew to 677 columns"*, gives no
tool-call counts, and the one use of "selection" is *blend weight* selection. So
the difference from us is **accumulation volume** (677 vs our 30), plausibly by the
same incremental mechanism.

Their setup: L40 GPU, 15-30 vCPU, minimal tuning, LightGBM+XGBoost+CatBoost blended
with non-negative OOF weights, ~60 feature iterations overnight across 4 datasets.
Three of four tasks beat the Kaggle median — so one did not.

### Blending is NOT our gap. Measured twice.

| | result |
|---|---|
| Rossmann, OOF weights over 4 models | **worse** (0.12991 vs 0.12741) |
| Rossmann, weights fitted on last fold only | 0.12792 — better, still not winning |
| house-prices, 3 GBDT families | +0.00024, i.e. nothing |

The reason is in the residuals: **GBDT-vs-GBDT correlation 0.86-0.88.** Three
boosted-tree models on the same features make the same mistakes. On Rossmann the
OOF optimiser put **0.8 on the lgbm that was worst on val**, because it had the best
`cv_score` — 0022's optimism biting the blending mechanism specifically.

**The gap is diversity, not averaging.** Which is why `elasticnet` exists now.

### `elasticnet` — one model type, and it is a diagnostic

On v8's winning table at defaults: **elasticnet 0.11217, catboost 0.11207, lgbm
0.12141.** A linear model matches a tuned catboost, so by the TabArena rule
(*"if your model can't clearly beat this, your features carry the signal"*) our
signal is in the features. **The run had no way to ask this before.**

One entry, not three: ridge 0.11420 / lasso 0.11215 / self-tuned elasticnet 0.11216
— it finds the better end unaided, the endpoints differ by 18% of one `val_se`, and
three menu entries would be three ways to spend an experiment on noise right after
we told the agent not to. **0021's "a menu entry makes it an experiment" cuts both
ways: entries should be things worth an experiment.**

Two things that were the actual work:

- **Preprocessing lives inside the estimator**, so `train_cv` refits it per fold and
  the median and scale come from training rows only. In a feature table it would
  leak across the fold boundary — batch dependence in the one place 0011's
  stability check cannot see, because it never runs on a model.
- **Infinities.** v8's own matrix had **78 non-finite values**; agents write ratio
  features constantly and something always divides by zero. GBDTs split on `inf`,
  linear models refuse. Non-finite -> NaN -> imputed.

Classification goes through `LogisticRegressionCV` with `scoring` **following the
run's metric** (`logloss`->`neg_log_loss`, `auc`->`roc_auc`/`roc_auc_ovr`,
`accuracy`->`accuracy`). My first version hardcoded `neg_log_loss`, which was the
same category of error as sklearn's accuracy default.

### Gotchas

- **Requesty's `/models` reports `supports_reasoning: true` for the tensorx route,
  and it is wrong.** TensorX's own docs say their List Models API returns no
  per-model capability flags, so the flag describes the *model family*, not the
  route. **Only a real call tells you.**
- **`ElasticNetCV` has no `scoring` parameter** — always selects alpha by squared
  error. Under `target_transform="log1p"` that *is* `rmsle` so it matches by
  construction; under `mae` it does not. Recorded in the code, not fixed.
- **sklearn 1.9.0 is the latest stable**; the deprecations we hit point at 1.10/1.11,
  so `use_legacy_attributes=False` is opting into future behaviour early, not a
  workaround for being behind.
- **v6's 113-minute fit is a contaminated measurement** — I ran heavy analysis on the
  same laptop during part of that window. Size any timeout from a clean run.
- **`t6` in v8 was claimed but never journaled** (0031's known gotcha: a failed call
  burns an id and can leave a stray parquet).
- **A blended house submission is written and unscored**:
  `_runs/house_prices_v8/submission_blend.csv` (lasso+catboost 52/48, weights from
  OOF only, val 0.10906 vs 0.11171). Scoring it settles whether OOF-weighted
  blending transfers or is just more val-flattery.
- **Kaggle-specific vs real:** the top-1% notebook we looked at blends **other
  people's public submission CSVs** selected by their LB scores. That is unservable
  by 0030's own test — a lookup table for 1459 ids, no function from features to a
  prediction. Do not chase that number.

### Owed, and next

- **No decision record for `elasticnet` / `fit_seconds` yet.** Both deserve one; the
  elasticnet-as-diagnostic argument is the part a newcomer cannot reverse-engineer.
- **No notebook for this chunk** — `fit_seconds`, the time budget and `elasticnet`
  are untaught. Quiz was done for 0032 only.
- **Next candidates:** score the blend submission; the house-prices **ablation**
  (same v8 script, dictionary withheld) to attribute v8's behaviour change; or
  `min_gain_ratio` (0028), which now has a second dataset *and* — from Rossmann v5's
  pessimistic exam — a validated yardstick.
- **A finding worth acting on eventually:** on ordered splits the **last fold** ranks
  models far better than the mean of folds (Spearman 0.960 vs 0.864 against val on
  v6), because early forward-chaining folds train on a sixth of the history. That is
  a candidate correctness fix to what `cv_score` means under `split_on`.

## 2026-08-11 (latest+7) — Both gaps closed; the run used the wrong one, and it did not help

**Machine:** MacBook Air | **Branch:** master | **291 tests pass** (255 + 36).

**Decision record `0032`.** Run is `_runs/rossmann_v5` (gitignored) — **finished
cleanly, and it is the first Rossmann run to finalize itself and write a
`submission.csv`** (41088 rows, zero negative predictions).

### Built

**`fit_group_stats(by, value, stat)`** — per-group statistics of a *non-target*
column, fitted once into the lineage sidecar and looked up at predict. `<by>_<value>_<stat>`,
one column. **Always out-of-fold, no opt-out**: the tool cannot tell whether the
value column is a proxy for the target, and group size hides that leak rather than
removing it. The payoff is that the *value* column need not exist at predict time —
only `by` does — so a column 0030 could only refuse becomes usable as its group's
history.

**`filter_rows(code)`** — `keep_rows(df)` returns a boolean **mask**. The rows are
dropped once, when the table is written, and the lineage step carries
`train_only: True` so `Pipeline.transform` skips it. That flag is load-bearing:
`transform` has exactly two callers, `_score_on_val` (which would misalign
predictions against targets, and grade a filtered experiment on an easier exam) and
`predict_bundle` (which owes an answer for every row). Third sandbox mode
(`run_filter`); `rows.py` holds the argument.

Two exemptions follow from "evaluated once, never replayed": the filter **may read
the target** (0009's reason does not apply to code that is never re-run) and **may
be batch-dependent** (nothing for it to disagree with). It is the only agent code
here needing no stability check.

Also: `transform.py`'s stability refusal no longer says a per-group aggregate of
another column "cannot be built this way" — that sentence sent v4 into the retry
loop that killed it. It names `fit_group_stats` now.

### The result, and it refutes the premise this was prioritised on

14 experiments, 21.1 min, best `e13` **0.13760** rmspe vs baseline 0.44541.
(v3's best servable was 0.12344; Kaggle median 0.11677.)

**`filter_rows` was used unprompted, twice** — the docstring was the whole channel.
The second use is a clean A/B:

| | table | features | model | val_score |
|---|---|---|---|---|
| `e8` | `t9` | 29 | catboost/log1p | 0.15589 |
| `e10` | `t13` = `t9` + filter | **same 29** | **same** | 0.15924 |

`vs_source_table -0.00336` on a `fold_spread` of ~0.07. **Dropping the 17% of rows
`rmspe` does not score cost nothing and gained nothing** — a twentieth of the
noise. The agent read that correctly and dropped the branch.

Two earlier entries here called row filtering "the cheapest gap to close, biggest
effect here", from the 17% figure alone. **Wrong, and the reason was never argued:
`Open` is a feature.** The model already learns "closed → 0", so removing those
rows takes information away and returns nothing. The step-8 notebook reproduces
this independently on synthetic data. The capability is still right to have — on
generality, not on this dataset's 17%.

**`fit_group_stats` was never called.** Zero calls in 69 across both attempts. The
tool the agent *asked for* in v4 went untouched; the tool nobody asked for got used
twice. Not proof it is unwanted (n=1, and `Store_te` already covers much of the
ground) — but it is the fourth time changing what the agent is *told* failed to
change what it *does* (0024, 0027, 0028).

### Kaggle scored it: 0.135, and our exam was PESSIMISTIC for the first time

| | our val | Kaggle | |
|---|---|---|---|
| house-prices run 3 (random split) | 0.11354 | 0.12298 | optimistic by 0.00944 |
| **rossmann v5 (ordered split)** | **0.13760** | **0.135** | **pessimistic by 0.0026** |

Rank ~2140/3300. The absolute number is middling; **the calibration is the
finding.** Every earlier run's validation half flattered itself — 0022 exists
because 91% of a celebrated gain turned out to be optimism. Here the model did
slightly *better* on unseen data than our own exam predicted.

That is 0029 paying off and the first direct evidence for it: an ordered split plus
forward-chaining folds asks the competition's question, so there is no
interpolation premium to hand back. Practical consequence — **this dataset's
`val_score` is trustworthy in a way house-prices' never was, which is exactly the
second dataset 0028's `min_gain_ratio` question has been waiting for.**

### Shipped alongside

- `docs/decisions/0032`, indexed in the decisions README.
- `notebooks/phase3_harness_step8_rows_and_groups.ipynb` — 43 cells, 24 code, all
  smoke-tested in order bar the two live-LLM ones. **Its filter demo loses**
  (`cv` 272→113 while `val` 132→1580) and the notebook teaches that rather than
  hiding it: two numbers over two populations, exam unmoved.
- `tests/test_group_stats.py`, `tests/test_rows.py`,
  `tests/test_harness_rows_and_groups.py`.
- `scripts/run_rossmann.py` → `rossmann_v5`, plus `summarise_new_tools`.

### Gotchas

- **`cv_score` changes meaning under a filter** and `val_score` does not. The note
  says so; two numbers on one row measuring different populations will get compared
  anyway. This is exactly what confused the notebook demo at first reading.
- **Attempt 1 died on `explore_experiment` exceeding its retry budget** —
  `UnexpectedModelBehavior`, v4's failure shape in a new tool. Restart-on-error
  salvaged it. That guard is now load-bearing rather than a nicety.
- **The first `filter_rows` use was confounded**: the agent stacked four tables on
  top before training, so `vs_source_table` was null and the branch's bad score said
  nothing about the filter. It self-corrected later, but a tool whose effect is only
  visible in a controlled pair is easy to misjudge once.
- **Out-of-fold group stats inherit the NaN block** under forward chaining — same
  gap as OOF target encoding, now on a second column family.
- **The agent narrated the stop rule at itself** in its summary ("the harness
  considered these improvements as being within noise") and miscounted its own
  experiments (said 10, ran 14).
- **Ten tools now**, up from eight. 0012's smallness argument gets a little thinner
  each time.
- **Every run to date has been on a NON-reasoning route of a reasoning-capable
  model. The fix is one string.** This is the biggest thing in this entry and it
  invalidates nothing, but it re-frames everything.

  Across both v5 transcripts: **0 reasoning tokens** on 1.40M input / 29k output,
  no thinking parts — only `text`, `tool-call`, `tool-return`, `retry-prompt`.

  DeepSeek V4 Flash **does** support thinking, via
  `extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"|"max"}`,
  returning `reasoning_content` beside `content`. Sending exactly that to our route
  changes nothing. Sending it to a different provider route for **the same model**
  works:

  | Requesty route | completion tokens | `reasoning_content` |
  |---|---|---|
  | `tensorx/deepseek-v4-flash-0731` *(ours)* | 4 | **absent** |
  | `fireworks/deepseek-v4-flash-0731` | 74 | present |
  | `novita/deepseek/deepseek-v4-flash-0731` | 95 | present |

  Same gateway, same request body, same model. **It is the provider route, not the
  model and not our code.** Note the gateway's own `/models` metadata reports
  `supports_reasoning: true` for the tensorx route — **that flag is wrong, do not
  trust it.** Only an actual call tells you.

  Confirmed end-to-end through our stack: on the fireworks route pydantic-ai
  surfaces a real `thinking` part, so `harness/usage.py` and the transcript
  tooling need no changes — only `MODEL` and a `model_settings`.

  Consequence: the four failed advisory steers, the orientation tax (0024/0025/
  0027) and 0032's unused `fit_group_stats` are all results on a model that was
  never allowed to think. They may be artifacts of that, or the structural fixes
  may be needed regardless. **This is now the cheapest and highest-value open
  experiment in the project**: re-run v5's configuration on
  `fireworks/deepseek-v4-flash-0731` and compare. Budget for it — reasoning tokens
  are billed output, and v5 spent only 29k output tokens precisely because it never
  thought.

- **Why tensorx behaves this way: no documentation says. Two things are known.**
  Searched; nothing states the limitation. What was found:

  *The wrong flag has a mechanism.* TensorX's own docs say "The List Models API
  does not currently return per-model capability flags." So Requesty cannot be
  reading `supports_reasoning` from the provider — it is attached to the **model
  family** (DeepSeek V4 Flash does support thinking) rather than to the **route**.
  General lesson: on a multi-provider gateway, a capability flag describes the
  model, and only a real call describes the route.

  *TensorX is a small EU-sovereign host* (Dublin/Helsinki, OpenAI-compatible,
  $0.04/M input). Their docs do not list `deepseek-v4-flash-0731` at all and their
  stated pick for "complex reasoning tasks" is the older `deepseek-r1-0528` —
  consistent with a deployment that never enabled the V4 thinking path. Inference,
  not documentation.

  *Ruled out empirically:* it is **not** a response-parsing gap. If thinking
  happened and `reasoning_content` merely failed to be split out, the chain of
  thought would arrive inside `content`, usually in `<think>` tags. A hotel-riddle
  prompt at `reasoning_effort: max` returned 266 tokens of direct answer with no
  `<think>` anywhere. Thinking never triggers.

- **Pre-flight for the thinking re-run: PASSED, and it was worth checking.**
  DeepSeek V4 thinking + tool calls is widely broken across intermediaries
  (litellm, opencode, codex and n8n all have open issues) because `reasoning_content`
  must be echoed back on every assistant message once a tool call enters the
  conversation, and clients drop it → 400. **Our harness is exactly a multi-turn
  tool loop, so this was the real risk.** Tested: pydantic-ai 2.19.0 on
  `fireworks/deepseek-v4-flash-0731` completed a two-call tool loop with thinking
  on, surfacing `thinking` parts and round-tripping correctly. No harness change
  needed.
- **Next:** `min_gain_ratio` (0028) now has its second dataset **and** a validated
  yardstick, which is the strongest it has ever been positioned. Otherwise the
  time-windowed group aggregate (the deep one, still unbuilt).

## 2026-08-11 (latest+6) — A decorative guard, a corrupted run, and the tool the agent keeps asking for

**Machine:** MacBook Air | **Branch:** master | **255 tests pass.**

**Decision record `0031`.** Runs v3/v4 are in `_runs/` (gitignored). **v4 was
stopped by hand mid-run** at 3 experiments — nothing is in flight.

### Read this first if you are picking up the next chunk

Ulf's next step: **the `fit_*` family and/or row filtering.** Both are designed
below. Everything else here is context for why.

### 0031 — the guard that never guarded

v3 died at experiment 8 holding **val 0.12344**, the best servable model any run
has produced, on `OSError: Corrupt snappy compressed data`. The journal had four
duplicate table ids (`t3`, `t7`, `t10`, `t13`), because two `fit_encoding` calls
computed the same id and wrote the same parquet at once.

**0008 set `max_concurrency=1` to prevent exactly this. That parameter limits
concurrent agent *runs*, not tool calls.** It never did anything. It survived
because models used to emit one tool call per message — and because
`test_tool_calls_are_serialised` asserted the constructor argument had been
passed rather than the property it was named for. Green throughout.

Fixed in the primitives, where no framework setting can undo it and which also
covers two processes on one run dir: `next_table_id` claims via
`O_CREAT|O_EXCL`, `next_experiment_id` via `os.mkdir`, `save_parquet` writes
tmp → fsync → `os.replace`. **Validated on v4: 0 duplicate ids.**

### The cheapest open win: the agent is not using `log1p`

| run | target transforms |
|---|---|
| v1 | `none` x16 — never once |
| v2 | `none` x12 — never once |
| v3 | `none` x4, `log1p` x3 — **and v3 scored best** |

`rmspe` is a *relative* error metric; squared error on `log(Sales)` approximates
it and is what every serious Rossmann solution does. Training on raw `Sales`
optimises absolute error — the wrong objective. **The tool has existed since
0021 and mostly goes unused.**

Compounding it: models train on squared error over **all** rows including the 17%
closed days with `Sales == 0`, while `rmspe` scores **only** non-zero rows. Wrong
error type, and fitted on rows nobody grades.

### What the agent actually asked for, and could not have

v4's first attempt died in 2.5 minutes with zero experiments. It tried to build
**`store_open_rate` and `store_promo_rate`** — per-store aggregates, the winner's
main feature family. 0011's stability check refused them (correctly: they are
batch-dependent), three refusals exhausted `transform_features`' retry budget, and
`UnexpectedModelBehavior` killed the run.

Two weaknesses, both fixed: the refusal message pointed at `fit_encoding`, **which
cannot do this** — `fit_encoding` gives per-group statistics of the *target*, and
these are per-group statistics of another column. And the driver now restarts a
fresh agent on an `error` ending, as it already did for provider errors.

Neither touches the gap. **The agent reaches for this unprompted and there is no
tool.**

### Design notes for the next chunk

**First, the framing — now a standing rule in `CLAUDE.md` ("Datasets are probes,
not targets").** Rossmann is an instrument for finding out what the harness cannot
express. Nothing dataset-specific goes into `src/`, and every capability below has
to be justified as a gap *any* tabular problem could hit. The test: can the
docstring be written without naming the dataset?

Both candidate chunks are exactly where that discipline gets tested, so state the
general form before writing code:

| Rossmann-shaped temptation | the general capability to build |
|---|---|
| "exclude rows where the store was closed" | some rows should not be trained on |
| "per-store average of Open/Customers" | per-group aggregates of a column, fitted once and replayed |
| "handle the 17% zero-sales rows" | nothing — `rmspe` already excludes them, which is the metric's job |
| "log the target because rmspe is relative" | nothing — `target_transform` exists; the finding is that the *agent* under-uses it |

The last row matters most: **the log1p finding is a behavioural result about the
agent, not a missing feature.** Do not add a rule that picks a transform from the
metric. What it argues for is better evidence about whether advisory text ever
changes behaviour — which is 0027's open question, not a Rossmann one.

**A `fit_*` group-statistics tool.** Sketch:

    fit_group_stats(by=["Store"], value="Open", stat="mean")   -> Store_Open_mean

Mechanically it is `FrequencyEncoder` with `value_counts` swapped for a stored
`groupby` — that class is the working template, down to its unknown-category
fallback. Three steps: **fit** on the work half
(`df.groupby("Store")["Open"].mean().to_dict()` — 1115 entries, 0.703 to 1.000 on
this data, plus a global default for a store never seen); **store** the dict in
the lineage sidecar as JSON; **look up** at predict with
`df["Store"].map(mapping_).fillna(default_)`. It never recomputes, which is
exactly why it is stable where the agent's `groupby(...).transform("mean")` is
not: one row maps to the same number as a million.

**CORRECTION to the first version of this entry.** It said a per-group statistic
of a non-target column "needs no out-of-fold discipline at all". **Too strong —
do not implement from that.** It holds for `Open`, which is genuinely
target-independent. It fails for `Customers`, which correlates 0.895 with `Sales`:
a per-store mean of `Customers` is very nearly a per-store mean of the target, and
computing it without holding out the row's own contribution is target encoding
with the guard taken off. Group size rescues it *here* — ~896 rows per store, so a
row moves its own group's mean by about a thousandth — but that is a property of
this dataset, not of the tool. On a key with three rows per group it is the same
leak `oof_target_encode` exists to prevent.

Since the tool cannot know whether a value column is target-correlated, **use the
out-of-fold path always.** `fit_encoding` already implements it, the cost is one
pass per fold, and it removes a judgement call the agent would otherwise have to
get right silently.

**The consequence worth having:** this makes `Customers` *usable* instead of
merely refused. `fit_group_stats(by=["Store"], value="Customers", stat="mean")` is
each store's typical footfall — a number fixed at training time and looked up at
predict, requiring no knowledge of the test day's actual count. Legitimate, and
precisely the shape of the winner's features: historical aggregates of quantities
you cannot observe on the day you predict. The tool turns today's hard refusal
into a sanctioned path.

Only the **time-windowed** variant stays hard: "last quarter's mean sales for this
store" needs a cutoff, per-fold discipline, and a frozen version for predict (test
rows have no history). Ship the plain version first; it covers what v4 asked for.

**Row filtering.** `Open == 0` is 17% of rows, unscored by `rmspe`, and cannot be
dropped today. The design point that makes this a decision rather than a patch:
**a row filter would be the first pipeline step that is NOT replayed at predict
time.** Every existing step is replayed — and it must not be here, because
`_score_on_val` replays lineage over the validation rows and a filter would drop
some, misaligning predictions from `y_val`; and at serve time every test row needs
a prediction. So a filter is a *train-time-only* step and the lineage needs to
know the difference. Nothing in `Pipeline` expresses that yet.

Note this stays inside 0029's line: filtering *training* rows is modelling, and
the validation half is untouched, so the exam does not move.

### Gotchas

- **v3's numbers are sound but its journal is not** — a losing write took its
  lineage sidecar with it, so what was scored was self-consistent, but the journal
  records a column living in a file that does not contain it.
- **A claimed id is never released**, so ids stay unique and stop being
  contiguous, and a crash between claim and write leaves a zero-byte parquet.
- **Best servable result so far is v3's 0.12344**, against a Kaggle median of
  0.11677 — but that median is each team's best of months of submissions (winner:
  289), not one 20-minute run. Realistic ceiling with today's tools is ~0.12-0.13.
- **No run has yet produced a `submission.csv`.** v1 shipped an unservable model,
  v2 ran out of calls, v3 crashed, v4 was stopped by hand.

## 2026-08-11 (latest+5) — Rossmann ran three times; the leak is now refused, and we audited what we cannot express

**Machine:** MacBook Air | **Branch:** master | **254 tests pass.**

**Decision record `0030`.** Runs live in `_runs/rossmann_v1..v3` (gitignored).

### v1 — the run that justified everything after it

The data-dictionary channel **worked and did not matter.** At experiment 9 the
agent wrote, unprompted and in the replay-safe form:

    # Customers is measured on the prediction day and is NOT available at predict time
    if 'Customers' in df.columns:
        df = df.drop(columns=['Customers'])

then built an honest chain (`Store_te` → `Store×DayOfWeek` → `Store×Month` →
tuning) taking 0.1698 → **0.13173**. It counted for nothing:

| | val_score | servable |
|---|---|---|
| shipped `e7` (used Customers) | **0.07912** | no |
| best honest `e17` | 0.13173 | yes, verified against real test rows |
| Kaggle winner, 3300 teams | 0.08932 | |

A val_score better than first place is the signature of a column nobody had.
`best` ranks on `val_score`, so the leaked model stayed incumbent, **ten genuine
improvements were each recorded as no-improvement**, the run stopped early on
patience, and finalize bundled a model that raises `KeyError` on the first real
prediction. The agent's own summary named the problem; it had no way to act on it.

### v2 — the fix, and a different failure

`RunSession(serving_columns=...)` + refusal at `train_tabular_model` (0030).
**`experiments_using_customers: 0`.** The refusal never even fired: the agent read
`columns_not_available_at_predict_time` off `inspect_data` and put the guarded drop
in its *first* table, before training anything. Same fact as v1's prose, but
machine-generated and backed by enforcement.

**But v2 hit a flat 120-call limit and never finalized** — a good model, no bundle.
Best 0.14049 vs v1's honest 0.13173, and **that 0.009 gap is nothing**:
`fold_spread` on the winner is 0.07342, five times larger. Not measurably worse,
not measurably better. What *is* different is behaviour — v1 descended steadily,
v2 thrashed (0.261, 0.185, 0.191, 0.197, 0.151, 0.167, 0.145, **0.285**, 0.149,
0.169, 0.156, 0.140) and took 33.5 min against 10.3.

Fixed for v3: call budget is `max_experiments x 12`, and the runner finalizes when
the conversation ends without one.

### The capability audit, against Gert Jacobusse's 1st-place solution

Asked because a run is only as good as what the tools can express. **Three gaps,
in priority order:**

1. **No row filtering.** `transform_features` must preserve the row index, so the
   agent cannot drop the 17% of rows where the store is closed — rows `rmspe` does
   not even score. Cheapest gap to close, biggest effect here. Needs a decision:
   filtering *training* rows while the validation half keeps them is the honest
   arrangement, and it touches 0029's line about who owns the split.
2. **No time-windowed group aggregates** — "last quarter / half-year / year,
   segmented by day-of-week and promo" was the winner's main feature family. A
   rolling per-store mean is batch-dependent, so 0011 rejects it *correctly*: at
   predict time the rows arrive without their history. Legal only as a **fitted**
   transform, which is the `fit_*` family flagged as having "no sanctioned path"
   since July. The deep one.
3. **No blending.** `blend.py` written 2026-07-29, still unregistered. He used
   20+ XGBoost models.

Deliberately not gaps: external data (0001), and the 0.985 prediction correction —
its own author reports the effect was smaller than the CV fold standard deviation
and only appeared on the leaderboard, which is 0022's failure mode exactly.

**Realistic ceiling with today's tools is ~0.13–0.14 against a Kaggle median of
0.11677.** Worth remembering before reading any Rossmann number as a verdict on
the agent.

### Gotchas

- **A flat tool-call budget is a per-run guillotine.** v1: 44 calls / 17
  experiments. v2: 120 / 13. Same dataset, same model, same prompt.
- **`val_se` is absent on this dataset** (ordered split, 0028), so `fold_spread` is
  the only yardstick — and at 0.073 it is far too coarse to separate these runs.
- Provider 502s killed v1's first attempt outright; the runner now restarts a fresh
  agent on the same run dir, which is exactly what an episode does.

## 2026-08-11 (latest+4) — Rossmann runner written, NOT run; a third failure mode found

**Machine:** MacBook Air | **Branch:** master | **250 tests pass.**

`scripts/run_rossmann.py` exists and **has not been run** — no API spend on this
dataset yet. `prep_rossmann.py` now also writes `test_joined.parquet` (the
competition's test rows through the same join and typing), so a run can end in a
real submission.

Config: `rmspe`, `split_on="Date"`, `val_fraction=0.05` (~45-day horizon vs
Kaggle's 48), `max_experiments=25`. That is **below 0025's ~35 crossover, so it
runs as one conversation, not episodes** — the switch is in the script, since
where it should live is still open.

### The finding, from dry-running the machinery without the LLM

Dropping `Customers` has **two** failure modes, not one, and only the third
option is submittable:

| what the agent does | val_score | bundle predicts test? |
|---|---|---|
| keeps `Customers` | 0.314 (flattered) | **no** — model needs a missing column |
| `df.drop(columns=['Customers'])` | 0.372 | **no** — stored transform re-runs and raises |
| `df.drop(..., errors='ignore')` | 0.372 | yes |

**The middle one is the trap.** The agent does the right thing, the score is
honest, and the bundle still cannot run: per 0010 a transform is stored as *code*
and replayed on raw rows, which at predict time never had the column.

**Nothing in the harness can catch it.** `_score_on_val` replays lineage on
validation rows drawn from the training file, and those still have `Customers`.
There is no row anywhere in a run that is shaped like a real serving row. The
first prediction on genuinely unseen data is the first chance to find out — which
is why the runner attempts a submission and reports the two failures distinctly.

Generalises well past this dataset: **any transform referencing a column absent at
predict time breaks the bundle silently.** 0011's stability check does not look for
it (it tests batch-dependence). Worth a decision record if we act on it.

### What the run will measure

1. **Does the agent drop `Customers`?** It is numeric, so it is in the feature
   matrix *by default* — doing nothing means training on it. The prompt does not
   mention it; the only reason to act is the dictionary line saying it does not
   exist at prediction time.
2. **Does any of this work at 1M rows?** Every earlier run was 1460.
3. **Is `cv_score` worse than `val_score`?** It should be, under forward chaining.
   The other direction means something is wrong.

### Gotchas

- **~1094 of 41088 test predictions came out negative** in the dry run. The runner
  reports them and does **not** clip: flooring in a script would hide a real defect
  in the fit.
- `write_submission` finds the bundle at `run_dir/bundle` on disk, because
  `finalize_run`'s return went to the agent, not to us.

## 2026-08-11 (latest+3) — The harness owns the exam; CV forecasts when the run does

**Machine:** MacBook Air | **Branch:** master | **250 tests pass.**

**Decision record `0029`.** It answers Ulf's question — "we are doing
preprocessing which could in principle be part of the agent's work, so the final
result depends on what we do now, which is kinda not what I want" — and the
split question, with one line.

### The line

**The harness owns the evaluation protocol; the agent owns the modelling.**
Metric, `split_on`, `val_fraction`, fold strategy, column descriptions and CSV
typing are ours — an agent that can change the exam can pass it without learning
anything (0012's argument, applied to measurement). Which columns to use, and
what to build from them, is the agent's.

**So `Customers` went back into the Rossmann data.** Dropping it was the harness
doing the agent's work. It stays, with the dictionary saying it is not available
at prediction time, and whether the agent uses it is now a **measured result**: if
it trains on it, `val_score` looks excellent and we learn the dictionary channel
does not change behaviour. Worth more than a clean number.

The `store.csv` join stays in prep — there is no join tool — and is recorded as a
real confound rather than explained away.

### Time or grouping? Time, and the data says so

1115 stores in the work half, 1115 in validation, **zero validation-only stores**.
Kaggle's test set is the same stores on later dates, so a time split asks the
deployment question. Grouping on `Store` would answer "how well do I predict a
store I have never seen" — a different product. General rule: **hold out whatever
deployment will hand you that is new.**

### Three fixes to the exam, which was subtly wrong

**1. CV forecasts when the run does.** `make_folds(order=)` builds forward-chaining
folds. Random k-fold on a panel lets a model predict Wednesday having trained on
Tuesday and Thursday *of the same store* — interpolation, not forecasting.
**Both strategies kept**; the choice follows `split_on`. On Rossmann: `cv_score`
0.371 vs `val_score` 0.314, i.e. CV is now the *pessimistic* number where random
folds made it optimistic.

**2. The split cuts on a value, not a row count.** It had put 603 of
2015-01-30's stores in work and 512 of the same day in validation.

**3. A table dropping the ordering column is refused**, rather than silently
falling back to random folds mid-run.

### Gotchas / open questions

- **Not every row gets an out-of-fold prediction now** — 160,559 of 965,919 on
  Rossmann. NaN in `oof.parquet`, `fold = -1` in `explore_experiment`'s frame.
  `explore_experiment`'s "positionally aligned" promise is now "aligned, some
  missing".
- **Out-of-fold target encoding leaves that same block NaN**, so ~1/(n_splits+1)
  of training rows have no encoded value, and the missingness is time-correlated
  — a tree can learn it as "early period". Not fixed: every alternative computes
  the value from the row's own future or from itself.
- **`fold_spread` now mixes noise with "how much does more history help"**, since
  fold training sets are unequal. Matters because 0028 withholds `val_se` on
  ordered splits, leaving `fold_spread` as the only yardstick.
- **`val_fraction` is approximate under an ordered split.** At 0.05 on Rossmann it
  gives a 45-day horizon against Kaggle's 48 — close enough to compare against the
  public leaderboard in the folder (3300 teams: best 0.08932, top 10% 0.10294,
  median 0.11677).
- **Still no runner for Rossmann** and no live run attempted. A hand-driven lgbm
  with 100 trees scores 0.314 rmspe in ~5s, so the dataset is affordable.
- **Model note:** runs so far use `tensorx/deepseek-v4-flash-0731` to keep costs
  down, deliberately not a frontier model. Ulf expects better models to help in
  production; every behavioural finding in 0024/0025/0027 is on this one model.

## 2026-08-11 (latest+2) — Rossmann: the data dictionary now has a channel

**Machine:** MacBook Air | **Branch:** master | **236 tests pass.**

**No decision record yet** — the channel is small and 0018-shaped, but if
Rossmann becomes a real second dataset the `Customers` finding below deserves
one. Flagging rather than deciding.

### The finding, which matters more than the feature

**`Customers` in Rossmann is a leak this harness cannot see.** It correlates
**0.895** with `Sales` and **does not exist in `test.csv`** — you cannot know
today's customer count when forecasting tomorrow's sales. Our validation half is
drawn from `train.csv`, so the column is present there too: the agent trains on
it, `val_score` improves, and **the leaderboard confirms the leak**. No fold
boundary crossed, no encoder misfitted, every guard holding.

This is the `Id` failure from run 1 (0019's gotcha) with a much bigger number.
Nothing computable from the training file reveals it — the only evidence is the
competition's data dictionary plus a test-set column list the harness never sees.

### Done this session

**1. The data dictionary channel.** `TaskSpec` gains `column_notes={col: text}`
beside the `notes` field that has existed and reached nobody since phase 1.
Per-column descriptions ride *on their column* inside `inspect_data`'s profile —
next to dtype and cardinality, where the reader already is — and run-level
`notes` comes back as `about_this_data`. A column nobody described carries no
`description` key, so an undocumented dataset costs exactly nothing.

**2. One line in the brief.** Since the prompt now tells a resumed episode to skip
`inspect_data`, a fresh agent would never see the dictionary. The brief says it
exists and where; it does not repeat it (O(columns) text in an O(1) brief, 0026).

**3. `scripts/prep_rossmann.py`** — joins `store.csv` on `Store` (1017209 x 17),
drops `Customers` with the reason recorded, settles `StateHoliday`'s mixed
`"0"`/`0` typing, writes `train_joined.parquet`. Carries the full 17-column
dictionary and an `ABOUT` blurb. **`rossmann_store_sales/` is now gitignored** —
Kaggle data, same as `house_prices/`.

### Gotchas / open questions — read before running this dataset

- **The metric is wrong.** Kaggle scores RMSPE; `metrics.py` has no such metric,
  so a run defaults to `rmsle`. Not implemented — it was your call to leave it.
- **RMSPE ignores zero-sales rows; we do not.** 17% of rows, every one a closed
  store (`Open == 0`). Dropping them changes what the model is *for*, so it is a
  decision, not a cleanup.
- **`split_on="Date"` is mandatory here** and has never been exercised before.
  Verified working: `mode: ordered`, 813,767 work / 203,442 validation.
- **`val_se` will understate uncertainty on this dataset** — the bootstrap assumes
  i.i.d. validation rows and an ordered split is exactly not that (0028's caveat,
  now with a dataset that triggers it).
- **1M rows.** Every experiment costs far more than house-prices' 1460, and no
  run has been attempted at this size.
- **This is the second dataset 0028 needs** for the `min_gain_ratio` question, and
  a good one: it has a genuine forward-in-time test set, so Kaggle can supply
  ground truth the way it did for house-prices.
- **`Date` is dropped before training** as non-numeric — date parts have to be
  derived with `transform_features`. The column note says so.

## 2026-08-11 (latest+1) — `val_se` shipped; the bar it gates is deliberately still open

**Machine:** MacBook Air | **Branch:** master | **233 tests pass.**

**Decision record `0028`. It corrects one claim in `0022` — read it before
assuming "0022 is implemented" means the run-3 failure cannot recur. It can.**

### Done this session

**1. `val_se` — 0022's decision, built.** `_score_on_val` bootstraps the
validation score over its own rows (200 resamples, seeded from the run seed, so
control flow stays deterministic). It lands on the leaderboard row, `meta.json`,
the training payload, `explore_journal`'s `lb` frame, and the report's headline
(`0.138 (+/- 0.012)`). `stop._noise` prefers it over `fold_spread`, which stays as
the fallback for old rows and un-resamplable validation halves.

Verified against 0022's independent bootstrap (~0.0098): we measure **0.0122** on
house-prices lgbm. `val_se / fold_spread` runs **1.12–1.40x** on this dataset.

**2. The correction, which is the important part.** 0022 claims the fix "would have
told run 3 at e4 that its 0.008 gain sat inside the noise". **It would not:**

    old:  0.25 x fold_spread (0.01008) = 0.00252  ->  gain 0.00808 PASSES
    new:  0.25 x val_se      (0.01223) = 0.00306  ->  gain 0.00808 PASSES

The enforced bar is a **quarter** of the yardstick. Rejecting that gain needs
`min_gain_ratio > 0.66`. 0022 fixed the *category* error (right quantity) and left
the *magnitude* error (factor of four) untouched — and its own opening sentence,
"the bar is four times smaller than the noise", is the accurate statement.

**3. An advisory, shipped; the bar, not changed.** `note` now says when a gain is
inside one standard error — refuses nothing, gates nothing, silent unless the gain
is in the band between zero and 1 SE. **`min_gain_ratio` stays at 0.25 pending a
second dataset, which Ulf is supplying.**

### Gotchas / open questions

- **`min_gain_ratio` is the open question, and it needs the second dataset.** What
  to measure when it lands: (a) does `val_se / fold_spread` diverge more than
  house-prices' 1.1–1.4x — that is how much the category fix was worth; (b) what
  ratio separates gains that survive a held-out check from ones that do not.
- **A stricter bar costs less than 0022 implied.** `min_gain_ratio` does not refuse
  experiments — it decides whether one counts as *progress* for the patience
  counters. Higher bar = the run **stops sooner**, not the agent blocked.
- **Advisory is, on this project's own evidence, the option least likely to work**
  — 0024, v4, and 0027 are three recorded failures of advisory guidance. Shipped
  anyway because it is free and it *is* the experiment: if the agent is told a gain
  is noise and grinds on, that is data point four and the structural version gets
  made on evidence.
- **`val_se` understates uncertainty under an ordered split** (`split_on`): the
  bootstrap assumes i.i.d. validation rows, which a time-ordered split is not.
  Inherited from 0022, unaddressed.
- **The payload is 14 fields now.** 0018's guard caught the addition and forced it
  to be declared, which is the guard working as designed.
- **No notebook yet for `val_se`.** This chunk is untaught.

## 2026-08-11 (latest) — v7 ran: the brief does not cut the orientation tax

**Machine:** MacBook Air | **Branch:** master | **226 tests pass** (no library code
changed — this was an experiment).

**Decision record `0027`. It supersedes the central claim of `0026`, which was
written earlier today. 0026's implementation stands; its hypothesis does not.**

### The result

Orientation per episode, both runs recomputed with **one** function
(`orientation_share`), because 0025's published v6 numbers were hand-counted and
do not reconcile:

| | ep1 | ep2 | ep3 | ep4 | ep5 | resumed mean |
|---|---|---|---|---|---|---|
| v6, no brief | 8% | 32% | 50% | 32% | 50% | **41%** |
| v7, brief | 0% | 40% | 44% | 35% | — | **40%** |

**No change.** The brief did not move the number it exists to move.

### What actually happened, which is the interesting part

The **composition** of orientation changed completely while its size did not.
`inspect_data` per resumed episode went 3,2,3,3 → 1,1,2 (the `SYSTEM_PROMPT` fix
landing exactly as designed), and the agent spent the freed calls on
`explore_journal` instead: 3,8,3,4 → 7,7,4.

So **orientation is not an information deficit that can be paid off — it is
roughly a constant fraction of whatever episode it is given.** 0025 saw one face
of this (a bigger cap buys more orientation); v7 shows the other (free information
buys none). Three advisory fixes in a row have now failed to change behaviour.
Whatever solves this will be structural.

### Two things NOT claimed, both tempting

- **v7 best 0.11977 vs v6 0.12525.** Inside one `val_se` (~0.010, 0022). Not
  evidence of anything.
- **v7 used 73 calls / 65 requests vs v6's 98 / 90 for the same 18 experiments.**
  Not attributable to the brief: **episode 1 is configured identically in both runs**
  (a fresh run gets no brief either way) and differed by 8 calls and 2 experiments.
  That is the noise floor, and it is large against a 25-call total difference.

### Gotchas / open questions

- **The v7 script printed a misleading comparison on its first run** because it
  hardcoded 0025's hand-counted baseline. Fixed to recomputed values. General
  lesson: a baseline not computed with the same code is not a baseline.
- **The brief stays**, demoted to a cheap correctness aid — it carries the metric's
  *direction* to an agent with no way to derive it, which is a correctness argument
  independent of the failed one. Do not size an episode budget assuming it helps.
- **The remaining options for the tax are all structural**: charge orientation
  against a budget, cap `explore_journal` per episode, or have the episode driver
  make the first query and hand over the answer. None are designed.
- **n=1 vs n=1.** This rules out a large effect, not a small one.
- **Next:** decide which structural lever to try, or park the tax and move to the
  `harness/` loop and `val_se` (0022).

## 2026-08-11 (earlier) — The derived brief is built; it is not yet measured

**Machine:** MacBook Air | **Branch:** master | **226 tests pass** (215 + 11).

**Decision record: `0026-the-brief-is-rendered-not-remembered.md`. Accepted and
implemented — the first record in a while that is not "not yet implemented" — but
read the "What it costs" section before trusting it: the thing it claims is
unmeasured.**

### Done this session

**1. `harness/brief.py` — `render_brief(session)`.** A `RUN STATE` block rendered
from the journal: task, target, metric **and its direction**, the split, counts,
baseline, best (plus the table it was trained on, looked up from the leaderboard
row because `journal["best"]` is a pointer that does not carry it), the stop state,
and tables built but never trained on. ~170 tokens, 9 lines.

**2. Wired as a *dynamic* system prompt** (`@agent.system_prompt(dynamic=True)`),
so it is rendered when the episode starts rather than when the agent is built.
Returns `None` on a run where nothing has happened — deliberately, so a first
episode is byte-identical to a run that never had episodes, which is what makes
v6 and v7 comparable.

**3. `SYSTEM_PROMPT`'s "`inspect_data` first, always" is scoped now.** It was a
run-level rule obeyed *per episode* — one wasted call every time, flagged in
0024 as the prompt's fault. It now applies to a run that has not started, and
points at the run state for the rest. **No test can catch a regression here**; the
only check is a live run's opening call.

**4. `scripts/run_house_v7_brief.py`** — the experiment, written and **not run**.
Controlled repeat of v6: same dataset, model, cap, budget, and prompts, including
the resume prompt that still says "earlier episodes did things you have no memory
of" even though the brief now contradicts it. Rewriting that sentence would improve
the prompt and forfeit the comparison.

**5. `notebooks/phase3_harness_step7_brief.ipynb`** — 29 cells, 13 code, 12
smoke-tested in order (§9 is the live LLM).

### The number this is all for, and it is still v6's

    orientation per episode, v6 (no brief):   26%  50%  26%  43%

That is what v7 has to move. **Three outcomes and the middle one is likeliest:**
the share falls and the freed calls become experiments (the brief works); the share
falls and the episode just ends sooner (cheaper, no more productive); or the share
does not move, in which case **0026 is a well-argued no-op and the record should be
marked as one** rather than quietly left standing.

### Gotchas / open questions

- **Running v7 costs API budget and has not been run.** It needs `config.env` and
  roughly what v6 cost (~13 min, 90 requests). Nothing else in this session needs
  it; the code and its tests stand on their own.
- **The flatness test is the load-bearing one.** `test_the_brief_does_not_grow_with_the_run`
  fakes 151 leaderboard rows and asserts the brief moves by <20 chars. It is the
  only thing standing between this design and someone adding "recent experiments",
  which is the natural next request and the thing that breaks it.
- **An empty system-prompt part is still sent on a fresh run.** `render_brief`
  returns `None` → the hook returns `""` → PydanticAI adds a zero-length
  `SystemPromptPart`. Harmless with every provider tried; noted because it is
  visible in transcripts and looks like a bug.
- **Two things now describe the run's state** — the brief and `note` on every
  training result. Deliberate (one is for a fresh agent, one for a working one) and
  the first place to look if a run starts contradicting itself.
- **0018's three prompt rules do not cover the brief**, which is entirely
  run-specific and names a tool. 0026 argues that is a missing category rather than
  a violation — `SYSTEM_PROMPT` is the rules of the game, the brief is the
  position — but if that distinction stops holding, this is where the rot starts.
- **Next:** run v7 and record the answer. Then the episode loop into `harness/`,
  which per 0025 also means deciding where the "episodes on above ~35 experiments"
  switch lives. Then `val_se` (0022).

## 2026-08-10 (latest+1) — `explore_journal` is taught, and writing it found a bug

**Machine:** MacBook Air | **Branch:** master | **215 tests pass** (214 + one new).

**No decision record** — nothing was decided, one thing was fixed and one thing was
written down. The design records this teaches from are `0020`, `0021`, `0022`, `0024`,
`0025`.

### Done this session

**`notebooks/phase3_harness_step6_journal.ipynb`** — 46 cells, 19 code, 18
smoke-tested in order (§11 is the live LLM). Clears the "no notebook for this chunk"
item logged two entries down. Demo run: 800 apartments, multiplicative target
(skew 1.64), 10 experiments over 5 tables — three encodings, a ratio transform, both
target transforms, two model sizes — so every query has a real answer.

Sections: the empty-run query first (declared columns, §2), then the cross-run
questions (§4 encoding method, §5 target transform, §6 params), the two traps (§7
metric direction, cv-vs-val), the no-leakage argument (§8), free/silent/after-finalize
(§9), and episodes as the actual justification (§10).

**Target transforms are taught here too**, via §5 rather than in their own notebook:
`target_transform` is a column in `lb`, so "did log1p pay" *is* a journal query. That
was the cheaper way to close the other half of the untaught chunk, and it makes `0021`'s
argument from the far end — a per-experiment argument leaves a column to group by; a
project-level setting would leave nothing and the question would be unanswerable.

### The bug the notebook found

**`explore_journal`'s `tables` frame reported zero columns added, for every table, in
every run since it was written.** `_journal_frames` read `t.get("columns")`; journal
table rows carry **`new_columns`** and have never had a `columns` key. Not a crash — a
confident `0`, to the one question that frame exists to answer, on a field the
agent-facing docstring advertises. One-word fix plus
`test_a_table_reports_the_columns_it_added`.

Worth noticing *how* it surfaced: six unit tests cover this tool and all of them pass
either way, because none asserted a column count it had no independent reason to
expect. The notebook asked "which tables were built and never trained on, and how big
were they" with the answer known in advance. That is the argument for the notebook
convention, and this is the first time it has paid in a bug rather than in prose.

### Gotchas / open questions

- **`notes` is dead weight, and unsafely so.** It comes from `journal["log"]`, which
  **no agent-facing tool writes** — `journal.append_note` exists and nothing calls it.
  Worse, unlike `lb` and `tables` its columns are *not* declared, so it is a `(0, 0)`
  frame and `notes["hypothesis"]` raises `KeyError` — exactly the failure the declared
  columns on `lb` were written to avoid. Harmless while always empty; a trap the day a
  note tool lands. Either declare its columns or drop it from the namespace.
- **The optimism number is queryable now, and it is bigger than the yardstick.** In
  this demo every strong model has `val_score - cv_score` of 0.010–0.019 against a
  `fold_spread` of ~0.008–0.012 — on all six. That is `0022` reproduced on a second,
  synthetic dataset without touching Kaggle, and it is cheap to re-check anywhere.
- **It does not sort by "did the encoder see the target".** The largest gap belongs to
  a *frequency*-encoded table. The pattern that fits is "the better the fit on the work
  half, the more of it fails to transfer" — worth remembering before blaming target
  encoding for a val/cv gap.
- **Notebooks from earlier phases are still stale** from the 0019 score shift. Step 6
  is current; steps 1–5 and the phase-1/2 notebooks are not.
- **Next is unchanged and now better motivated:** the derived brief. §10 states the
  case in the notebook — the tool is discoverable from its docstring alone, but
  orientation still eats ~1/3 of an episode, so the brief is complementary, not
  optional.

## 2026-08-10 (latest) — Episodes work; the agent will not end one by itself

**Machine:** MacBook Air | **Branch:** master | **214 tests pass** (no library
code changed — this was an experiment).

**Decision record: `0024`. Accepted, not yet implemented: the loop lives in
`scripts/run_house_v5_episodes.py`, not in `harness/`.**

### Two runs, two answers

**v4 — the agent does NOT end its own episode.** Asked in plain terms to follow
one idea then stop, it ran all **25 experiments** in a single episode and
finalized. No second episode. This refutes the parked design's allocation ("the
episode stop is cheap and reversible, so the agent can own it") — it may own it,
it will not exercise it. Same under-compliance pattern as every other advisory
steer since July.

**v5 — with a structural cap (`UsageLimits(tool_calls_limit=6)`,
`max_experiments=12`), episodes work.** Peak context per episode
**9,170 / 11,170 / 12,274 / 9,971** against v3's single climb to **43,691**:
linear becomes sawtooth, 3.6x lower peak, and it does not grow with run length.
That is the RLM double-check, measured on both sides with `harness/usage.py`.

**A resumed agent orients itself, unprompted.** The resume prompt said only
"earlier episodes did things you have no memory of" — no counts, no scores, **no
mention of `explore_journal`**. Every resumed episode opened
`inspect_data -> explore_journal -> explore_journal`. The tool is discoverable
from its docstring alone.

### The catch, which reverses something I predicted

Orientation works but is **not free**, and I had guessed the brief might become
unnecessary. Wrong:

| ep | calls | orienting | new experiments |
|---|---|---|---|
| 1 | 7 | 0 | 2 |
| 2 | 7 | 3 | 2 |
| 3 | 11 | 3 | 1 |
| 4 | 9 | **8** | **0** |

Episode 4 called `explore_journal` **seven times and trained nothing** — a
degenerate mode where orientation eats the episode; only the loop guard stopped
it. The tax is a fixed ~2-3 calls, so it scales as `1/episode_length` and hurts
most exactly where short episodes are wanted. **The brief and the tool are
complementary, not alternatives.**

### Gotchas / open questions

- **`SYSTEM_PROMPT`'s "`inspect_data` first, always" is a run-level rule obeyed
  per-episode.** Every episode opened with a redundant `inspect_data` — the task
  and metric are already in the journal. One wasted call per episode, and it is
  the prompt's fault. Fix when the brief lands.
- **The seam cost was not isolated.** At cap=6 orientation overhead dominated, so
  we still do not know what cutting mid-idea costs on its own.
- **v5 scored 0.12873 vs v3's 0.11354** — from 5 experiments against 24. This
  measured orientation and context, not model quality. Do not read it as evidence
  against episodes.
### v6 (cap 16, max_experiments 30) — episodes do real work, and two of my own
### claims from 0024 did not survive it

**Decision record `0025` supersedes those two claims. 0024's decisions all
stand.** 5 episodes, 18 experiments, 12.9 min; 2-6 experiments per episode
against v5's 0-2. Peaks 18k/21k/23k/28k/19k — bounded, and ep5 *falling* shows
peak tracks what happens **inside** an episode, not run position. Episode 5 ended
by the agent, calling `finalize_run` — so the agent ends an episode only when the
*run* is done, never for context hygiene.

**Correction 1 — the orientation tax is ~1/3 of every episode at BOTH caps**
(v5: 43/27/89%, v6: 26/50/26/43%), counting `explore_journal` plus repeat
`inspect_data`. 0024 called it "a fixed 2-3 calls" that a longer episode would
dilute; wrong — given room the agent orients *more*. A bigger cap is not the fix;
**the brief is**, which is 0024's conclusion for a reason that actually holds.

**Correction 2 — episodes cost MORE per experiment at this scale**: 66,455 vs
v3's 45,650 input tokens/experiment (1.20M over 90 requests vs 1.10M over 48).
Episodes buy **feasibility, not cost**. The model `total ≈ ½·slope·turns²`
predicts v3 to within 6%, giving: crossover ~35 experiments, single-run
impossible ~60-80, and at 150 it is ~40M tokens/270k peak (dead) vs ~10M/28k.
**So: episodes on for long runs, off for short ones — not the default path.**

**Not claimed:** v6 best 0.12525 vs v3 0.11354 is ~1 standard error given
`val_se` ~0.010 (0022). Whether episodes cost quality is **unmeasured**.

- **Next:** the derived brief (now the primary lever, not a nicety); then move
  the loop into `harness/`; then `val_se` (0022).

## 2026-08-10 (earlier) — Three live Kaggle runs measured the leaderboard problem

**Machine:** MacBook Air | **Branch:** master | **214 tests pass.**

**Decision records: `0021` (target scale), `0022` (the yardstick), `0023` (the
third split). 0022 and 0023 are `accepted, not yet implemented` — read them
before trusting any `val_score` in this repo.**

### Done this session (after the entry below)

**1. `explore_journal` — the eighth tool.** Read-only code over the run's own
records: `lb` (one row per experiment), `tables`, `notes`, `run` (task, split,
stop, best, baseline, `higher_is_better`). No rows at all, so unlike
`explore_experiment` it needs no leakage argument. Free, works after finalize.
Two calls worth knowing: `params` is a **JSON string** (a dict cell is
unhashable, so `groupby("params")` would raise), and columns are **declared**
so an empty run hands over a well-formed frame instead of a `KeyError`.

**2. Target transforms (0021).** `train_tabular_model(target_transform=...)` over
a fixed menu — `none`/`log1p`/`sqrt`. Ulf's framing, and it is the point: a
per-experiment argument makes it an *experiment*, not a project assumption. The
agent immediately ran a controlled pair (e2 `none` 0.1366 vs e4 `log1p` 0.1287,
same table and model) and switched. Every score is reported on the **original
scale** — five call sites had to learn the inverse and each fails silently if
missed.

**3. `harness/usage.py`** — context growth per run, from the `input_tokens` each
response already records. This is the baseline for judging RLM-style changes.
Both earlier runs profiled retroactively; curve is **linear**, ~830–990
tokens/turn, peak 29k–44k. Linear context means quadratic cost, which is what
makes 150 experiments unaffordable rather than merely impossible.

**4. Logfire works fully offline.** `logfire.configure(send_to_logfire=False)` +
`logfire.instrument_pydantic_ai()` + an OTel span processor writing local JSONL.
103 spans on a real run. Note `local=True` does **not** mean local dashboard (it
returns a non-global instance); there is no bundled UI — point
`additional_span_processors` at a local OTLP collector if you want traces.

### The three runs, and what they proved

Kaggle house-prices, `tensorx/deepseek-v4-flash-0731`, same four-sentence prompt
every time. Runners in `scripts/`. **`_runs/` is gitignored** (128MB of fitted models) and so
is **`house_prices/`** (Kaggle data, not ours to redistribute) — download
train.csv/test.csv from the competition and `train_no_id.csv` is train.csv minus
the `Id` column.

| run | data | exp | our val | Kaggle | rank |
|---|---|---|---|---|---|
| 1 | with `Id` | 15 | 0.12095 | not submitted | — |
| 2 | no `Id` | 17 | 0.12162 | **0.12368** | ~831/3739 (top 22%) |
| 3 | no `Id` + transforms | 24 | 0.11354 | **0.12298** | ~730/3739 (top 20%) |

**`Id` was a feature in run 1** (train 1–1460, test 1461–2919 — entirely
disjoint). Dropping it moved `val_score` by 0.0007, i.e. nothing: random folds
and a random val split make it inert in-sample, and the damage is serve-time
only. No CV, no holdout drawn our way, and no ablation can see this.

**The big one: our val said the target transform bought 0.00808. Kaggle says
0.00070.** 91% was optimism, and optimism grew **4.6x** (0.00206 → 0.00944) for
**1.4x** the experiments (17 → 24). Bootstrapping `val_score` over its own 292
rows: **sd ~0.010**, CIs of the two runs almost fully overlapping. The gain was
smaller than one standard error of the instrument that measured it.

That is 0022: **0019 moved ranking onto `val_score` and left the yardstick on
`fold_spread`.** The bar is `0.25 x fold_spread ≈ 0.0025`; the noise in the
compared number is `≈ 0.0098`. Four times too small, and the two happening to be
numerically close on this dataset is why it hid.

### Gotchas / open questions

- **Correcting myself:** after run 2 I wrote that the third split "isn't urgent"
  and the premium exceeded the risk. Run 3 refutes that. One optimism measurement
  at n=17 was one point on a curve that climbs steeply. 0023 records it.
- **0022 first, it is much cheaper.** `val_se` by bootstrap costs milliseconds
  and tells a run *while it is running* that a gain is inside the noise. The
  third split only tells the reader afterwards. Both are wanted; neither
  substitutes.
- **Nothing warns that a feature's range at predict time is disjoint from
  training.** Proposed and not built: flag near-unique **and** monotonic columns
  at `inspect_data` — advisory, two branches (drop it if an identifier, pass
  `split_on=` if it is time). Note `split_on` already solves this and defaults to
  `None` on purpose; the detector's job is to make the mechanism discoverable.
- **Still missing vs a 0.11775 Kaggle notebook:** linear models + `blend_models`
  (`blend.py` is written and unregistered since 2026-07-29), a `fit_*` family for
  learned imputation/clipping/scaling (currently guarded by 0011 with **no
  sanctioned path**), and `onehot` in `fit_encoding` (`get_dummies` is
  batch-dependent by construction).
- **No data dictionary reaches the agent** — there is no channel for it. It cost
  less than expected here: the model recovered every ordinal order
  (`Po<Fa<TA<Gd<Ex`, `Unf<LwQ<Rec<BLQ<ALQ<GLQ`) from pretraining. Ulf has said
  adding the channel is fine; not built.
- **Free tools leave no journal trace.** `explore_experiment`/`explore_journal`
  write nothing, so the only record of what the agent *looked at* is the
  transcript. Fine for one run, a hole for studying behaviour across runs.
- **No notebook for this chunk yet** — `explore_journal` and target transforms
  are untaught. Step 5's notebook covers `explore_experiment` only.

## 2026-08-10 (later) — The agent can see its own mistakes; the RLM kernel was declined

**Machine:** MacBook Air | **Branch:** master | **197 tests pass.**

Started from "we should tackle the readonly python sandbox, not sure how we
handle the RLM part". The RLM part turned out to be two things bundled, and
separating them was most of the work.

**Decision record: `0020-the-sandbox-has-no-kernel.md`. Read it for the
reasoning, especially the kernel rejection. This entry is only what changed.**

### Done this session

**1. The read-only sandbox was already built** (`explore.py` + `run_readonly`,
tested since phase 2) and simply never registered in `agent.py`. The work was
never the sandbox — it was deciding what to point it at.

**2. `explore_experiment(exp_id, code)` is the seventh tool.** Read-only code
over one experiment's rows: the **whole source table** (including string columns
the model could not train on) plus `oof_pred`, `residual`, `fold`. This closes
the `score → hypothesise` arc that has been called the missing arc since
2026-07-30. Free like `explain_table` — no leaderboard row, no budget, works
after `finalize_run`.

**3. No persistent kernel — the rejection is the real decision.** RLM's context
economics come from the *sandbox boundary*, which we already had; a live kernel
adds only names surviving between calls. That is run state not in `run_dir`
(against 0003/0004/0010 and `RunSession`'s resume promise), it dies at the
episode seam anyway, and it saves re-pickling a ~100KB frame. Declined, with a
test (`test_nothing_survives_between_calls`) holding the line.

**4. `run_readonly` takes a namespace,** `{name: frame}`, and passes the agent's
function only the frames its signature names. This is `explore_journal`'s whole
prerequisite, built now because `explore_experiment` needed the same call.

**5. stdout is captured** (2000 chars) and returned beside the result —
**including on the error path**, so prints before a crash come back in the
message. Fixes a silent trap: models `print()` by reflex and got nothing.

**6. Two docstrings that claimed "no free-form code"** (`SYSTEM_PROMPT`,
`harness/__init__`) were already false since `transform_features` and are now
accurate: two tools run agent code, and what happens to the output is the
difference.

**7. `notebooks/phase3_harness_step5_residuals.ipynb`** — 42 cells, 20 code, 19
smoke-tested in order (§7 is the live LLM). Also: 0019 was missing from the
decisions README index; both it and 0020 are in now.

### The demonstration, on a purpose-built fixture

Apartment prices with a per-district price level, `district` left as a string so
training drops it. Mean residual **−47** on prices in the hundreds of thousands —
the aggregate says "unbiased, just noisy". One `groupby("district")`:
`altstadt +76,855`, `gruenau −61,825`. Biases cancelling. The fold breakdown is
flat (47k–52k), so it is a slice and not an unlucky split. Encode `district` →
`rmsle` **0.485 → 0.221**, `vs_source_table` `+0.264` against a fold spread of
`0.048`, and the per-district bias collapses to `+1,226` / `−5`.

### Gotchas / open questions

- **The leak that remains is through the agent's reasoning.** An agent picking
  features by staring at out-of-fold errors is doing selection informed by
  held-out predictions. Every data-path guard holds; no guard catches this. Named
  as hypothetical on 2026-08-05 — it is **live** now. That is the price of 0020.
- **The target is in the frame, deliberately.** `residual = y - oof_pred`, so
  omitting `y` would be decoration, not a guard. Protection is the absence of a
  persistence path (0010: code is stored, so a hardcoded literal dies at predict
  on a length mismatch), plus the 200-row cap. Careless threat model, not
  adversarial.
- **Unverified on a live run.** The open question is behavioural: does a model
  actually *reach* for `explore_experiment` after a bad score, or keep reaching
  for `price_squared`? Everything here is hand-driven and unit-tested. The
  docstring is the only place it learns what is in the frame.
- **Two output contracts now.** `run_readonly` returns `{"result", "stdout"}`;
  `explore_dataframe` still returns the summary alone, so the phase-2 notebook
  stays valid. Deliberate but worth knowing before touching either.
- **`explore_journal` is still unbuilt** and the namespace plumbing is waiting
  for it. Not skipped — deferred: its justification is a 150-experiment run, and
  that needs episodes. **Episodes are now the single thing blocking the overnight
  run.**
- Notebooks from earlier phases are still stale from the 0019 score shift.

## 2026-08-10 — The input is split; the leaderboard is a public leaderboard

**Machine:** MacBook Air | **Branch:** master | **Uncommitted, including the
2026-08-09 work below, which was never committed.**

Started from "what should we implement next", landed on the holdout — item 3 of
the sequence in the entry below, and a blocker for running overnight at all.
Ulf's call on the shape, after the trade was laid out: **two splits, and the
validation half is what the leaderboard ranks by.**

**Decision record: `0019-the-leaderboard-is-a-public-leaderboard.md`. Read it for
the reasoning, including the three-way split that was rejected and the two
reasons it may come back. This entry is only what changed.**

### Done this session

**1. `RunSession` splits the input once, before any tool runs.**
`run_dir/data/work.parquet` + `data/val.parquet`. `table=None` is the work half.
Time-ordered on `split_on`, seeded-random otherwise. Idempotent and re-read on
resume — a fresh process on the same `run_dir` must validate against the same
rows or the leaderboard compares scores taken on different data.

**2. Every experiment is scored on the validation half, and the leaderboard ranks
by it.** `_score_on_val` replays the trained table's lineage over the raw
validation rows — the same pipeline `finalize_run` bundles — so encoders apply
fitted parameters and nothing learned from validation rows enters the features.
`best`, `baseline`, `vs_baseline`, `vs_source_table`, `attribution` and the stop
rule all moved onto `val_score`. `cv_score` stays on every row and every tool
return: the gap between them is the run's record of a gain that lived only in the
folds.

**3. `resolve_table` is an allowlist now, not a convention.** It used to accept
any path that existed, *including absolute ones anywhere on disk* —
`available_tables()` was a display detail. Now: `None`, `tables/*`, and the work
table's own path (leaderboard rows name it, so the agent must be able to pass it
back). Nothing else resolves.

**4. `finalize_run` is genuinely once.** Two holes, both reachable before any of
this existed: it had no guard against a second call, and the `finalized` check in
`_refuse_if_finished` sat *after* an early return that fired whenever the stop
rule said keep going — so finalize early, keep training, finalize again. This is
**run integrity, not leakage**: with `val_score` on every training result the
agent queries the validation half continuously anyway. It becomes a leakage guard
only if a sealed third split is ever added.

**5. Report and prompt.** The report leads with the validation score, prints both,
states the split, and says in words that this is a public leaderboard. The prompt
explains the two numbers and what their disagreement means.

### Gotchas / open questions

- **The reported number is the one the agent optimised against.** 150 experiments
  ranked on the same rows is Kaggle's public leaderboard, and nothing in a run
  measures how much the search gamed it. Deliberate (0019); the fix if it ever
  matters is the third split.
- **`explain_table` stays on CV** — Ulf's call, reasoning in 0019. It ranks
  nothing and CV is the lower-variance instrument. Watch for the failure it
  admits: a table can ablate at `+0.10` and move `val_score` not at all, and no
  field says so.
- **Every score in the repo moved** (models train on 80%). Notebooks re-executed
  last session are stale again and still need re-running.
- **The size floor was wrong on the first pass** — 25 rows a side refused a
  120-row dataset outright. Now 5 (computability) plus a `val_is_thin` warning
  under 30. Informativeness is the data owner's judgement, not the library's.
- **Still blocking the overnight run: episodes.** No compaction, so one `run_sync`
  over 150 experiments still dies on context. The split makes the answer
  trustworthy; it does not make the night survivable.

## 2026-08-09 — Credit by ablation; the payload and the prompt cut hard

**Machine:** MacBook Air | **Branch:** master | **170 tests pass. Uncommitted.**

Started from Ulf reading §8 of `phase3_agent_map.ipynb` and not being able to
follow it. That turned out to be the notebook faithfully reporting a measure that
answers the wrong question, and everything below followed from chasing it.

**Decision records: `0017-credit-by-ablation.md`, `0018-the-payload-is-a-budget.md`
(new). `0016` marked superseded — its *unit* of credit survived, its *measure* did
not. Read those for the reasoning; this entry is only what changed.**

### Done this session

**1. `importance_share` is gone; credit is now measured by refitting.** New tool
`explain_table` fits once with every column, then once per contributing table with
that table's columns held out; `contribution` is the difference, signed as a gain.
Importance decomposes one fitted model's *attention*; the question is a
counterfactual, and the two come apart routinely. Verified on the motivating case:
the date columns score `+1.16`, the four original columns `-0.06` — the ones that
held **77%** of the splitting. Costs a fit per contributing table, writes no
leaderboard row, spends no budget, and is allowed on a table the stop rule has
already declined.

**2. One sign convention.** `vs_baseline` was the raw difference and flipped with
the metric; `vs_parent` was already normalised. A `logloss` run reported the same
improvement as `-0.19` and `+0.19` **in one result dict**. Everything now goes
through `_gain`: positive is better, always.

**3. `vs_source_table` (was `vs_parent_table`) compares an identical fit only.**
It used to compare against the source table's best score whatever model made it,
so a model swap was credited to the features. Now matched on `(model_type,
params)`; `null` otherwise, never bare — `vs_source_basis` says what would produce
it. Affordable only because `explain_table` covers the gap.

**4. Two concepts deleted, not renamed-and-kept.** "Parent" was `source_table`;
"block" was a table (one tool call writes exactly one). Four names for two things.
`vs_parent_table`→`vs_source_table`, `feature_blocks`→`columns_by_table`,
`block_credit`→`credit_by_table`, `journal.blocks`→`journal.columns_by_table`,
`"original columns"`→`"the raw input table"`.

**5. Training payload 21 → 13 fields, system prompt 924 → 229 words.** Three
subtractive rules, now in `instructions.py`'s docstring and guarded by tests:
say only what is *not derivable* (a competent data scientist working it out from
the tool signatures is not a reason to spend tokens every run), *not tool-specific*
(PydanticAI ships each docstring as the tool description — verified — so per-tool
facts belong there), and *not run-specific* (varying wording goes in the payload;
`note` and `vs_source_basis` are the whole of that category). Removed captions,
run constants, counters `note` already restated, and every justification whose
fact stayed. No "agent" in the prompt — it is the reader.

**6. Notebooks re-executed, not just renamed.** `phase3_agent_map` (cells 0–104;
§17 needs the API key and is untouched), `phase3_harness_step1`,
`phase3_harness_step4_stopping`. `phase1`/`phase2` needed nothing — their
`fold_metrics` come from `TrainResult`, which still has it.

### One finding that changed a lesson

`agent_map` §8 used to say the four original columns "contributed ~nothing".
Ablation says `+0.124`. They tie the baseline **alone** (§6) and are worth `0.124`
**alongside the date columns** — `price`/`area`/`rooms` say little until the model
can see *when* the listing went up. So the old framing was wrong in both
directions: the 77% was misleading and so was "~nothing". §8 now runs three
questions about the same four columns; §10 pairs it against the ratio branch
(`-0.026`, genuinely redundant despite being a real driver in the generating rule).

This also **retired 0016's advice to keep interacting features in one call**.
Tables chain, so both halves land in the same fit anyway, and leave-one-out
credits both. Verified on a pure XOR split across separate calls: each scores
`+0.82` (`test_both_halves_of_an_interaction_are_credited_across_separate_calls`).

### Gotchas / open questions

- **Uncommitted, and the renames break any saved journal.** No migration written;
  runs in flight are abandoned. Fine now, not later.
- **`nbclient` + `ipykernel` added as dev deps** (`pyproject.toml`, `uv.lock`) to
  execute notebooks. Drop them if unwanted.
- **`docs/feature_attribution_and_agent_iteration.md` is stale** — it is the long
  version behind 0015/0016 and still argues importance share. Flagged in the
  decisions README, not rewritten.
- **The prompt guards are keyword-based.** They catch `block`/`parent`/`branch`
  and verbatim docstring repetition — not paraphrase. The docstring-duplication
  test did **not** fire on a real paraphrased duplication Ulf caught by reading.
  This rule still needs a human.
- **`explain_table` understates every member of a redundant group.** Leave-one-out
  reports each table *given the others*, so two copies of a signal both read ~0.
  Documented in 0017; it does not test subsets.
- **`vs_source_table` is `null` more often now.** Watch a live run: if the agent
  changes model and features together as a habit, it never sees the number.
- **Not touched:** the stop rule's interaction with half-built ideas. A table
  holding one half of an interaction scores nothing and starts burning
  `table_patience` before its partner exists. The prompt no longer warns about
  this (cut as derivable); first thing to restore if a run prunes too early.

## 2026-08-06 — Two journal fixes shipped; metadata-kernel + episodes parked

**Machine:** MacBook Air | **Branch:** master

**Context for all of it: Ulf intends to run overnight, 100-200 experiments.**
Every item below is a consequence of that number. At 5 experiments none of this
matters; at 150 most of it does. Sequenced at the bottom.

### Done this session

**1. `journal.save` is now atomic.** Was `open(path, "w")` + `json.dump` —
truncate-then-write, so a crash or kill between the two left a corrupt or empty
`run_journal.json`, i.e. the whole run's state including hours of trained
artifacts that could no longer be attributed. Now writes `.tmp`, `fsync`s, and
`os.replace`s. A reader sees the old journal or the new one, never a partial one.

**2. `vs_parent` and `feature_blocks` are now persisted.** They were computed by
`train_tabular_model`, *returned to the model*, and then dropped — they never
reached the leaderboard row. So the two numbers that say whether an **idea**
worked (as opposed to which experiment scored best) lived only in the model's
context and died at the next reset. New `journal.update_experiment(run_dir,
exp_id, fields)` merges them in after the append, because both are derived from
the journal *including* the new row and so cannot be part of it. Neither affects
`best`/`baseline`, which still derive from `cv_score` alone.

Correcting an overstatement made while sketching this: `n_features` and
`fold_metrics` were **not** lost — `features` is on the row already and
`fold_metrics` is in `experiments/<id>/meta.json`. Only `vs_parent` and
`feature_blocks` were unrecoverable. 156 tests pass; verified end-to-end that a
derived table gets a real `vs_parent` and its block credit on disk.

### Parked: a metadata kernel (RLM-style) over the journal, not the rows

Nothing built. This is a parked design idea from reading
`PrimeIntellect-ai/prime-agent`, whose "RLM" abstraction turns out to be the same
move as 0003 — its model gets a persistent Python kernel, data lives in kernel
variables, and only what the model *prints* enters the context. Their sentence for
it is ours: the store is the state, the context is a view of it. **If this is
adopted, it needs a decision record — this entry is pruned in two weeks.**

**The observation.** 0003 got the rows out of the context. It did not get the
*records about the run* out. Every tool return — leaderboard rows, profiles,
block credit — still lands in context as JSON and stays there. At experiment 20
the agent is comparing 20 summaries by eye, and each one is re-sent every turn.
Same failure shape 0003 fixed, one level up: reasoning over data-as-text, with
the arithmetic done by vibes.

**The shape.** A persistent kernel seeded with the run's records and **no path to
`tables/`** — it cannot call `load_table`, cannot open a parquet:

    lb        leaderboard as a DataFrame, one row per experiment
    profiles  per-column profiles from inspect_data
    tables    table id -> lineage steps

Then `lb.groupby("encoding_method")["cv_score"].mean()` is exact, and the 19
experiments the agent is not currently thinking about are on disk instead of in
the window. What stays in context is a constant-size index (names + shapes, ~60
tokens) plus the result of the call just made — the working set. Context growth
goes from linear in tool calls to roughly constant.

**Secondary win: the view stops being pre-rendered.** `inspect_data` truncates at
`MAX_REPORTED_COLUMNS` and sets `columns_truncated`, i.e. the tool author decides
in advance which slice of the profile the agent gets. With a kernel the agent
asks its own question — `profiles["t1"].query("missing_frac > 0.3")`.

**Why 0001 survives it.** The guarantee is "our tools are the only path to the
data", and this touches no data. The kernel cannot create a column; every path
that produces a feature (`fit_encoding`, `transform_features`,
`train_tabular_model`) is untouched, still typed, still sandboxed, still
target-dropped. The kernel's only output is a decision about which tool to call
next, and that still goes through the same narrow door. 0001 rejected MCP because
a general client points a shell and an interpreter *at the data*; a kernel with no
read access to `tables/` is not that.

**The caveat, and it is the same one already logged below.** Split what goes in:

- **Safe:** leaderboard scores. Already out-of-fold — the whole point of
  `docs/oof_vs_final_encoder.md`.
- **Not safe:** anything target-conditional computed on the full dataset (e.g. a
  per-column correlation with the target in the profile). Let the agent compute
  freely over that to pick features and it is feature selection on all rows —
  selection leakage arriving through the *metadata* path while every data-path
  guard holds. This is exactly the hole named in the 2026-08-05 entry for
  residual feedback: **the leak is through the model's reasoning, not through
  data, so no guard catches it.** Same class, same answer — gate it at the door
  by not exposing it, or compute it per-fold.

So the profile has to be split: distributional facts about features (dtype,
cardinality, missingness) are kernel-visible; target-conditional statistics are
not.

**Cheap because it is already half-planned.** `explore_experiment(exp_id, code)`
over `features + oof_pred + residual + fold` routed through `run_readonly` is the
same machinery pointed at one experiment's rows instead of at run history. If
both are built, they should share the sandbox and the read-only routing — and
note that `explore_experiment` is the one that actually needs the leakage
argument, because it does see rows. The metadata kernel does not.

### Parked: episodes, the mechanism that makes a 150-experiment run possible

There is no compaction, so one `run_sync` driving 150 experiments simply hits the
model's context limit and dies. The fix is not a bigger context — it is a driver
loop where each iteration is a **fresh agent on the same `run_dir`**:

    while not journal.load(run_dir)["stop"]["finalized"]:
        run(session, EPISODE_PROMPT, model)     # new context every time

`RunSession` already holds only pointers, and its docstring already promises
this ("a fresh agent after a context reset can be handed the same `run_dir`").
What is missing is that the fresh agent currently wakes up knowing *nothing*.
Three pieces:

1. **A derived brief** — task, counts, baseline, best, `since_improvement`,
   not-finalized, plus how to query history. ~120 tokens, and *flat forever*
   because it is rendered from the journal rather than accumulated. Hook is
   PydanticAI's dynamic system prompt (`@agent.system_prompt`, gets `ctx.deps`).
2. **`explore_journal`** as above, so history is queried rather than recited. At
   150 experiments the journal is 60-120KB — reloading it as *text* costs
   15-30k tokens and stops being viable, which is what turns the kernel from an
   optimisation into the thing holding "resumable memory" up.
3. **`SYSTEM_PROMPT` says "no way to run free-form code"** — already loose
   (`transform_features` runs pandas) and would become wrong. A model told it
   cannot run code will not think to query.

**Episode length is a seam, not a count.** An idea spans several experiments
(build table -> train -> extend -> train), and cutting mid-idea makes the next
episode re-derive it and possibly repeat it. Soft target 10-15, hard cap ~25, and
let the agent end early when an idea resolves. Clean split available: the *run*
stop is a correctness concern so 0012 rightly gives it to the harness; the
*episode* stop is only context hygiene, cheap and reversible, so the agent can
own it. Add an `episode` column to experiments — it powers the loop guard
(experiments must increase per episode, else the loop spins and bills all night)
and answers "did that episode accomplish anything".

### Parked: 150 experiments breaks `best` — the holdout is now a blocker

`city` is `rng.choice`, independent of the target by construction, and it cleared
the bar and shipped as `best` with **five** experiments (2026-08-05 finding 3).
Keeping the max over 150 noisy draws is a 150-way maximum: with fold_spread
~0.08 a spurious winner is not a risk, it is a certainty. The final `cv_score`
stops being an estimate of generalisation and becomes "the luckiest of 150".

That promotes the already-planned holdout from prerequisite to **blocker**:
`build_report` takes `holdout_score` and no caller passes one. Do not run
overnight before it is wired, or the morning's leaderboard is confident and
uncheckable. Also confirm the two-level stop budgets (0014) are sized for 150 —
they are designed to *end* runs and may halt at ~30.

### Parked: SQLite for the journal — sketch

Deferred deliberately: it is a refactor of something that works, so it is the
*least* urgent item here. Recorded because the shape is decided. Note the
readability argument for JSON does not survive 150 rows, `sqlite3` is stdlib, and
run dirs are not in git so diffability was never real. Keep **one DB per
`run_dir`** (0003 self-containment; runs stay movable), and do cross-run analysis
by `ATTACH`ing many — that is where it really pays.

    run             singleton, CHECK (id = 1): schema_version, data_path, target,
                    task_type, metric, n_splits, higher_is_better, finalized,
                    refusals, created_at
    tables          table_id PK AUTOINCREMENT  <- kills the 0008 t3/t3 race
                    name GENERATED 't'||table_id, path UNIQUE,
                    parent_id -> tables.table_id (NULL = raw; ancestry() becomes
                    a recursive CTE), method ('frequency'|'target'|'ordinal'|
                    'transform'), code (0010: the code IS the artifact),
                    input_columns JSON, new_columns JSON, created_at
    experiments     exp_id PK AUTOINCREMENT, name GENERATED 'e'||exp_id,
                    table_id -> tables, episode, model_type, params JSON,
                    cv_score, fold_spread, fold_metrics JSON, n_features,
                    vs_baseline, vs_parent, artifact_path, created_at
    feature_blocks  (exp_id, block_table_id) PK -> experiments, tables;
                    importance_share, n_columns   -- 0016 credit, joined to the
                    tool call that produced the block
    notes           note_id PK, exp_id, episode, hypothesis, rationale,
                    next_ideas, created_at   -- the append_note layer

Four things the schema buys, beyond storage:

- **`best`/`baseline` become VIEWs, not columns.** `append_experiment` already
  recomputes them from the whole leaderboard "so it stays correct even if a row
  is edited" — that intent *is* a view, and as a view it cannot drift.
- **Metric direction handled once.** Put `cv_score * (CASE WHEN
  higher_is_better THEN 1 ELSE -1 END) AS score_signed` in the leaderboard view
  so every query is `ORDER BY score_signed DESC`.
- **`method` on `tables`** (already true in the row construction) makes the
  question worth asking one hop: `SELECT method, COUNT(*), AVG(vs_parent) FROM
  leaderboard GROUP BY method` — did target encoding ever actually help.
- **`parent_id`** replaces `ancestry()` and makes "all descendants of t3"
  expressible.

Migration cost is the test suite (asserts on journal dict shape), plus a
migration story or an explicit "runs before schema v2 are unreadable" — the
latter is fine for disposable research runs. `explore_journal` is unaffected
either way: it hands a DataFrame to `run_readonly` and does not care where the
frame came from, so the storage decision stays decoupled.

**If a `profiles` table is ever added, target-conditional statistics stay out or
get marked** — otherwise the metadata path becomes a selection-leakage route
while every data-path guard holds.

### Sequence

1. ~~atomic write~~, 2. ~~persist `vs_parent`/`feature_blocks`~~ — both done, and
   both were the *irreversible* ones: a field not written tonight cannot be
   recovered tomorrow.
3. **Holdout** — blocks trusting any overnight result at all.
4. **`explore_journal` + episodes** — makes the long run mechanically possible.
5. **SQLite** — any time; nothing depends on it.

The temptation is to start at 5 because it is the most interesting.

## 2026-08-05 — Orientation notebook: the whole agent, one call at a time

**Machine:** MacBook Air | **Branch:** master

Ulf could not hold the agent's shape in his head — "what does the agent call, and
what do those tools write". Nothing was built; `notebooks/phase3_agent_map.ipynb`
(109 cells, 47 code, all smoke-tested in order, ships unexecuted) drives all five
tools by hand and prints the return value **plus a `tree()` of the run dir** after
every call, so the folder filling up is the spine of the explanation. Sections
1-11 the happy path, 12 the five structural refusals, 13 stopping, 14-15 bundle +
`predict_bundle` round trip from raw strings, 16 the recap map, 17 the live LLM.

A wiki page in `docs/` was explicitly **not** written — the anti-wiki decision
from 2026-07-31 stands, and a notebook breaks loudly when it drifts.

**Three findings, from writing prose against real numbers rather than expected ones.**
All three are now teaching moments in the notebook, and they are the argument for
the holdout below:

1. **lgbm on the raw table barely beats the baseline** (0.6903 -> 0.6881) with
   fold_spread 0.0416 — a gain 20x under the noise. The dataset's signal is locked
   in the two string columns. Good: it is the state every run starts in.
2. **`importance_share` and marginal gain point opposite ways, in both directions.**
   On t1 the date block holds 23% of splits and delivered the entire 0.19
   improvement, while the 4 original columns hold 77% and delivered ~nothing
   (continuous columns have more split points). On t3 the reverse: `price_per_m2`
   took 37% of splits and made the score **worse** (`vs_parent -0.026`) — the model
   could already approximate the ratio from `price` and `area`.
3. **The noise floor accepted noise.** `city` is `rng.choice`, independent of the
   target by construction. Encoding it scored `vs_parent +0.029` against a fold
   spread of `0.081`. The bar is `min_gain_ratio 0.25 x 0.081 = 0.020`, so it
   **cleared** it and was recorded as progress — and `finalize_run` then shipped
   that experiment (e4) as `best`. The stop rule is a defence against *sustained*
   self-deception, not a per-experiment lie detector. With one comparison at that
   spread, no threshold can be right.

**Next — sequence changed, and (3) is why.** Ulf's original question was whether
residual feedback needs fold membership. It does not: `oof.parquet` is
positionally row-aligned with its source table (`train_cv` resets the index), and
OOF already absorbs folds by construction, so slices are addressable by row
position alone. A `fold` column is still worth 2 lines (deterministic from
`make_folds`) purely to tell a real weak slice from one fold's bad fit.

But the blocker is not folds, it is that `cv_score` is **both** the number the
agent optimises and the number the report headlines. Five experiments were enough
for that to ship a noise column. Residual feedback makes it strictly worse: the
agent would then be *selecting features using out-of-fold errors*, so OOF stops
being an innocent bystander. Note the leak there is through the model's reasoning,
not through data — no guard can catch it.

1. **Holdout the agent's tools cannot reach**, scored once at `finalize_run`.
   `build_report` already takes `holdout_score` and **no caller ever passes one**.
   Prerequisite for (2), not a follow-up.
2. **Residual feedback.** Shape: `explore_experiment(exp_id, code)` over
   `features + oof_pred + residual + fold`, routed through `run_readonly` only.
   Never into the frame `transform_features` sees, or 0009 breaks both ways.
   Free by construction — every counter in `stop.py` reads the leaderboard, and
   only training appends to it. That is also the relief for the
   measurement-vs-patience tension logged below.
   - Known hole, judged acceptable: free calls x 200-row cap means the target is
     reconstructable across calls. There is no path back in — a hardcoded literal
     is the only route, and per 0010 it is stored as code and dies at predict on a
     length mismatch. Threat model is careless, not adversarial (`sandbox.py`).
3. `blend_models` + `predict`; then §19 compaction, `max_tables`, exclusive-create
   id claim.

**Also noted:** `explore_dataframe` exists in `explore.py` and is **not registered
in `agent.py`** — the agent has no read-only code path at all today. Tests: 156 passed.


## 2026-07-31 — Decision records backfilled

**Machine:** MacBook Air | **Branch:** master

`docs/decisions/` now holds 16 records backfilled from this file and the commit
log, plus an index that defines the vocabulary (run / table / experiment /
journal / lineage / bundle) for someone who has never seen the project. Written
as teaching material: each has the problem, the decision, **what was rejected and
why**, and the consequences including the bad ones.

Ulf's framing: a wiki describing *how the system works* was rejected — it
duplicates code and notebooks and rots silently. A decision record is history,
dated, never edited to match later reality, so it cannot go stale the same way.

Convention added to `CLAUDE.md`. Several records exist mainly because *Ulf's*
call overrode the initial proposal (0014, 0015, 0016) and the rejected
alternative would otherwise have been pruned out of existence with the handoff
entry.

**Next:** unchanged — `blend_models` + `predict`, then residual feedback, then
§19 compaction. See the entries below.


## 2026-07-30 (latest+1) — Block-level feedback: what the agent needs to iterate

**Machine:** MacBook Air | **Branch:** master

**Read `docs/feature_attribution_and_agent_iteration.md` first** — it records a long
discussion with Ulf whose conclusions are not obvious and which *reversed* two earlier
positions. Summary:

- Measuring every link would still not answer "what is this feature worth": a chain
  gives marginal-given-everything-before, and credit is order-dependent (redundancy
  inflates whoever goes first; interaction punishes isolation). That is the Shapley
  argument, arrived at from the data side → §18 SHAP is the tool for the *explanation*
  question, and the wrong tool for the *search* question.
- **Ulf's reframe, which set the direction:** it is less about final credit, more about
  the agent iterating well. Search guidance only has to beat nothing — and nothing is
  what it got (one scalar per experiment after adding 16 columns).
- **Ulf: "it could build a feature on top of parent giving no improvement, but the two
  together help."** Correct, and it means greedy per-feature measurement actively
  misleads. **This reverses my earlier "kitchen sink = noise" criticism**: a big batch is
  the only way interacting pairs arrive in the same fit. Real trade — one-at-a-time is
  attributable but interaction-blind; batch is the reverse.

**Built:**
- **`feature_blocks` in every training result** — importance share summed per *tool call*
  (`journal.blocks` regroups by `new_columns`; no new bookkeeping). Never per column: a
  feature that only pays off next to another scores zero alone, and per-column feedback
  is how it gets pruned before its partner exists.
- **`vs_parent_table`** — gain over the best experiment on the parent table, **`None` if
  the parent was never trained**. Deliberate: it makes measuring the previous step *pay
  off* instead of being a rule. Same shape as the advisory that moved the agent off t6
  without ever refusing it.
- **`how_to_read_blocks`** ships with every result and the wording is load-bearing: zero
  share = "no evidence it helped **in this feature set**", explicitly *not* "useless".
  Nothing auto-prunes. Tables are immutable, so abandoning a branch is free — the agent
  is told that too.
- **SYSTEM_PROMPT steer: one tool call = one idea** (dates in one, ratios in another),
  but keep plausibly-interacting features together. A steer, never enforced — refusing a
  mixed transform would be wrong.
- Notebook `step4` extended to 43 cells (new §10), all code cells smoke-tested.
  **Tests: 156 passed.**

**Verified:** a dates block scores `share=0.298`, a squares block `share=0.000`,
`unused=2`, `vs_parent_table=0.0` — the exact signal that was missing.

**Next (sequenced, Ulf agreed):**
1. `blend_models` + `predict`.
2. **Residual feedback — the missing arc.** Mapped our loop against the CoreWeave post's
   (inspect → hypothesise → build → train → score(OOF) → decide): five of six arcs exist.
   The gap is **score(OOF) → hypothesise** — we write `oof.parquet` and never show it to
   the agent, so it hypothesises from column names alone. That is why it reaches for
   `price_sq`. Needs a tool + a summary format that does not leak the target back.
3. §19 compaction; `max_tables`; exclusive-create id claim.

**Caveat worth keeping:** that post's "60 cycles overnight compounds" assumes each cycle
carries information. Our data: 19 cycles found the same score as 8. Better feedback per
cycle beats more cycles — which is the argument for (2) above.

**Open tension — measuring a link is taxed by the same counter that protects exploration.**
`vs_parent_table` was framed as "makes good behaviour rewarding, not mandatory". That is
only half true. Every training call costs one of `max_experiments=25` *and*, when it does
not beat the incumbent, one unit of `patience=10`. An agent that trains each table before
extending it therefore accumulates no-improvement rows **by construction** — not because
it is failing, but because measurement costs an experiment. So thorough measurement buys
information and buys fewer ideas.

- **Candidate fix:** do not count a table's *first* training against `patience` — that is
  the measurement, not an attempt to win. Two lines.
- **Why not yet:** it loosens the stop rule again, and the loose version already bought 19
  experiments for the same score as 8 (see the entry below). Do not stack another
  loosening without a second dataset.
- **Also probably the wrong worry:** the risk this guards against is an agent *farming*
  `vs_parent` numbers. An LLM is not optimising a reward signal, and the failure actually
  observed, repeatedly, is the opposite — under-compliance with guidance until something
  structural intervenes.
- **Revisit when:** we run on a second dataset. Same evidence needed for the `patience=10`
  vs `patience=4` question, so do both at once.

## 2026-07-30 (latest) — Stop rule reworked, attribution gaps, and a real leakage hole

**Machine:** MacBook Air
**Branch:** master

### 1. `pd.qcut` — the guarantee had a hole (the important one)

A live run wrote `pd.qcut(df["price"], 10)` inside a good 60-line transform. **Proved
the skew:** loaded the shipped bundle, predicted the same 5 rows batched with 800 and
alone — different predictions. `qcut` takes deciles of whatever frame it is handed.

The step-3 notebook claimed the sandbox's import block means the code "cannot learn a
parameter". **That was wrong.** Blocking sklearn does nothing against `qcut`,
`rank(pct=True)`, `x / x.mean()` — pure pandas, all batch-dependent, and only the
*code* is stored, so replay recomputes them from whatever arrives.

Blacklisting calls is unwinnable (any aggregation is one), so `transform.unstable_columns`
tests the **property**: re-run the code over a random 25% subset, compare rows present in
both, reject columns that disagree. Costs one extra sandbox call at fit time, nothing at
predict (`check_stability=False` on the replay path). Verified against the real offending
code — flags exactly `['area_bin', 'price_bin']`, passes the other ~50 columns.
Docstrings in `transform.py` / `tools.py` and the step-3 notebook all corrected.

### 2. Stop rule reworked — **Ulf's call: "let it iterate and search"**

`patience=4` punished exploration. Two-level rule now:
- **`patience=10`** (run level, generous) — eight ideas that fail to find the ninth is
  the job, not a failing search.
- **`table_patience=3`** (per feature table, tight) — experiments on one table can only
  differ by model/hyper-parameters. Refuses **that table**, not the run, and names
  `transform_features`/`fit_encoding` as the next move. A redirect costs nothing; a stop
  costs the answer. Escalates to the run level only if the same dead table is demanded
  `max_refusals` times.

**This settles the deferred `SYSTEM_PROMPT` question: "features over tuning" does NOT go
in the prompt — it is now structural, in the tool.**

### 3. Attribution gaps (Ulf spotted this)

`t5` was built and immediately extended into `t6`; only `t6` was trained. So the jump
`0.316 → 0.2741` cannot be split between t5's columns and t6's — no measurement in
between. **Ulf chose option 2 of 3: report it, don't constrain the search.**
`journal.attribution()` walks the winner's ancestry; `finalize_run` returns
`unmeasured_tables`; the report gets a "How the winning feature set was built" section
plus an explicit **Attribution gap** warning. Rejected option 1 (refuse to stack on an
untrained table) for the same reason patience is generous.

**Tests: 152 passed.** Notebooks `step3` (45 cells) and `step4` (38 cells) both
regenerated and fully smoke-tested.

### Live run (deepseek, apartments) — three-way comparison

| rule | experiments | best |
|---|---|---|
| uncapped | 11 | 0.2718 |
| patience=4 | 8 | 0.2743 |
| patience=10 + table redirect | 19 | 0.2719 |

**All three found the same score** (differences ≪ fold spread 0.02). Two findings:
- **The structural refusal never fired — `refusals: 0`.** t6 hit its limit at e12, the
  *advisory* said "this table is used up", and the model went and built new features on
  its own. The guidance layer did the work; enforcement was the backstop that wasn't
  needed. Best evidence yet for the two-layer design.
- **The loosening bought nothing here: 19 experiments vs 8 for the same answer.** Open
  question, not a settled result — this dataset's signal is nearly all in one column, so
  there was never a ninth idea waiting. Needs a second dataset before judging.

**Next:**
- `blend_models` + `predict`, then §19 compaction.
- **Known gap (Ulf raised, deferred):** nothing caps table *building* — only trainings
  (`max_experiments=25`). A model could call `transform_features` forever without
  training. Not observed; the fix is `max_tables` on `StopPolicy`.
- Exclusive-create id claim (other half of the concurrency fix).

**Gotchas:**
- The model's summary claimed "city one-hot encoding" and "target encoding of listed_at
  captured it" — it did hand-roll one-hot in transform code (`is_berlin` etc., which is
  fine, stateless). Prose still drifts from the journal; the journal is the record.
- `attribution()` picks best-per-table with `max`/`min` off the metric — don't hardcode.

## 2026-07-30 (later) — Harness step 4: the stop rule (§19.3)

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`harness/stop.py`** — `StopPolicy` (patience=4, min_gain_ratio=0.25,
  max_experiments=25, max_refusals=3) + `evaluate(leaderboard, metric, policy)`, a
  **pure function of the leaderboard** so the interesting histories can be unit-tested
  instead of trained. On `RunSession(..., stop_policy=...)`, since it's this run's
  budget, not a library constant.
- **The yardstick is `fold_spread`, not an absolute threshold.** A gain counts only if
  it beats `0.25 x` the incumbent's own fold-to-fold wobble. An absolute number would
  need tuning per metric and per dataset and fails silently. The incumbent tracks the
  *true best*, so a creeping series of sub-noise gains can't be laundered into progress
  by moving the comparison point.
- **Patience > 1 on purpose** — a run branches; a dip before a real improvement is the
  search working. This is early stopping, same trade-off.
- **Two layers.** Advisory: every train result carries `since_improvement`,
  `experiments_remaining` and a note, so a cooperative model never gets refused.
  Structural: past patience, `train_tabular_model` returns `{"trained": false, ...}`.
- **A refusal is a return value, not an exception** — raising would eat `TOOL_RETRIES`,
  and exhausting that ends the run and discards every completed experiment. Being asked
  to train once more is not a malformed argument.
- **Escalation:** refusal → (at `max_refusals`) **the harness finalizes the run itself**
  and returns the bundle → after that, `RuntimeError`. The bundle is the deliverable; a
  loop that ends without one is the real failure.
- `finalize_run`'s auto-baseline calls with `enforce_stop=False` — the harness training
  on its own behalf. Without it, finalizing a stalled run crashes on a missing baseline.
- **Tests: 134 passed** (8 new in `test_stop.py`, 5 in `test_harness_tools.py`).
- **Notebook `phase3_harness_step4_stopping.ipynb`** (22 cells, unexecuted, all 8 code
  cells smoke-tested in order).

**Live comparison, same prompt/model/dataset:**

| | experiments | best | refusals |
|---|---|---|---|
| uncapped (step 3) | 11 | 0.2718 | — |
| stop rule | **8** | 0.2743 | 1 |

Gave up `0.0025` against a fold spread of `0.0209` — i.e. nothing measurable — for three
fewer experiments. **The agent complied with the first refusal and finalized.**

**Next:**
- `blend_models` + `predict` — the last two tools.
- §19 **compaction**: the journal grows without bound and a long run will outlive the
  context it's read into. This is the remaining half of §19.
- Exclusive-create id claim (the other half of the concurrency fix).
- **Ulf's call, now deferred to after a few more runs:** "prefer feature work over
  hyperparameter tuning" in `SYSTEM_PROMPT`. His decision was to see whether the tuning
  spiral still happens now the run is capped — keeps the two effects separable. First
  capped run still spent e5–e8 on tuning, so the answer isn't in yet.

**Gotchas:**
- The advisory describes the run *after* the experiment just finished; the refusal is
  decided *before* the next one starts. So a model is always told once before being
  refused — don't read the off-by-one as a bug.
- The stop rule says the *search* stopped moving. It knows nothing about whether the
  features are exhausted, and a leaky run converges fast and stops early, pleased.

## 2026-07-30 — Harness step 3: `transform_features` tool + the concurrency fix

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`transform_features` is now a tool** (5th on the surface) —
  `tools.transform_features(session, code, table, timeout)`. Runs the agent's
  `df -> df` code in the existing sandbox, writes `tables/<id>.parquet` + the
  lineage sidecar, records the table in the journal. Chains onto encoded tables
  (`t3` lineage = `["transform", "encode", "transform"]`, replayed in that order).
- **The design call: the target column is dropped before the code sees it.** Two
  reasons and the *second* is the one that decides it: (1) a derived column can't
  be a function of the target if the target was never in the frame — the sandbox's
  import block does nothing against `df["cheat"] = df["y"]`, that's pure pandas;
  (2) **there is no target at predict time**, so code referencing it would crash or
  silently compute something else when the pipeline replays. Same frame both
  places or "re-run the code at predict" is a lie. A `KeyError: 'y'` from the child
  is rewritten into a message that says which of those happened — a bare `'y'`
  reads like a typo and the model just tries again.
- **The lineage stores the code, not the columns.** Feature *pipeline*, not feature
  *store*: a new row goes through identical arithmetic. Journal table rows carry
  the code too, so a resuming agent sees what was *tried*, not just what landed.
- **`SandboxError` joined `ValueError`/`FileNotFoundError` as retryable** in
  `_retryable`. Code the model wrote badly is an argument problem — the argument
  just happens to be a program, and those sandbox messages are already written to
  teach ("use fit_encoding instead").
- **`max_concurrency=1`** (the cheap concurrency fix from the last entry) — asserted
  in a test via `agent._concurrency_limiter.max_running`. **Still only half a fix:**
  it's per-`Agent`, so two processes on one run dir need the exclusive-create id claim.
- **Tests: 121 passed** (+8 tool-layer, +2 agent; tool-surface set now 5).
- **Notebook `phase3_harness_step3_transforms.ipynb`** (40 cells, ships unexecuted,
  all 15 smoke-tested code cells run in order + a real deepseek run verified separately).

**Live run** (deepseek via Requesty, apartments dataset): baseline `0.6903` → best
`0.2718`. It parsed `listed_at` on its own and used the tool hard. **0 fatal errors,
no parallel tool calls observed in any single message.**

**Next:**
- `blend_models` and `predict` — the last two tools.
- §19 compaction + harness-owned stop. The live run did **11 experiments**; the last
  six moved the score by ~0.003. It doesn't know when to stop, and nothing makes it.
- Exclusive-create id claim + write-then-rename (the other half of the concurrency fix).
- Still Ulf's call: put "prefer feature work over hyperparameter tuning" into `SYSTEM_PROMPT`?

**Gotchas:**
- **A GBDT recovers `price / area` on its own** — the first draft of the notebook was
  built on "ratios are hard for trees" and the numbers refused to cooperate (raw beat
  the derived features). Rebuilt around a **datetime column**, where the raw model is
  level with the baseline because the information is *absent*, not merely awkward.
  Worth remembering before claiming a feature helps: check it.
- The model writes **kitchen-sink transforms** — ~20 columns including `price_sq`,
  `area_sq`, `price_x_rooms`. Free noise for a tree (it can already split anywhere on
  `price`). Safe, not smart; another argument for the stop rule.
- The notebooks in the working tree are modified (executed outputs from Ulf's runs) and
  were **not** committed — the convention is that they ship unexecuted.

## 2026-07-29 (later) — Live-run hardening: everything the first real agent runs broke

Four live runs on `tensorx/deepseek-v4-flash` (Requesty) found five bugs the unit tests
could not, because they were all *environment* or *agent-behaviour* bugs.

**Fixed:**
- **`config.env` wasn't found from `notebooks/`** — a notebook's CWD is its own folder. `load_api_key` now walks up from CWD *and* from the package. My smoke test ran from the repo root, which is exactly why it passed.
- **`run_sync` in Jupyter** — "This event loop is already running". Notebook cells must `await agent.run(...)`; scripts want `run_sync`. Noted in the notebook.
- **Ordinal encoding was impossible to request.** `inspect_data` reported `n_unique: 3` and never the values, so the agent could not state a level order it had never seen — it sent an empty list and the run died. Non-numeric columns with ≤20 uniques now list their `categories`. **After this fix the agent chose `ordinal` for `size` on the very next run, having gone `frequency` three runs running.** It wasn't reluctance, it was missing data.
- **One bad argument killed a whole run** (8 completed experiments discarded). Tool `ValueError`/`FileNotFoundError` are now raised as `ModelRetry` → the agent self-corrects in one turn. Other exceptions still stop the run loudly.
- **…which then died at PydanticAI's default budget of 1 retry.** `TOOL_RETRIES = 3` on the `Agent`, and unknown table paths now list the tables that exist instead of failing deeper down.
- **`_make_model` param collisions** (see previous entry) — same class of bug: agent-supplied input meeting our defaults.

**Result — "don't tune hyperparameters, work on features" beats tuning, on this data:**
`0.2825` in 5–6 experiments (untuned) vs `0.2870` in 12 (tuned). The agent's own ablation:
CatBoost on the raw table scores `0.6282` against a `0.6905` baseline — essentially all the
signal is in the encoding. It also **branched** exactly as the lineage design assumes:
`t1 (ordinal size)` → `t2 (freq city)` **and** → `t3 (target city)`, holding `t1` fixed to
compare two encoders. Reproducible across two runs, 0 retries.

**Next:**
- **Concurrency (real, unfixed):** `next_table_id`/`next_experiment_id` read-then-write without a lock. A model can put several tool calls in **one** message, and PydanticAI runs those concurrently (sync tools go to worker threads), so two `fit_encoding` calls can both claim `t1` and write the same parquet — one corrupt-parquet crash seen, consistent with this, but **not reproduced on demand**; it is not confirmed that this model emits parallel calls. Worth checking first: count tool-call parts per message in `result.all_messages()`; >1 in a message means it happens. `e1` collisions would overwrite artifacts *silently*, which is worse than the crash.
  Two fixes, do both eventually:
  1. `Agent(..., max_concurrency=1)` — one line, serializes tool execution, kills the whole class now. Costs parallelism we don't currently benefit from (each tool takes seconds anyway). **Unverified — no live run since finding it.**
  2. Exclusive-create id claim + write-then-rename. The real fix: `max_concurrency` is per-Agent, so it cannot protect a run dir touched by two processes (a resumed run alongside the original).
- Ulf's call to make: put "prefer feature work over hyperparameter tuning" into `SYSTEM_PROMPT`? The evidence above supports it, but this dataset is friendly to that conclusion.
- Then `transform_features`.

**Gotchas:**
- The model's prose lies in small ways — it reported the ordinal codes as "small=1, medium=2, large=3" when they are 0/1/2. The journal and the lineage sidecar are the record; the summary is a claim.
- Restart the kernel after pulling, or the notebook keeps the old module.

## 2026-07-29 — Harness step 2: `fit_encoding` tool + table lineage + non-Anthropic model

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`fit_encoding` is now a tool** (4th on the agent's surface) — `tools.fit_encoding(session, columns, method, table, categories, smoothing)`. Writes `tables/<id>.parquet` (source table + encoded columns), records the table in the journal (`next_table_id`/`append_table`), returns paths + column names only. Refuses to encode the target. Chains: pass a previous tool's `table` back in.
- **Table lineage — the real design call.** Each derived table gets a `tables/<id>.pipeline.json` sidecar holding the fitted encoders that produced it. `finalize_run` now loads the lineage of *the table the winning model trained on* instead of building an empty `Pipeline`. Per-table rather than per-run because a run branches (t1, t2, …) and a run-wide pipeline would be a lie about at least one model. Generalizes to `transform_features` (`Pipeline.add_transform` already exists).
- **Provider-agnostic model wiring** — `agent.gateway_model(model_id)` → `OpenAIChatModel` over an OpenAI-compatible gateway (Requesty, `https://router.requesty.ai/v1`); `build_agent`/`run` now accept a model string *or* a `Model` object. `load_api_key` reads `$REQUESTY_API_KEY` then `config.env`. **`DEFAULT_GATEWAY_MODEL = "tensorx/deepseek-v4-flash"`.** `.gitignore` now covers `*.env`, `.secrets/`, `*.key` — `config.env` is untracked and was never committed.
- **Bug found by the live run** (`models._make_model`): caller params collided with our keyword defaults — `TypeError: got multiple values` for lgbm/xgb, and CatBoost *rejects* two spellings of one setting (`iterations` vs `n_estimators`), which killed a whole agent run. Now merged through `_resolve_params` with a CatBoost alias table. 2 tests.
- **Notebook `phase3_harness_step2_encoding.ipynb`** (40 cells, ships unexecuted, all 19 code cells smoke-tested in order **including a real agent run on deepseek**). Covers the label-encoding trap, the three encoders, OOF-vs-full-fit, lineage, and journal-vs-model-prose.
- **Tests: 103 passed** (7 new in `test_harness_tools.py`, tool-surface set updated to 4).

**Next:**
- `transform_features` (sandboxed free code) — the lineage step type already exists, so this is the same adapter pattern plus `add_transform`.
- Then `blend_models`, `predict`, `journal_append`; §19 compaction + harness-owned stop (the live run did 12 experiments long past the point of gains).

**Gotchas:**
- The live agent frequency-encoded `size` instead of using `ordinal` — safe but suboptimal. The harness enforces *can't do the unsafe thing*, not *does the smart thing*. Don't confuse the two when reading a run.
- `result.usage` is a property in pydantic-ai 2.19, not a method.
- Requesty rewrites the model id in its response (`tensorx/...` → `deepseek/...`); that's the router, not a misconfiguration.

## 2026-07-29 — `OrdinalEncoder`: ordered categoricals as a first-class encoder

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`OrdinalEncoder` in `encoders.py`** — `OrdinalEncoder(categories={"size": ["small","medium","large"]}, unknown=-1.0)`, emits `<col>_ord`, registered in `ENCODER_CLASSES` so it round-trips through `Pipeline`/`save_bundle` JSON. Uses only `X` → no folds, no leakage (same class of encoder as `FrequencyEncoder`).
  - **Order is a required parameter, never inferred.** That's the whole point: `pd.factorize` orders by first appearance, `.cat.codes` lexicographically — both encode `Large/Medium/Small` as `0/1/2` and look fine doing it.
  - **Strict at fit, lenient at predict:** a value in the fit data missing from the declared order raises (typo / forgotten level); a category first seen at predict maps to `unknown = -1`, below every level, so one split isolates it. Duplicate levels also raise.
- **`fit_encoding(..., method="ordinal", categories={...})`** — third method alongside `target`/`frequency`; errors if `categories` is absent or doesn't cover `columns`.
- **`inspect_data`'s note** now steers the agent: ordinal when the categories are genuinely ordered, frequency/target when nominal.
- **Tests:** 8 new in `test_fit_encoding.py` + a bundle round-trip in `test_pipeline.py` (reversed row order + a level absent from the predict frame → codes unchanged). **Full suite: 95 passed.**

**Why (Ulf's prompt):** he'd been label-encoding by hand and asked whether that was wrong. It's not leaky — the real risk is that an ad-hoc `factorize`/`cat.codes` mapping isn't *stateful*, so train and predict can silently disagree, and nothing lands in the pipeline. Ordered columns being common, the fix was a real encoder rather than a warning.

**Next:**
- Unchanged from below: `fit_encoding` (now with three methods) still needs its **harness tool wrapper** in `tools.py`, plus `transform_features`, `blend_models`, `predict`.

## 2026-07-28 — Harness step 1: file-based tool layer + PydanticAI loop (thin vertical slice)

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **New package `src/tabular_agent/harness/`** — the constrained harness from the pivot.
  - `session.py` — `RunSession(run_dir, data_path, spec)`. Holds *pointers only*; the run dir is the state (§4). Resolves+caches the task spec (task type/metric auto-detected once, then pinned so leaderboard scores stay comparable).
  - `tools.py` — **the file-based adapter layer** (this was the real design call, see below): `inspect_data`, `train_tabular_model`, `finalize_run`. Paths in → artifacts into `run_dir/` → compact JSON summary + paths out. Plain functions, **no LLM dependency** → unit-testable and hand-drivable.
  - `journal.py` — `run_journal.json` (§19), minimal: `task`/`baseline`/`best`/`leaderboard`/`log`. `train_tabular_model` appends a leaderboard row **as a side effect of training** (§19.2) — can't be skipped. `best` re-derived from the whole leaderboard each append; baseline tracked separately so it's never shipped as the deliverable. `append_note` exists but isn't wired to a tool yet.
  - `instructions.py` — system prompt (§12 revision): only *order* + how to read results; no "please don't train outside the tool" guidance, because there's no other tool.
  - `agent.py` — thin PydanticAI loop. `DEFAULT_MODEL = "anthropic:claude-opus-5"`, `defer_model_check=True`, `deps_type=Deps(session)`. Registers exactly the three functions.
- **Tests:** `test_harness_tools.py` (10) + `test_harness_agent.py` (3, incl. **tool surface == exactly those 3** — fails loudly if anyone adds a `read_file`). **Full suite: 87 passed.**
- **Notebook** `notebooks/phase3_harness_step1.ipynb` (38 cells, ships unexecuted, all 15 code cells smoke-tested in order) — drives the whole pipeline by hand *first*, LLM only at the end. Live-API cell is commented out.
- `pydantic-ai` 2.19.0 added as a hard dep (`uv add pydantic-ai`).

**Design call made this session (Ulf's, from a 3-option question):**
- Handoff said "register exactly the existing library functions as tools" — **not literally possible**: every library fn takes/returns an in-memory DataFrame, while §4 demands stateless file-based tools. Chose **file-based paths** over the cheaper in-memory-deps option, because in-memory state dies with the process → §19 resume would have forced a rewrite. Hence `tools.py` as a real adapter layer.
- Scope: **thin vertical slice** (inspect → train → finalize) over all-7-at-once, so any adapter-shape mistake is made once.

**Next:**
- **The other four tools** — `transform_features`, `fit_encoding`, `blend_models`, `predict`. Same adapter pattern, now proven. This is where non-numeric columns (currently dropped-with-a-warning) become usable.
- Then: `journal_append` for the agent's note layer; §19 compaction + harness-owned stop; multi-provider model config.

**Gotchas:**
- `train_tabular_model` currently **drops non-numeric columns** and reports them in `dropped_non_numeric` — deliberate (a silent cast is how train/serve skew starts), and it's the hook for the `fit_encoding` slice.
- `finalize_run` **auto-trains a baseline** if the run never did, so the report can always answer "did this beat doing nothing".
- Bundles ship `final_model` (refit on all data) with weight 1.0 — matches the existing phase-1 convention in `test_pipeline.py`.
- pydantic-ai 2.19.0's `KnownModelName` literal **doesn't list `claude-opus-5` yet** (lags releases); the string passes through to the provider fine — that's what `defer_model_check=True` is for.
- Tool-surface assertions reach into `agent._function_toolset.tools` (private API). If pydantic-ai churns, that's the line that breaks.

## 2026-07-28 — Architecture pivot: owned constrained harness (not MCP-for-others)

**Machine:** MacBook Air
**Branch:** master

**Done (spec + design only — no code yet):**
- **The pivot (Ulf's call):** the "leakage-safe *by construction*" guarantee only holds if **our tools are the only path to the data**. A general-purpose MCP client (Claude Code's bash + scratchpad) makes free code the path of least resistance → around the sandbox/OOF discipline. So: **we own a thin, constrained harness; only the LLM is swappable.** MCP demoted from "the product / distribution surface" to **optional internal transport**.
- **Harness framework decision: PydanticAI** (leaning) — model-agnostic (satisfies "LLM free"), typed tools (matches `fit_encoding` et al., no hidden tools), self-managed message history (needed for §19 compaction), headless/embeddable. Explicitly **not** Claude/OpenAI agent SDKs (model lock-in) and **not** smolagents/code-first (the free-code escape *as a feature* = anti-fit). TUI deferred (interface is Axis B, not where the value is).
- **Spec updated** (`docs/superpowers/specs/2026-07-21-...md`), all as dated `Revision (2026-07-28)` blocks so original reasoning stays visible: §3 (diagram + note), §12 (instructions → harness system-prompt, boundary enforced by tool-set not guidance), §13 (ship the harness, `pydantic-ai` hard dep / `mcp` optional), §15 (decision #2 struck + superseded by rows 2′/11-13), §16 (autonomous loop-runner promoted to v1 core — the harness *is* that runner). Commit `dfc6e7b`.
- Also this session: docs `oof_vs_final_encoder.md` (why `final_enc` isn't leakage) `f1f31dd`; **spec §19** harness & resumable memory (`run_journal` = tool-written leaderboard + thin agent notes; compact on token budget; harness owns stop) `78a7fdd`.

**Next:**
- **Build the harness, step 1:** a thin PydanticAI loop that registers exactly the existing library functions (`inspect`, `transform_features`, `fit_encoding`, `train_cv`, `blend`, `predict`, `finalize`) as its *only* tools — headless, one model to start. Validates the constrained-tool-surface end-to-end before adding model-abstraction or the §19 journal.
- Then: `run_journal` writes (train side-effect + `journal_append`), §19 compaction/stop, then multi-provider model config.

**Gotchas / open:**
- Spec **title still says "MCP-Server für …"** — now stale; retitle later (didn't, to avoid breaking refs).
- Decision still a *leaning*, not locked: PydanticAI is young/moving fast; fine, but revisit if API churn bites.
- Phase-2 core lib (transform_features/fit_encoding) is done + committed; the pivot doesn't touch it — sandbox just gets *promoted* to "the agent's scratchpad".

## 2026-07-25 — Phase 2: transform_features + fit_encoding (+ notebook)

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **`transform_features` (§5.1)** — the stateless `df→df` escape hatch. Reuses the same sandbox: extended `sandbox.py` with a shared `_run` driver + `run_transform` (mode `"transform"`) that returns the **whole** DataFrame via pickle (not JSON-capped like `run_readonly`). Contract enforced in-child: must return a DataFrame **with the same row index** (no reorder/drop/add rows) → teaching errors otherwise. New thin wrapper `transform.py` (`transform_features(df, code) -> (df, summary)`), symmetric to `explore.py`.
- **Pipeline persistence** — `Pipeline.add_transform(code)` stores a `{"type":"transform","code":...}` step; `Pipeline.transform` re-runs it via the sandbox at predict (lazy import so predict doesn't need the sandbox unless a code step exists). Only the **code** is stored, never the produced columns → reproducible, no train/serve skew.
- **`fit_encoding` (§5.2)** — structured stateful encoding tool in `encoders.py`. Dispatches `method="target"` (needs `y`+`folds`, returns `(oof_features, final_encoder)` via existing `oof_target_encode`) vs `method="frequency"` (new `FrequencyEncoder`, X-only so no OOF, unknown category → 0). Both registered in `ENCODER_CLASSES` and serializable.
- **Tests:** `tests/test_transform.py` (7) + `tests/test_fit_encoding.py` (8). **Full suite: 74 passed** (libomp now installed — the old 7 lgbm/xgboost failures are gone).
- **Notebook** `notebooks/phase2_transforms_and_encoding.ipynb` — phase-style, low-jargon, covers **both** tools + an end-to-end transform+encode→train→predict. Ships **unexecuted**; every cell smoke-tested in order.

**Next:**
- **The MCP server itself (§3)** — none exists yet. Wrap the primitives (`inspect_data`, `transform_features`, `fit_encoding`, `train_tabular_model`, `blend_models`, `predict`, `finalize_run`) as the stateless, file-based tool surface + server-instructions + OTel spans. This is now the big remaining v1 piece.
- Optional: §18 post-hoc SHAP explainability.

**Gotchas:**
- Task-type string is **`"binary"`** (not `"classification"`) for `train_cv`/`make_folds`/`Pipeline`; `TrainResult` exposes `fold_metrics` (list), no `cv_score` — mean it yourself. (Both bit me while writing the notebook.)
- `libomp` fixed via `brew install libomp`; the `LDFLAGS`/`CPPFLAGS` export hints Homebrew prints are **compile-time only** and not needed — the pre-built lgbm/xgboost wheels just needed `libomp.dylib` present.

## 2026-07-24 — Phase 2 started: sandbox + read-only exploration tool

**Machine:** MacBook Air
**Branch:** master

**Done:**
- **Design change (Ulf's call), recorded as §17.1 in the spec:** dropped both original §17 items — enriching `profile_dataframe` with skew/quantiles, and a narrow `group_stats` tool — in favour of **one general read-only Python tool** next to `transform_features`. Reverted the uncommitted skew/quantile work in `inspect.py`/`test_inspect.py`; profile stays a cheap structural fingerprint (dtype/missing/unique).
- **`src/tabular_agent/sandbox.py`** — the §6 default tier: fresh subprocess per call, import whitelist (blocks `lightgbm`/`xgboost`/`catboost`/`sklearn` with *teaching* errors → structurally can't train; allows pandas/numpy/scipy + stdlib), CPU/mem limits, own temp cwd, wall-clock timeout, size-capped JSON result contract.
- **`src/tabular_agent/explore.py`** — `explore_dataframe(df, code)`: the read-only tool. `explore(df) → summary`, shown to the agent then discarded. Non-persistence + size cap = the structural guard that keeps the §17.2 leakage backdoor closed (can't hand-build a target-leaky feature).
- **Tests:** `tests/test_sandbox.py` (11 total across both) + `tests/test_explore.py`. All pass.
- **Notebook `notebooks/phase2_sandbox_and_exploration.ipynb`** — phase1-style walkthrough, low-jargon. Ships **unexecuted** (run cell-by-cell) but every cell smoke-tested.

**Next:**
- `transform_features` (§5.1): same sandbox, `df→df`, persisted as a pipeline step, re-run at predict. Reuses `sandbox.py` — just a second entry point + contract (return a DataFrame) + pipeline-step persistence.
- Then `fit_encoding` as a structured tool; §18 SHAP explainability is still an *option*.

**Gotchas:**
- **libomp still missing on this Mac** → 7 pre-existing failures in `test_models.py`/`test_pipeline.py` (lightgbm/xgboost `dlopen`). Unrelated to Phase 2 work; fix with `brew install libomp`. Everything else (49 tests) passes.
- pytest is the `dev` extra: `uv run python -m pytest` (after `uv sync --extra dev`).

## 2026-07-23 — explainability option added to Phase-2 spec

**Machine:** MacBook Air
**Branch:** master

**Done:**
- Added §18 to `docs/superpowers/specs/2026-07-21-tabular-agent-mcp-design.md`: per-prediction explainability ("influencing factors") via post-hoc grouped TreeSHAP. Flagged as an **option**, not a hard v1 requirement (Ulf's call).
- Also wrote `docs/target_encoding_leakage_insight.md` (who gets lied to: model vs. metric) — companion to the tutorial notebook.

**Key design decisions (from a long discussion with Ulf):**
- Post-hoc SHAP over glass-box (EBM/GAM): decouples "strongest model" from "explain it"; no accuracy sacrifice, only explanation-exactness-under-interactions, partly recovered via opt-in pairwise interaction values.
- Attribution unit = **raw input feature** (derived columns like one-hot / `_te` folded back via encoder metadata). Multi-feature "factors" need domain knowledge → out of scope.
- Native booster `pred_contrib` → no `shap` dependency. Composes linearly over the blend, reconciles exactly to the served prediction.
- Present in target space (additive vs. multiplicative for log-target). Label output "model reasoning, not causation." Optional declared monotone constraints attack "weird prediction" complaints at the source.
- Trustworthy only on leakage-clean encoding — SHAP is faithful to leakage-inflated feature importance.

**Context:** Ulf's prior company did demand forecasting with a block-additive neural GAM (`exp(Σ ReLU-MLP blocks)`); influencing-factor attribution was per-block. tabular_agent can't hand-design blocks (genre-agnostic), hence per-raw-feature SHAP.

**Next:** Phase 2 still open (profile stats §17, `group_stats` §17, now explainability §18).

## 2026-07-23 — target-encoding tutorial notebook

**Machine:** MacBook Air
**Branch:** master

**Done:**
- New `notebooks/target_encoding_step_by_step.ipynb` (unexecuted on purpose — Ulf wants to run it cell by cell). 15-row toy dataset; sections 1–7 = core (naive vs. OOF encoding, leak traced per row), sections 8–12 = deep dive: encoder-vs-model distinction, three schemes (A shared-fold OOF = the library, B fit-on-train = sklearn-Pipeline style, C nested CV) compared mechanically and empirically (B: honest metric but degraded model, ≈0.79 vs 0.81 AUC, 16× over-weight on a noise feature; A: ≈0.52 on pure noise = second-order optimism).
- Set up Python env on this Mac: installed uv, `uv sync` from uv.lock. All notebook cells smoke-tested.

**In progress / next:**
- Ulf is working through target encoding conceptually — found the second-order material dense; keep explanations concrete/low-jargon.
- Phase 2 (profile stats, `group_stats` tool) still open.

**Gotchas / open questions:**
- XGBoost fails to load on the MacBook Air: missing libomp (`brew install libomp`). Affects `models.py`, not the new notebook (uses sklearn's HistGradientBoosting instead).
- Jupyter is not a project dep; launch via `uv run --with jupyter jupyter lab`.

## 2026-07-22 — setup

**Machine:** MacBook Air
**Branch:** master (clean, pushed through c19ebc6)

**Done:**
- Established this handoff workflow (HANDOFF.md + CLAUDE.md convention).

**In progress / next:**
- Phase 2: profile stats + `group_stats` tool (see `docs/` Phase-2 requirements, commit c19ebc6).

**Gotchas / open questions:**
- (none yet)
