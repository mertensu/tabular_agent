# Otto: the first multiclass run

Opened 2026-11-03. Status: **paused 2026-11-03 after r1 and two multiclass fixes; r2 not run.**

## Question

Every dataset so far was regression or binary. Multiclass touches the probe, the
early-stopped fit that ships (0098), the refit, the report and the bundle with a
column of probabilities per class. Does a run go through end to end, and where does
it land? Otto Group Product Classification (Kaggle, 2015): 61,878 products, 93
obfuscated count features, 9 classes, scored by multi-class log loss.

## Design

- Data: the competition's `train.csv` into `otto_group/` (gitignored);
  `scripts/prep_otto.py` changes nothing and transcribes the data page as notes.
- Test rows: 20% of products (seed 0, stratified on the class), never seen by a run.
  The competition's own test set has no labels; scoring it would mean a Kaggle
  submission, which is not part of this.
- `references.py` (no LLM): the probe and an early-stopped LightGBM on the raw 93
  features.
- `runner.py`: one run of `ta.Agent()`, minimal profile with 0098, metric `logloss`,
  agent model `gpt-6.1-terra@eu` at medium effort. `id` is proposed as an identifier
  by name and accepted. The file is sorted by class, so `id` carries the target.

## Prediction (written before running)

- The run finalizes and its bundle returns a 9-column probability matrix; if
  something breaks, it is in finalize or the bundle rather than in the search.
- Early-stopped LightGBM on the raw table: about 0.47 log loss; the probe at 200
  rounds is worse by at least 0.01.
- The agent's features (row totals, non-zero counts, log or tf-idf transforms of
  the counts) are worth at most 0.01; the run lands within 0.01 of the
  early-stopped raw table.

## Readings

References (`references.json`): probe 0.470, early-stopped LightGBM 0.455 (751
rounds). Kaggle public leaderboard for scale: top 10% 0.429, top 25% 0.458, median
0.542 (a different test set; ours can differ by about ±0.01).

**r1: 0.473, and two bugs.** The run went through end to end (9-column bundle), but
both 0098's early-stopped fit and the refit on training + validation failed on
string class labels, and the run quietly shipped the probe fitted on training rows
only:

- `fit_fitted_ruler`: LightGBM encodes string labels for the training rows but not
  for `eval_y`. Fixed: the stopping fit gets integer class codes.
- `models._fit_frame`: every target went through `apply_reference_target`, which
  casts to float. Fixed: class labels pass through.

Test: `test_minimal_ships_an_early_stopped_refit_for_string_class_labels`. The agent
built row totals, non-zero counts and count shares (t2, t3); none beat the raw table
under the probe, and the shipped table was the raw 93 counts. Its first table (t1,
1.655) returned only its 6 new columns, dropping the raw ones; it corrected that
itself.

Paused before r2 and before `gap_split.py` finished, to move to IEEE-CIS: on
independent rows with clean columns, an early-stopped GBM already builds most of
what classic feature engineering would; the features that need other rows are
where the harness is meant to earn its place.
