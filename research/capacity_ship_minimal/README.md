# Does the default product ship enough model?

Opened 2026-10-26. Status: **r1 and r2 run 2026-10-26/27; readings below.** The first House run
exposed that the minimal profile always shipped the work table (fixed in 7e68cfc);
both readings are from runs after that fix.

## Question

`ta.Agent()` (minimal profile) ships the probe itself: one LightGBM at 200 rounds,
library-default learning rate, no early stopping, refitted on the selected table.
0035 built the probe as a ruler that "only has to rank". Is a ruler good enough
to ship, or does the same selected table, fitted with a real capacity budget,
predict materially better on rows nothing in the run has seen?

This decides whether the minimal profile's "one model for ranking and shipping"
gets a record superseding 0035, or whether the default should ship a
capacity-fitted model while the probe keeps ranking.

## Design

Two independent-row datasets, one small and one large, because the hypothesis is
that a fixed 200-round budget bites hardest on large tables:

| dataset | rows | task | metric |
|---|---|---|---|
| House Prices | 1,460 | regression | rmsle |
| Porto Seguro | 595,212 | binary, 3.6% positive | auc (gini = 2·auc − 1) |

Per dataset:

1. **Audit slice.** 20% of `train.csv` (seed 0, stratified for the binary task) is
   removed before the agent sees anything. No tool, ruler or finalize step can
   read it, unlike the run's own evaluation rows, which ranked the tables.
2. **One run of the default product.** `ta.Agent()` with its defaults, the minimal
   profile, on the other 80%; agent model `fireworks/deepseek-v4.1-flash-1017`
   with thinking enabled.
3. **Offline arms on the run's selected table** (same columns, same work rows,
   same target transform):
   - `probe` — what shipped (`run.bundle.predict`).
   - `lgbm_es` — LightGBM, learning rate 0.03, up to 5000 rounds, early stopping
     (200 rounds patience) on a random 15% of the work rows, then refit on all work
     rows at the stopped round count.
   - `catboost_es` — the same protocol for CatBoost.
   - The same three on the **work table** (no agent features), to separate what the
     features bought from what capacity bought.
4. Score every arm on the audit slice; paired bootstrap (2,000 resamples) for each
   arm's difference to `probe`.

## Prediction (written before running)

- House Prices: capacity buys little, ≤ 0.005 rmsle; 200 rounds is about right
  at 1,168 work rows.
- Porto Seguro: capacity buys ≥ 0.005 auc (≥ 0.01 gini) on the selected table.
- On both, the features' gain over the work table shrinks when measured at
  capacity (the pattern Rossmann v7 showed).

## Decision rule

- If the best capacity arm beats `probe` on the audit slice with a paired 95%
  interval excluding zero, **and** by at least 0.005 rmsle / 0.005 auc, on either
  dataset: the default should ship a capacity-fitted model, and the probe stays a
  ruler only.
- Otherwise: shipping the probe is defensible; write the record superseding 0035
  for the minimal profile.

One agent run per dataset measures one selected table each. The comparison is
between arms on the same table, so LLM variance does not enter it, but a second
run would be needed before generalising beyond these two tables.

## Files

- `runner.py` — carves the audit slice, runs `ta.Agent()`, writes `result_<ds>.json`.
- `evaluate.py` — fits the arms, scores them, writes `arms_<ds>_<tag>.json`.
- `restaurant_diag.py` — Restaurant only: reservation skew vs stale fit, writes
  `diag_restaurant_<tag>.json`.
- `rerank.py` — refits every table the run scored with each ruler, writes
  `rerank_<ds>_<tag>.json` (added after r1, for: does the probe pick the table a
  fitted model would?).
- Run directories and data copies live under `_runs/capacity_ship_minimal/`
  (untracked).

## Reading r1 (2026-10-26)

Audit-slice scores; the refitted probe reproduced the shipped predictions exactly
(max abs diff 0.0) on both datasets.

**House Prices** (rmsle, lower is better; 292 audit rows; selected t4, 112 cols):

| | work table | selected table |
|---|---|---|
| probe | 0.1430 | 0.1348 |
| lgbm_es | 0.1436 | 0.1367 |
| catboost_es | 0.1346 | 0.1275 |

- vs probe: lgbm_es −0.0019 [−0.0050, 0.0016]; catboost_es +0.0073 [−0.0048, 0.0200].
- feature gain: 0.0082 at probe, 0.0069 at lgbm_es, 0.0071 at catboost_es.
- The run's own evaluation score was 0.1254: best of 11 tables on 234 rows reads
  0.009 optimistic.

**Porto Seguro** (auc, higher is better; 119,043 audit rows; selected t5, 39 cols):

| | work table | selected table |
|---|---|---|
| probe | 0.6254 | 0.6257 |
| lgbm_es | 0.6279 | 0.6297 |
| catboost_es | 0.6303 | 0.6308 |

- vs probe: lgbm_es +0.0039 [0.0011, 0.0068]; catboost_es +0.0050 [0.0015, 0.0084].
- feature gain: 0.0004 [−0.0035, 0.0048] at probe, 0.0018 [0.0002, 0.0034] at
  lgbm_es, 0.0005 [−0.0010, 0.0019] at catboost_es.
- The run's own evaluation said t5 beat the work table by 0.0077; on the audit
  slice it is 0.0004. The search's gain did not replicate.
- Early stopping chose 242 (lgbm) and 525 (catboost) rounds at lr 0.03: *less*
  effective capacity than the probe's 200 at lr 0.1. On this noisy target the
  fixed budget overfits rather than underfits.

**Against the prediction.** House: capacity ≤ 0.005 held for lgbm; catboost's
0.007 is inside noise. Porto: capacity ≥ 0.005 held for catboost only, by
0.00003. Features shrinking at capacity: not on House; on Porto there was no
feature gain to shrink.

**Against the decision rule.** Met on Porto by catboost_es (interval excludes zero,
gain 0.0050 ≥ 0.005), at the threshold rather than clear of it. Formally: the
default should ship a fitted model and keep the probe as a ruler only. The more
robust reading is that a *fitted* budget beats a fixed one in both directions;
"more trees" is not the mechanism on Porto.

**Not asked, but read.** On both datasets the model family moved the audit score
as much as or more than everything the agent built: catboost_es on the raw work
table matches or beats the shipped probe on the agent's table. Two runs, one
table each; a second run per dataset would be needed before generalising.

## Reading r1 + r2 (2026-10-27)

r2 repeats r1 with a fresh agent run on the same audit slice (`holdout_fraction=None`
in the run, so the product's own sealed slice does not shrink work). All numbers on
the audit slice. House rmsle (lower better), Porto auc (higher better).

**Capacity: early-stopped fit vs the shipped probe, same selected table.**

| | House r1 | House r2 | Porto r1 | Porto r2 |
|---|---|---|---|---|
| lgbm_es | −0.0019 | −0.0006 | +0.0039 * | +0.0047 * |
| catboost_es | +0.0073 | +0.0043 | +0.0050 * | +0.0070 * |

(* = paired 95% interval excludes zero; positive = better than the probe.)
CatBoost is better in all four, clear of noise on Porto both times and at or above
the preregistered 0.005 both times. The decision rule is met twice: ship a fitted
model; the probe stays a ruler.

**Feature gain over the raw work table, under the shipped probe.** House +0.0082,
+0.0019; Porto +0.0004, −0.0008. Only House r1 is clear of zero, and only
marginally. The runs' own evaluation rows reported gains of 0.019/0.019 (House)
and 0.0077/0.0087 (Porto). CatBoost on the *raw* table (House 0.1346, Porto
0.6303) matches or beats what the product shipped in all four runs.

**Does the probe pick the table a fitted model would?** Spearman between the
probe's ranking of a run's tables and a fitted ruler's, on the evaluation rows:

| | House r1 | House r2 | Porto r1 | Porto r2 |
|---|---|---|---|---|
| lgbm_es | 0.96 | 0.65 | 0.41 | 0.06 |
| catboost_es | 0.39 | 0.24 | 0.75 | 0.42 |

What the disagreement costs, with catboost_es shipping, audit score of the probe's
pick vs catboost_es's own pick: House 0.1275 vs 0.1238, 0.1368 vs 0.1227; Porto
0.6308 vs 0.6309, 0.6316 vs 0.6324. Small on Porto, up to 0.014 rmsle on House.

**What this says.** A fixed 200-round ruler ranks tables only loosely like a
fitted model does, and on these four runs most of what it ranks are near-ties
whose differences do not replicate. The model shipped mattered more than the
table chosen. Two datasets, one agent model (DeepSeek v4.1 flash), two runs
each.

## Restaurant r1 (2026-10-27): `ForecastingAgent`, full profile

Audit slice: the last 39 days (2017-03-15 to 04-22, 28,031 rows), with reservation
columns rebuilt from bookings made before the cut, as serving rows see them. The
agent saw 224,077 rows up to 2017-03-14; its run split those into 179,415 work and
44,662 evaluation rows. Store-level bootstrap. rmsle, lower is better.

| | raw table (9 numeric) | selected t5 (53 cols) |
|---|---|---|
| shipped blend (7 members) | – | 0.6112 |
| probe | 0.7957 | 0.6070 |
| lgbm_es | 0.7955 | 0.5998 |
| catboost_es | 0.7960 | 0.6080 |

- Features: +0.19 over the raw table under every model (the raw table has no store
  id and no weekday, so this is a weak baseline).
- The blend is worse than each single model: lgbm_es +0.011 [0.008, 0.014].
- Rank agreement probe vs lgbm_es 0.89, vs catboost_es 0.66.
- The run's own evaluation score was 0.513.

**Where the 0.098 between 0.513 and 0.611 went** (`restaurant_diag.py`):

| shipped bundle, scored on the audit days | rmsle | vs shipped |
|---|---|---|
| as shipped, reservations as of the cut | 0.6112 | |
| same bundle, every booking counted | 0.5842 | +0.027 [0.022, 0.033] |
| lineage and blend refitted on all 224k rows | 0.5249 | +0.086 [0.074, 0.099] |
| same refit, lgbm_es | 0.5243 | +0.087 [0.075, 0.100] |

The shipped models and their history encoders are fitted on the work rows only,
which end about three months before the audit days. Refitting the same lineage on
every row the run had closes most of the gap; what remains (0.525 vs 0.513) is
about what the reservation skew and selection would explain. The skew costs 0.027.
After the refit the blend and lgbm_es are equal.

## Restaurant r2 (2026-10-27): the refit inside the product (0092)

Same 39 test days, same agent model, a fresh run. This run's finalize refitted the
selected table's features (6 steps) and all 7 blend members on its 224,077
training + validation rows before shipping; nothing was re-chosen. rmsle, lower is
better; store-level bootstrap.

| | r1 | r2 |
|---|---|---|
| what shipped | 0.6112 (training rows only) | **0.5225** (refit) |
| the same blend, training rows only | 0.6112 | 0.5678 |
| r1's offline refit replay (`restaurant_diag.py`) | 0.5249 | |
| the run's own validation score | 0.513 | 0.510 |

- Refit vs the same blend fitted on training rows only: **+0.045 [0.031, 0.062]**.
  Smaller than r1's offline +0.086, on a different table (24 tables, t21 selected).
- What ships now lands 0.012 from the run's own validation score, where r1 was
  0.098 away: the rest of r1's gap was stale fits, not selection.
- Capacity on the refit table: lgbm_es +0.0015 [−0.006, 0.014], catboost_es
  +0.0038 [−0.003, 0.016] against the shipped blend; neither clear of noise. After
  the refit, the blend, the probe and the capacity fits are level.
- Kaggle files: `_runs/capacity_ship_minimal/restaurant/submission_r2_all_rows.csv`
  (refitted on every training-file row, including our test days, after they were
  scored; the one to submit) and `submission_r2_shipped.csv` (the bundle as shipped).

## Rossmann r1 (2026-10-27): `ForecastingAgent`, refit inside the product

Test days: the last 48 of the training file (2015-06-14 to 07-31, 53,520 rows);
the agent saw everything before. Settings as in `research/rossmann_forecasting_agent/`
(validation 5%, 20-minute fit budget). Selected t14 (108 features, 8 steps incl. a
training-only filter), refitted on 798,508 training + validation rows. rmspe,
lower is better; store-level bootstrap.

| | test days |
|---|---|
| what shipped (refit, 5-member blend) | **0.1162** |
| the same blend, training rows only | 0.1237 |
| probe / lgbm_es / catboost_es on the refit table | 0.1205 / 0.1218 / 0.1213 |
| raw columns, best arm (lgbm_es) | 0.1469 |
| the run's own validation score | 0.1238 |

- Refit vs training-only: **+0.0075 [0.0058, 0.0093]**.
- The blend beats every single model by 0.004–0.006 (all intervals exclude zero).
- Features over raw columns: +0.025 at lgbm_es, +0.091 at the probe.
- Both xgb configurations dropped out of the blend: an agent transform produces
  `inf`, which xgboost refuses. The reason is not recorded in the journal or report.
- An earlier version of `evaluate.py` let `Customers` into the raw-column arms (it
  is not a serving column); they read 0.076 then. Fixed: raw arms use serving
  columns only. No other dataset in this directory had such a column.

## Kaggle (2026-10-27), private / public

| | Restaurant r2 | Rossmann r1 |
|---|---|---|
| refit on every training-file row (our test days included, after scoring) | **0.5181** / 0.4840 (~105/2150) | 0.1328 / 0.1289 (~1774/3300) |
| bundle as shipped | 0.5481 / 0.5017 (~1413/2150) | **0.1319** / 0.1198 (~1728/3300) |

Restaurant's previous best was 0.5498 private; median 0.5322, top 10% 0.5182.
Rossmann median 0.1297: still about the median. On Restaurant the latest 39 days
are worth 0.030 on Kaggle; on Rossmann the extra 48 days did not help (−0.001
private, −0.009 public), unexplained.

## Rossmann r3 (2026-10-27): known-in-advance calendar declared, after 0093

Same 48 test days as r1. `Open`, `Promo`, `SchoolHoliday` declared known in advance;
trailing windows reach back across the block start (0093). rmspe, lower is better.

| | r1 | r3 |
|---|---|---|
| shipped (refit) | **0.1162** | 0.1214 |
| same models, training rows only | 0.1237 | 0.1196 |
| refit vs training-only | +0.0075 [0.0058, 0.0093] | **−0.0018 [−0.0027, −0.0008]** |
| Kaggle private / public, refit on all rows | 0.1328 / 0.1289 | **0.1312** / 0.1306 |
| Kaggle private / public, as shipped | 0.1319 / 0.1198 | 0.1377 / 0.1265 |

- The agent built two future-window tables (`Promo` on t5: −0.015; `SchoolHoliday` on
  t10: −0.001); neither shipped, and it never tried `Open`. All 7 blend members trained.
- **The refit hurt here**, against 0092's registered prediction 1. Different table and
  run from r1; one reading, not explained.

### Offline: windows over the calendar on r3's table (`rossmann_open_windows.py`)

Windows per store, computed on every row (closed days included) with the product's
encoder: closed yesterday / tomorrow (2-row min), open days in the last / next 7.
One LightGBM, early-stopping protocol of `evaluate.py`; store-level bootstrap.

| | test days | vs r3's table |
|---|---|---|
| r3's table | 0.1269 | |
| + `Open` windows | 0.1204 | +0.0064 [0.0057, 0.0073] |
| + `Open`, `SchoolHoliday`, `Promo` windows | **0.1145** | +0.0124 [0.0112, 0.0136] |
| r3 shipped blend | 0.1214 | |

The calendar windows are worth more than anything else measured on Rossmann in this
directory, and a single model with them beats every shipped blend.

**Why the agent's windows failed, and a product gap.** r3's lineage starts with a
training-only filter dropping closed days; its future-window tables came after it.
Over the filtered rows `Open` is constant, and a window counted in rows covers more
calendar days in training (closed days skipped) than at prediction, where the filter
does not run and the request holds every day. A training-only filter before a
row-counted window step changes what the window means between training and
prediction. Any dataset with a training-only filter and a window step can hit this.

## Rossmann r4 (2026-10-27): after the window/filter ordering fix — not scored

Same setup as r3, after future-covariate windows were refused on a filtered table.
23 tables. The agent tried a filter on closed days once (t7, −0.0015) and dropped
it; every window after that was built on unfiltered tables. `SchoolHoliday` windows
on t14 gave the run's best table (t16, +0.0022, 0.1236 on validation). `Promo`
windows: −0.0053, −0.0004. It tried `Open` (t21, −0.0030), but as 364- and 112-day
means: a store's usual opening rate, not the short windows around a closure that
were worth +0.0064 offline.

**What shipped is t5** (target encodings, transforms, one sales history; 69
features): no later step beat its source by more than the noise bar of 0090, so
none of the calendar windows shipped. The evaluation on the test days and the
Kaggle files were skipped, since the shipped table holds none of what r4 was run
to test.

Reading: the ordering fix changed what the agent built, but on this run the agent
did not find the closure windows, and the gains it did find were too small for the
validation rows to confirm.

### Could r4's ruler have seen the closure windows? (`ruler_check.py`)

The windows of `rossmann_open_windows.py`, judged the way r4 judges a table: the
200-tree ruler fitted on r4's training rows, scored on its 48 validation days,
compared with `compare_predictions` and 0090's selection rule.

| added to | windows | gain on validation | block spread | passes 0090 |
|---|---|---|---|---|
| t5 (shipped) | `Open` | +0.0031 (5 of 5 blocks positive) | 0.0015 | yes |
| t16 (best) | `Open` | +0.0035 (4 of 5) | 0.0038 | no, just under |
| t5 | `Open` + `SchoolHoliday` + `Promo` | −0.0035 | 0.0106 | no |
| t16 | `Open` + `SchoolHoliday` + `Promo` | −0.0017 | 0.0080 | no |

- `Open` closure windows: the ruler sees about half their test-day gain, consistently;
  on t5 they would have shipped. The agent's miss is its choice of window (it built
  364/112-day means), not the harness.
- The full calendar set: +0.0124 on the test days, a small loss with a wide spread on
  the validation days. No agent could have shipped it. Not tested why; candidates are
  season (the validation days hold little summer school holiday, the test days a lot)
  and the ruler's fixed 200 trees (early stopping chose ~720 rounds for that set).
  A fixed validation stretch can only judge features whose effect falls inside it.

## Rossmann r5–r8 (2026-10-27): does the agent find the closure windows?

All with `Open`, `Promo`, `SchoolHoliday` declared known in advance, same 48 test days.
None was scored on the test days: none shipped a calendar window.

| run | agent model | agent texts | calendar windows built | shipped |
|---|---|---|---|---|
| r5 (stopped) | deepseek-v4-pro | "aggregate them forward" in 4 places | `Promo`, `Open`, `SchoolHoliday`: forward only | – |
| r6 (stopped) | deepseek-v4-pro | 3 of 4 pared back | `Promo`, `SchoolHoliday`: forward only | – |
| r7 | deepseek-v4-pro | all neutral | none (filtered early; refusals not recorded yet) | t10 |
| r8 | deepseek-v4-pro | all neutral, rejections recorded | `Promo` fwd 7/14/28 + block (+0.0001); `Open` fwd 14 + block (−0.046) | t13 |

- The agent texts had steered forward (system instructions, run brief, tool
  description, inspect_data note); all four now only name the columns and the tool.
- r8's journal records its rejections: a sandbox timeout and history windows shorter
  than the horizon. No window call was refused, so the ordering rule blocked nothing.
- With neutral texts and nothing in the way, the model still built only forward
  windows: the backward side (closed yesterday, reopened after a closure) was never
  tried in any of these runs.
- Unexplained: r8's two forward `Open` windows cost −0.046 on validation, far more
  than an uninformative feature should. Not checked whether that is real or a bug.

## Rossmann r9–r10 (2026-10-27): whole-horizon validation, "can look back as well as forward"

Both with deepseek-v4.1-flash-1017@eu, the calendar declared known in advance, and two
changes since r8: validation spans whole forecast horizons (48 days, not 43; r8's
whole-block windows had been scored on a different scale), and the run brief and tool
description say the windows "can look back as well as forward". Not scored on the
test days: neither shipped a calendar window.

| run | calendar windows built | shipped |
|---|---|---|
| r9 (cut off after 12 tables by a provider timeout) | `Promo`: trailing 1/7/28 + forward 7/14 (−0.0024) | t8, 54 features |
| r10 | `Promo`: week + trailing 7 + forward 7 (+0.0014); week + forward 7 (+0.0018). `Open`: group statistic only (+0.0003) | t14, 38 features |

- With the one-clause nudge the agent built backward windows for the first time (both
  runs), but over `Promo`. It never built short windows over `Open` in any run r4–r10.
- r10's rejections show it reaching for "stores open on this date" as a row-local
  transform (refused: batch-dependent) and for history keyed on a derived weekday
  (refused by a name check; fixed in d8f4883: derived keys are checked by replay).
- No later step cleared 0090's noise bar, so both runs shipped early tables.
