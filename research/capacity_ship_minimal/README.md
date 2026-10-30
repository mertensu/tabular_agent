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

## Rossmann r11 (2026-10-27): two-horizon validation; the selection rule checked offline

r10's setup with validation spanning at least two forecast horizons (96 days, not 48;
22408a5) and derived keys checked by replay. 30 minutes, 25 tables. It built `Open`
windows for the first time (t16: trailing 28 mean, forward 7 mean, forward 14 min;
−0.0024) and holiday-calendar flags from its own knowledge (t15, +0.0060). It shipped
t8 (probe 0.1426 on validation) although the run reached t24 (0.1250).

The longer validation did not lower the bar: 0090's `block_sd` is the spread of the
gain between periods, which more days estimate but do not shrink (0.002–0.011, as in
r10). What blocked t9…t24 is that each step is compared with its immediate source
only, and each step was inside its own spread while the chain was not.

**Offline check (`selection_check.py`, no LLM):** the alternative rule is that a table
passes when its gain over its *nearest passing ancestor* (its source while none has
passed) exceeds the spread of that comparison. It was applied to the stored validation
predictions of every run, and both picks were scored by the ruler (200-tree probe,
training rows only) on our own test days:

| run | 0090 ships | alternative ships | test probe, 0090 → alternative | test gain, 5 blocks |
|---|---|---|---|---|
| r1 | t14 | t10 | 0.1271 → 0.1268 | +0.0003 (3/5, inside spread) |
| r3 | t10 | t10 | 0.1252 | – |
| r4 | t5 | t16 | 0.1601 → **0.1377** | **+0.0224 (5/5)** |
| r7 | t10 | t18 | 0.1554 → **0.1411** | **+0.0144 (5/5)** |
| r8 | t13 | t13 | 0.1304 | – |
| r10 | t14 | t21 | 0.1330 → 0.1364 | −0.0034 (3/5; one block −0.017) |
| r11 | t8 | t20 | 0.1319 → 0.1295 | +0.0024 (4/5, beyond spread) |

- Better on test in 3 of 5 runs where the picks differ, two of them by a lot. One is
  neutral and one is slightly worse but inside noise. It is not simply looser: in r1 it
  shipped an earlier table.
- r4's and r7's 0090 picks stopped short, and on the test days they are the worst picks
  of the seven runs (0.160, 0.155).
- These are probe scores, not the shipped refitted blend; seven runs from one dataset.

**Kaggle (r11, private / public).** Both files use the shipped 7-model recipe and blend
weights, refitted on all training rows (`submit.py rossmann r11 [t24]`):

| table | validation probe | Kaggle private / public |
|---|---|---|
| t8 (shipped by 0090) | 0.1426 | **0.1316** / 0.1278 |
| t24 (best probe score) | 0.1250 | 0.1318 / **0.1234** |

The 0.018 validation lead of t24 is gone on the private board (−0.0002) and is
+0.0044 on the public one. Picking by the probe argmax alone would not have helped
here. The alternative rule's pick (t20) was not submitted.

**Kaggle, the alternative rule's picks (2026-10-27, private / public).** Same recipe for
each pair: the shipped 7-model blend and weights, refitted on all training rows
(`submit.py rossmann <tag> [table]`).

| run | 0090 ships | alternative ships | private | public | offline test gain |
|---|---|---|---|---|---|
| r4 | t5: 0.1410 / 0.1201 | t16: **0.1290** / **0.1140** | +0.0120 | +0.0061 | +0.0224 |
| r7 | t10: 0.1364 / 0.1249 | t18: **0.1321** / **0.1230** | +0.0042 | +0.0019 | +0.0144 |

Both runs where the offline check was clearest carry over to Kaggle in the same
direction, at about half the size. r4's t16 (0.1290 private) is the best Rossmann
submission of this experiment. Not yet scored: r10 (offline −0.0034) and r11's t20
(+0.0024).

**r10 and r11 added (2026-10-28).** r10's first files were broken by a train/serve type
skew: Kaggle's test file stores `Open` as floats (`1.0`, plus 11 missing), and r10's
lineage built a text key from it (`0_1.0`, never seen in training), so its encoding
fell back to the overall mean and predictions came out ~30% low. Fixed in the bundle
(serving columns get their training type back when no value changes); only r10's
tables were affected, every other submitted table replays identical features.

| run | 0090 ships | alternative ships | private | public | offline test gain |
|---|---|---|---|---|---|
| r4 | t5: 0.1410 / 0.1201 | t16: **0.1290** / **0.1140** | +0.0120 | +0.0061 | +0.0224 |
| r7 | t10: 0.1364 / 0.1249 | t18: **0.1321** / **0.1230** | +0.0042 | +0.0019 | +0.0144 |
| r10 | t14: 0.1310 / 0.1260 | t21: **0.1293** / 0.1259 | +0.0017 | +0.0000 | −0.0034 |
| r11 | t8: 0.1316 / 0.1278 | t20: **0.1270** / **0.1145** | +0.0046 | +0.0133 | +0.0024 |

The alternative is better on the private board in all four runs where the picks
differ, including r10, where the offline check had it slightly worse. r11's t24 (the
best probe score, 0.1318 / 0.1234) did worse than t20: picking by validation alone
would not have done this. r11's t20 (0.1270) is the best Rossmann submission of the
experiment.

## Where the remaining gap is: features, not models (2026-10-28, `gap_split.py`)

On r11's t20 (the 0094 pick), r11's 7 shipped recipes fitted on training rows
(validation), training + validation (our 48 test days) and all rows (Kaggle). rmspe.

| r11 t20 | blend, shipped weights | best member by validation |
|---|---|---|
| as the agent built it | 0.1215 · Kaggle **0.1270** / 0.1145 | 0.1292 |
| + short calendar windows | **0.1171** · Kaggle **0.1226** / 0.1139 | 0.1204 |

Test days, then Kaggle private / public. The windows are 12 columns per store:
`Open`, `SchoolHoliday`, `Promo` yesterday / tomorrow (2-row min) and over the last /
next 7 days (sum), as a `FutureWindowEncoder` step in the lineage.

- **The windows are worth +0.0044 on the test days and +0.0044 on Kaggle private.**
  0.1226 is the best Rossmann score of the experiment, still about 0.01 short of the
  top quarter.
- **The blend is not the problem.** It beats every single member by 0.003–0.008 even
  with elasticnet (~0.20) at about 1/7 weight; the member that is best on validation
  is fifth on the test days. r11's very weak members were a property of its t8 table.
- What is missing is signal the agent does not build. The tool can express these
  windows and 0094 can ship them; in r4–r11 the agent never built short windows
  over `Open`.

## Rossmann r12 (2026-10-28): gpt-6.1-terra@eu, medium reasoning, after 0094

Current code (0094 selection, serving types restored, two-horizon validation), the
r10 setup otherwise. 24 minutes, 26 tables. Shipped t26 (validation 0.1159; r11
shipped 0.1466 on the same 96 days).

- **It built the short closure windows unprompted**: `Open` over the last / next 7
  days (t10, +0.0077, passes), then the last / next 2, 3 and 14 days (t19). Then a
  `Sales > 0` filter and trend ratios (recent vs long-run level per store × weekday
  and promo). Each of t19 → t20 → t26 was inside its own noise; against t15, the
  nearest eligible ancestor, the chain cleared it (+0.0063 vs 0.0060), so 0094
  shipped t26. 0090 would have shipped t15, without the closure windows.
- It analysed its own errors by segment (Sunday 24% error vs 13–15% on other days),
  tried filters side by side and dropped what did not help.
- It cannot be told apart from recall: closure windows, holiday distances and trend
  ratios are well-known public Rossmann features, and the notes name the dataset.

| r12 t26 | Kaggle private / public |
|---|---|
| refit on all rows | 0.1233 / **0.1070** (public ~1,110 / 3,300) |
| bundle as shipped (data to 13 June) | 0.1303 / 0.4787 |
| as shipped, our 48 test days passed as `known_past` (0095) | 0.1303 / 0.1133 |

The shipped bundle's 0.479 was a product bug (0095): its trailing `Open` windows
could not reach back over the seven weeks between its data and Kaggle's test, and
were computed over the forecast rows alone, so the first days read as closures
(predictions ~2× on 1 August, normal by mid-August). The bundle now refuses and
takes `known_past`; with it, predictions match the all-rows refit day by day
(median ratio 0.98–1.03), and Kaggle public goes from 0.479 to 0.113. Private is
unchanged, so the private part of the test is the later days the bug never reached.
The rest of the gap to the all-rows file (0.007 private) is the seven weeks of
staleness only a refit removes.

## Restaurant r3 (2026-10-28): gpt-6.1-terra@eu, medium reasoning, after 0094/0095

r2's setup on the current code, with r12's agent model. Restaurant declares
`forecast_horizon=1` (the block's floor, 0047), so the two-horizon validation change
is a no-op here: validation is 43,969 rows, as in r2 (44,662). **4 minutes, 15
tables**; the agent stopped on its own. rmsle, lower is better.

| table | step (source) | validation probe | gain vs ancestor (spread) | passes |
|---|---|---|---|---|
| t1 | calendar, holiday distances (raw) | 0.7550 | −0.0142 (0.0016) | no |
| t2 | store × weekday history, 4–8 weeks (t1) | 0.7053 | +0.0497 (0.0597) | no |
| t3 | `holiday_flg` trailing / forward 7-day sums (t1) | – | refused, see below | – |
| t4 | store, store × weekday target encodings (t1) | 0.5351 | +0.2199 (0.0093) | yes |
| t5 | store history, 60–120 days (t1) | 0.5663 | +0.1887 (0.0115) | yes |
| **t6** | **store × weekday history, 90–300 days (t1)** | **0.5263** | +0.2288 (0.0106) | **yes, shipped** |
| t7–t11, t13, t14 | target encodings, store history, frequencies, genre / area × weekday history, store × holiday history, calendar flags (each on t6) | 0.5256–0.5396 | −0.0133 … +0.0006 | no |
| t12, t15 | t6's windows with other lengths (t1) | 0.5318, 0.5268 | +0.223, +0.228 | yes |

- **0094 and 0090 ship the same table (t6).** Every step on top of t6 is inside its
  noise or worse, and there is no chain to clear the bar together.
- t2's gain is all in the first validation blocks (+0.14, +0.11, +0.02, −0.002,
  −0.004): short history windows are worth a lot just after the cut and nothing six
  weeks out.

**Test days and Kaggle, next to r2:**

| | r2 (t21, 94 features) | r3 (t6, 34 features) |
|---|---|---|
| run's validation probe | 0.5105 | 0.5263 |
| shipped (refit) on our test days | **0.5225** | 0.5361 |
| probe / lgbm_es / catboost_es on the refit table | 0.5207 / 0.5210 / 0.5187 | 0.5330 / 0.5308 / 0.5303 |
| Kaggle, all-rows refit, private / public | **0.5181** / **0.4840** | 0.5323 / 0.5071 |
| Kaggle, as shipped, private / public | **0.5481** / **0.5017** | 0.5753 / 0.5380 |

- On Kaggle private r3 is 0.014 behind r2, the same as on our test days (0.0136):
  the test days carried over exactly. 0.5323 is at the median (0.5322). As shipped,
  the bundle loses 0.043 private to the all-rows refit, against r2's 0.030; the
  difference includes the linear member's runaway predictions described below.
- **The stronger model built a weaker table and stopped sooner.** The table is
  0.012 behind r2's under every model on the test days; the blend loses another
  0.003–0.006. r2's flash model stacked store × weekday target encodings, longer
  history and holiday distances over 24 tables; r3 found one strong history step and
  stopped after seven misses on top of it.
- The blend is worse than single lgbm_es / catboost_es (+0.005 [0.001, 0.011], +0.006
  [0.002, 0.012]). Members range 0.533 (lgbm) to 0.621 (an xgb) and 0.585
  (elasticnet) on the test days, at ~1/7 weight each. Without elasticnet: 0.5344.

**A train/serve skew: shipped vs all-rows prediction means per day.** On most days
the shipped file is ~0.85× the all-rows file (r2: ~0.91×, ordinary staleness), but
every Wednesday 1.07–1.12×, and on 2017-05-31 the mean is **121 visitors against
17**. Two stores: `air_cfdeb326418194ff` at 84,690 visitors that day,
`air_e42bdc3377d1eee7` at 3,288–3,482 on Wednesdays.

- The store had one day of 877 visitors (2017-03-08), the largest target value in
  the data. It falls after the last training fold's freeze, so no training row ever
  read it: there the 90-day mean never exceeds 176. The shipped bundle's rows end
  2017-03-14 and Kaggle's days are 40–78 days later, so on 05-31 its Wednesday
  window holds that day alone (count 1) and reads 877.
- Thin windows are not themselves the skew: the run's folds are 2–3 months long, so
  training rows sit 40–80 days past their fold's freeze too, and 1–3% of them have a
  90-day count of 1. (An earlier version of this section said the windows were
  nearly full in training; they are not.)
- The tree members stay near log 3 (~20 visitors). The elasticnet member, linear in
  log space, extrapolates to log 13.3, and the blend averages after inverting the
  transform (0044), so one member at 1/7 weight carries the row to 84,690.

**Fixed: linear members clip each feature to its fitted range** (`_ClipToFitRange`
in the elasticnet pipeline, learned per fold; rows inside the range are unchanged,
so a refit learns the same coefficients). Inserted into r3's shipped member:

| r3 | before | after |
|---|---|---|
| shipped, our test days | 0.5361 | 0.5336 |
| shipped, Kaggle rows: max prediction | 84,690 | 412 |
| shipped / all-rows per-day ratio, Wednesdays | 1.07–1.12 | 0.83–0.88 (as other days) |
| shipped, Kaggle private / public | 0.5753 / 0.5380 | **0.5676** / **0.5314** |
| all-rows refit, Kaggle private / public | 0.5323 / 0.5071 | 0.5322 / 0.5070 |

The clip is worth +0.0077 private and +0.0066 public on Kaggle. What the shipped
bundle still loses to the all-rows refit (0.035 private) is close to r2's 0.030:
staleness, as in r2. On the all-rows refit the clip changes almost nothing (+0.0002;
its largest prediction goes from 795 to 633): what remains large there is the
in-range case below. Files
`submission_r3_shipped_clip.csv`, `submission_r3_all_rows_clip.csv`.

**Not fixed: the linear member still leaves the target's range on in-range
inputs.** In the all-rows file, a store with seven weeks of history
(`air_1c0b150f9e696a5f`, never above 166 visitors) gets 500–630 on weekends: its
90-, 180- and 300-day windows are identical, every value within its training range,
and the elasticnet predicts 1,000–3,960 where the six tree members say 38–137. The
long windows are collinear in training, and a row where they coincide is a
combination the linear fit never saw. Per-column clipping cannot see that.

**0095 refused inside the run.** t3 (`holiday_flg` over the last / next 7 days per
store) never got a score: its trailing window needs the 6 rows before each
validation series' first row, and for 168 stores the training rows end earlier,
because a store has no row on a day it did not report (e.g. last training row
01-11, first validation row 01-14). The column is a calendar fact known for every
date, but the window counts the series' own rows, so the refusal fires on gaps no
caller could fill with `known_past`. The agent did not try that lane again.

Files: `_runs/capacity_ship_minimal/restaurant/submission_r3_all_rows.csv` and
`submission_r3_shipped.csv` (`submit.py restaurant r3`); `arms_restaurant_r3.json`.

## Restaurant r4 (2026-10-28): r3 with the reservation streams declared

`DataPlan.streams` (82f39d1) lets a forecast plan declare a second input recorded
one event per row with the time it became known. r4 is r3 with the two booking files
declared (one row per booking: party size, visit datetime, booking datetime), cut
where the agent's rows end (2017-03-14). The collapsed reservation columns are still
in the table, as in r2 and r3. Same model and prompt; no guidance added. 4 minutes,
18 tables. The elasticnet input clip (60d147d) is in.

| table | step (source) | validation probe | gain vs ancestor (spread) | passes |
|---|---|---|---|---|
| t1 | calendar, holiday distances; drops the collapsed reservation columns (raw) | 0.7546 | −0.0138 (0.0007) | no |
| t3 | store / genre / area target encodings (t1) | 0.5682 | +0.1864 (0.0064) | yes |
| t4 | store × weekday history, 90–300 days (t1) | 0.5266 | +0.2280 (0.0118) | yes |
| t5 | `holiday_flg` windows (t1) | – | refused by 0095, as in r3 | – |
| t6 | HPG bookings, made ≥1 day before (t4) | 0.5253 | +0.0013 (0.0011) | yes |
| **t7** | **AirREGI bookings, made ≥1 day before (t4)** | **0.5158** | **+0.0108 (0.0019)** | **yes, shipped** |
| t10 / t14 / t16 | AirREGI bookings made ≥42 / ≥21 / ≥7 days before (t4) | 0.5267 / 0.5242 / 0.5200 | −0.0000 / +0.0024 / +0.0066 | no / yes / yes |
| t11 | t7 + HPG bookings ≥1 day | 0.5149 | +0.0009 (0.0011) | no |
| t8, t9, t12, t13, t15, t17, t18 | target encodings, store history, a filter, ratios, store × holiday history | 0.5262–0.5577 | ≤ +0.0004 | no |

- **Given the streams, the agent used them on its fifth step** and compared four
  booking cut-offs. It again dropped the collapsed reservation columns first, for the
  same stated reason as r3. 0094 and 0090 ship the same table.
- A lead of 1 day is honest only for the first forecast day (0047): further out, the
  bookings made after the forecast is issued do not exist yet. In the run, validation
  reads the uncut stream, so it flatters short leads. On our test days the stream is
  cut at the first test day, as a forecast issued then would see it.

**Test days** (rmsle; shipped is the refitted blend):

| | r2 | r3 | r3, input clip | r4 |
|---|---|---|---|---|
| run's validation probe | 0.5105 | 0.5263 | | 0.5158 |
| shipped | **0.5225** | 0.5361 | 0.5336 | 0.5293 |
| probe / lgbm_es / catboost_es | 0.5207 / 0.5210 / 0.5187 | 0.5330 / 0.5308 / 0.5303 | | 0.5273 / 0.5266 / 0.5281 |

**By week after the forecast start** (r3 with the input clip vs r4, same refit
recipe; the difference is the booking features):

| week | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| r3 | 0.5435 | 0.5293 | 0.5496 | 0.5460 | 0.5170 | 0.5036 |
| r4 | 0.5341 | 0.5240 | 0.5432 | 0.5434 | 0.5158 | 0.5050 |
| gain | +0.0094 | +0.0053 | +0.0063 | +0.0027 | +0.0012 | −0.0014 |

- The bookings are worth what bookings should be: most in the first week, nothing by
  the sixth. No sign of a skew that hurts the far days.
- Overall r4 gains 0.004 on r3 with the clip, and stays 0.007 behind r2. The gap
  from validation to the test days is 0.0135 (r3: 0.0098, r2: 0.0120), which leaves
  room for the short lead's optimism but does not show it on its own.
- Kaggle files: shipped / all-rows prediction means per day 0.82–0.93 (no weekday
  spikes); largest predictions 353 / 540.

**Kaggle (all-rows refit, private / public):** r4 **0.5286 / 0.4960**; r3 with the
input clip 0.5322 / 0.5070; r2 0.5181 / 0.4840. The bookings are worth +0.0036
private (+0.0043 on the test days) and +0.0110 public, and r4 stays 0.0105 behind r2
on the private board (0.0068 on the test days).

**Kaggle, as shipped (private / public):** r4 0.5693 / **0.5214**; r3 with the input
clip **0.5676** / 0.5314; r2 0.5481 / 0.5017. The bundle was handed the full streams
(to 2017-04-22), so its booking features are as fresh as the all-rows file's while
its other fits end 2017-03-14. Public (the early forecast days) gains 0.010 on r3;
private (the later days, where bookings are sparse) is level (−0.0017).

## Restaurant r5 (2026-10-28): r4 repeated — a short history window, filled only after fold freezes

r4's setup unchanged. 7 minutes, 21 tables. Path: calendar (t1, −0.010), store history
28 / 90 / 180 days (t2, +0.184), store × weekday history 56 / 112 / 224 (t3, +0.035),
AirREGI bookings ≥1 day before (t4, +0.013), HPG bookings (t5, +0.001, no), then
store × weekday medians 70 / 140 / 210 (t13: +0.0016 over t5, inside its noise, but
+0.0027 over t4 against a spread of 0.0014). **0094 shipped t13; 0090 would have
shipped t4.** Both contain t2.

| | r4 | r5 |
|---|---|---|
| run's validation probe | 0.5158 | 0.5159 |
| shipped on our test days | **0.5293** | 0.5458 |
| probe / lgbm_es / catboost_es | 0.5273 / 0.5266 / 0.5281 | 0.5642 / 0.5382 / 0.5236 |

Same validation score, 0.017 apart on the test days, and the members on r5's table
spread from 0.524 to 0.693.

**The cause: t2's 28-day store window.** In training it is missing on 62% of rows and
its median count is 0. History is frozen at each fold's training prefix and the
blocks are ~3 months, so the window is filled only for rows within 28 days of a
freeze: in a few calendar stretches (filled share by month: 0 in Jan–Apr 2016 and Jul,
0.65 in Jun, 0.73 Aug, 0.90 Oct, 0.76 Dec, 0.39 Jan 2017, 0.60 Feb). On the test days, 1–39 days after
one freeze, it is filled for every row in the first four weeks.

| test days after the cut | 0–6 | 7–13 | 14–20 | 21–27 | 28–38 | all |
|---|---|---|---|---|---|---|
| r5 shipped | 0.5834 (bias +0.23) | 0.5463 (+0.16) | 0.5535 (+0.14) | 0.5628 (+0.14) | 0.5048 (+0.02) | 0.5458 |
| the 28-day window blanked | 0.5370 (+0.07) | 0.5171 (+0.03) | 0.5379 (+0.03) | 0.5520 (+0.09) | 0.5048 | **0.5273** |

(bias: mean log error; +0.23 is ~25% over-predicted.) Blanked, r5 beats r4. The run's
own validation could not see it: the validation block lies 1–80 days after one freeze,
the same spread of distances as training.

**A harness gap, not fixed.** `fit_history` refuses windows shorter than the declared
horizon (here 1 day, the floor; 0047) but not windows shorter than the folds' block
length. Such a window is present in training only right after each freeze, so the
model learns it from a few calendar stretches, while at serving it is present for
every row of the first weeks. Nothing about this is specific to the dataset: it
follows from the fold geometry whenever a history window is shorter than a block.

Kaggle files: r5's all-rows file starts the day after its data and carries the same
bias (day-1 mean 24.7 visitors vs r4's 21.3); the shipped file starts 40 days out,
where the window is empty (shipped / all-rows 0.70–0.88 per day, against r4's
0.82–0.93). **Prediction before submitting:** r5's all-rows file scores worse than
r4's, by more on the public board (the early days).

**Fixed (option 1 of three):** the history tool now refuses windows shorter than the
furthest a validation row lies past its fold's freeze. For Restaurant's runs that is
83 days. It would have refused r5's 28 / 56 / 70-day windows, and also r2's 56-day
store window in its shipped table; r3's and r4's (90 days and up) pass. Conservative:
it refuses windows that are only partly affected too.

## Restaurant r6 (2026-10-28): r5 repeated, after the history-window refusal (cf817c1)

Same setup, unchanged. 4 minutes, 18 tables. The refusal fired twice: the agent asked
for 4 / 12 / 24-day windows, then 28; after the second refusal every window it built
was 84 days or longer.

Path: AirREGI bookings ≥1 day before (t2, +0.037) and ≥43 days (t4, +0.019) on the
calendar; store target encodings (t5, +0.219, 0.5356); store × weekday history from
84 days (t7, +0.219, 0.5358); + target encodings (t9, +0.005); + store history from 84
days (**t14, +0.0025 against 0.0016, shipped, 0.5281**). Four later steps (store ×
month encodings, Golden Week distances, area and holiday history) did not pass. 0094
and 0090 ship the same table. **No booking feature reached the shipped table**: the
bookings were added on top of t7 in t8 (+0.0009), but the agent continued from t9.

| | r2 | r4 | r5 | r6 |
|---|---|---|---|---|
| run's validation probe | 0.5105 | 0.5158 | 0.5159 | 0.5281 |
| shipped on our test days | **0.5225** | 0.5293 | 0.5458 | 0.5280 |
| validation → test | +0.012 | +0.014 | +0.030 | **−0.000** |
| probe / lgbm_es / catboost_es | 0.521 / 0.521 / 0.519 | 0.527 / 0.527 / 0.528 | 0.564 / 0.538 / 0.524 | 0.527 / 0.530 / 0.526 |

- The best test-day score of the gpt-6.1 runs, and the first whose validation score
  is what the test days give. The members agree on its table (0.526–0.530).
- By week the bias is small and negative throughout (−0.05 to −0.12 in log), with no
  first-week excess like r5's +0.23.
- Kaggle files: shipped / all-rows per day 0.82–0.91; largest predictions 137 / 212.
- Still 0.0055 behind r2. One reading per setup; the two r4-setup runs before the
  refusal differed by 0.017.

**Kaggle, all-rows refit (private / public):** r6 0.5309 / 0.4954; r4 **0.5286** /
0.4960; r2 **0.5181** / **0.4840**. r6 is 0.0013 ahead of r4 on the test days and
0.0023 behind on Kaggle private: level within noise. r2 stays ~0.012 ahead.

**Kaggle, as shipped (private / public):** r6 0.5645 / 0.5383; r4 0.5693 / **0.5214**;
r2 **0.5481** / 0.5017. r6 loses 0.034 private to its all-rows refit (r2: 0.030), the
ordinary staleness. Against r4 it is 0.005 better on private and 0.017 worse on public:
r6 shipped no booking feature, and r4's bookings were fresh for Kaggle's early days.

### Did dropping the reservation columns cost r6 anything? (`reservation_check.py`)

r6's shipped recipe (7 models, parameters, blend weights) refitted on its training +
validation rows, once on t14 as shipped and once with the six collapsed reservation
columns added. On our test days the columns hold only bookings made before the first
test day, as a forecast issued then would see them.

| test days | shipped recipe | + reservation columns |
|---|---|---|
| all | 0.5280 (reproduces the shipped 0.5280) | 0.5262 |
| week 1 / 2 / 3 | 0.5389 / 0.5237 / 0.5377 | 0.5307 / 0.5224 / 0.5385 |
| week 4 / 5 / 6 | 0.5381 / 0.5135 / 0.5072 | 0.5367 / 0.5146 / 0.5055 |

- **+0.0018 [−0.0015, 0.0047]**, inside the noise, almost all of it in the first
  week. The same shape as r4's booking features from the stream. r6 shipped no
  booking feature, so this is roughly what any booking signal is worth on top of it.
- The agent's decision to drop the columns cost at most about 0.002. **It does not
  explain r2's lead** (0.0055 on the test days, 0.012 on Kaggle private). That lies
  elsewhere: r2's stacked target encodings, or its search, are the remaining
  candidates.

## Restaurant r7, r8 (2026-10-28): r6 with the finalize inventory (af1ad0b)

The agent's first `finalize_run` now returns an inventory of the run and packages
nothing: what would ship and its steps, tables built and passed by kind, tables that
passed but were never built on, and table-building tools never called. A second call
finishes. Otherwise r6's setup.

| | r2 | r6 | r7 | r8 |
|---|---|---|---|---|
| tables / minutes | 24 / – | 18 / 3.9 | 15 / 4.9 | 17 / 4.1 |
| shipped table | t21 | t14 | t6 | t12 |
| run's validation probe | 0.5105 | 0.5281 | 0.5134 | 0.5250 |
| shipped on our test days | **0.5225** | 0.5280 | 0.5270 | 0.5243 |
| probe / lgbm_es / catboost_es | 0.521 / 0.521 / 0.519 | 0.527 / 0.530 / 0.526 | 0.522 / 0.521 / 0.526 | 0.525 / 0.525 / 0.524 |

- **r7** shipped t6: store history ≥90, store × weekday history ≥90, AirREGI then HPG
  bookings ≥1 day before (validation 0.5134, the best probe of any run). Nine later
  steps did not pass. Validation → test +0.014, as r4's (also 1-day bookings).
- **r8** shipped t12: store target encodings, store history ≥90, median target
  encodings by store and store × weekday. Its bookings (t5, +0.0035) passed and were
  never built on. Validation → test −0.001, as r6's (no bookings shipped).
- The history-window refusal fired in both (28; 8 / 20 / 40).
- **Whether the inventory made the agent search longer cannot be read off these runs.**
  The journal records that it was shown, not at which table. In both runs the last
  table was written 1.5–2 minutes before the bundle (r7 14:28:37 → 14:30:18, r8
  14:35:20 → 14:37:07), which the final fits account for, and neither built more
  tables than r6. Most likely the agent read it and finished.
- r8 is the closest any gpt-6.1 run has come to r2 (0.0018 on the test days).

## Restaurant r9 (2026-10-28): r7/r8 with the transcript and the inventory's timing

Two additions to the harness, otherwise r7/r8's setup. Each run now writes its full
transcript to `messages.json` in its run directory (what the agent said, every tool
call and what it was told), however the run ends. The journal records how many
tables existed when the inventory was shown (`stop.inventory_shown.tables`) and at
finalize (`stop.tables_at_finalize`); equal numbers mean the agent finished
straight after reading it. The run used these changes uncommitted on top of
ff6ef65, the commit `result_restaurant_r9.json` names.

| | r2 | r8 | r9 |
|---|---|---|---|
| tables / minutes | 24 / – | 17 / 4.1 | 12 / 33.7 |
| shipped table | t21 | t12 | t4 |
| run's validation probe | 0.5105 | 0.5250 | 0.5166 |
| shipped on our test days | **0.5225** | 0.5243 | 0.5250 |
| probe / lgbm_es / catboost_es | 0.521 / 0.521 / 0.519 | 0.525 / 0.525 / 0.524 | 0.524 / 0.523 / 0.523 |

- **The inventory was never shown.** The stop rule ended the search at 12 tables:
  the seven after t4 were all inside the noise. The agent's next call was
  `finalize_run`, and the inventory is skipped once the stop rule has ended the
  search. The agent never asked to finish on its own. r9 reads like r4 and r5, not
  like r7 and r8.
- The seven tables after t4 were mostly siblings built on t3 (encodings, holiday
  windows, genre × weekday and store × holiday history, HPG bookings), not steps on
  t4. Only the last request, HPG bookings on t4, built on the winner, and that was
  the one refused.
- r9 shipped t4: calendar, store history ≥90, store × weekday history ≥90, AirREGI
  bookings at leads 1/7/28/45. Validation → test +0.009, between r7's +0.014 (also
  1-day bookings) and r8's −0.001 (none shipped).
- In its closing message the agent flagged its own 1-day booking lead: "those
  features may have contained more information in validation than would be
  available at the forecast cutoff, so real-world accuracy could be worse."
- After the run was finalized it called `finalize_run` a second time and was told
  the run was already finalized. The tool's docstring says the first call returns an
  overview; when the stop rule has ended the search, the first call finishes instead.
- The 33.7 minutes are two tool calls (store × weekday history 17.5 min, AirREGI
  bookings 9.5 min), not the gateway. The full test suite ran alongside on the same
  machine; after it was stopped the remaining tables came in seconds, as in r8 (5–10 s
  per table, with the same windows and a 180-day median). Contention is the only
  difference found; whether it explains a hundredfold slowdown was not checked.

### Would the table the agent trusted have done better? (`terminal_check.py`)

r9's shipped recipe (7 models, parameters, blend weights) refitted on its training +
validation rows, once on t4 as shipped and once on t3's columns only (t4 without
the 12 AirREGI booking columns). Streams hold only bookings made before the first
test day.

| test days | t4 (shipped) | t3 (what the agent built on) |
|---|---|---|
| all | 0.5250 (reproduces the shipped 0.5250) | 0.5287 |
| week 1 / 2 / 3 | 0.5288 / 0.5212 / 0.5381 | 0.5416 / 0.5261 / 0.5435 |
| week 4 / 5 / 6 | 0.5392 / 0.5136 / 0.4977 | 0.5388 / 0.5111 / 0.4979 |

- **t4 is better by 0.0037 [0.0013, 0.0060].** Shipping t3, as the agent's judgement
  suggested, would have cost that. The gain is all in the first three weeks, where
  the cut-off bookings still cover the visit; from week 4 the two are level.
- Validation put the same step at +0.010. The agent was right that validation
  overstates the bookings and wrong that they were worth nothing.
- Why validation overstates: the run declares a forecast horizon of 1 day (the gap
  to the first forecast day), so leads of 1 day pass the stream tool's
  lead ≥ horizon check. Each validation row then sees its bookings up to 1 day before
  its visit, while a served row 30 days into the forecast sees only those made before
  the forecast was issued.

## Restaurant r10 (2026-10-28): r9 with streams read as of the forecast's issue (0096)

| | r8 | r9 | r10 |
|---|---|---|---|
| tables / minutes | 17 / 4.1 | 12 / 33.7 | 17 / 3.6 |
| shipped table | t12 | t4 | t8 |
| run's validation probe | 0.5250 | 0.5166 | 0.5285 |
| shipped on our test days | 0.5243 | 0.5250 | 0.5248 |
| validation → test | −0.001 | +0.009 | **−0.004** |
| probe / lgbm_es / catboost_es | 0.525 / 0.525 / 0.524 | 0.524 / 0.523 / 0.523 | 0.528 / 0.528 / 0.524 |

- **AirREGI bookings at leads 1/7/30 now gain +0.0037 on validation** built on the
  encodings (t3), +0.0009 on store history (t7, t11), inside the noise. r9's same
  step gained +0.010 on validation and +0.004 on the test days: the cut removed the
  overstatement, and what is left is about what the test days gave.
- Shipped t8: calendar, target encodings, store history ≥90 with a 180-day median.
  No booking feature. Test days level with r8 and r9; validation is no longer
  optimistic.
- **The agent finished on its own** after 17 tables, at the same moment the stop
  rule's patience (6 tables without beating the best, t12–t17) ran out. No call
  was refused, but the inventory was not shown either: it is skipped once the stop
  rule considers the search ended.
- The agent never saw the stop rule's progress line: it is attached to training
  results only, and the minimal agent builds tables without training.

## Rossmann r13, r14 (2026-10-29): r12 after 0096 and 0097

r12's setup on the code with 0096 (Rossmann declares no stream: no effect) and 0097
(a run is bounded by 60 tables and 30 minutes; no stop after six tables without
improvement; the budget in every table result). rmspe on our 48 test days, lower is
better.

| | r12 | r13 | r14 |
|---|---|---|---|
| tables / minutes of search | 26 / 24 | 14 / ~6 (asleep) | 33 / 23 |
| shipped table | t26 | t14 | t20 |
| shipped on our test days | 0.1189 | 0.1247 | **0.1186** |
| probe / lgbm_es / catboost_es | 0.122 / 0.119 / 0.121 | 0.126 / 0.121 / 0.120 | 0.122 / 0.119 / 0.118 |
| training-only fit | 0.1202 | 0.1296 | 0.1232 |

- **r13 is not a reading of 0097.** The machine slept from 23:09 to 06:59; the run
  advanced only in ~2-second dark wakes, and the 30-minute clock ran on, so the
  agent saw no time left after t14 and finished. Runs now go under `caffeinate -i`,
  lid open.
- **r14: level with r12 on the test days** (0.1186 vs 0.1189). It searched 33
  tables in 23 minutes and finished on its own with 7 minutes left. First run in
  which the inventory was shown (at 33 tables); it finalized straight after
  (`tables_at_finalize` 33). Shipped t20: calendar, store × weekday history, forward
  and trailing `Open` / `Promo` windows, a training-only filter.
- Of its best tables, t17 (+0.011), t20 and t23 came after table 15. Under the old
  six-table rule the run would have ended around t16 (t10 was the best for t11–t16).
- r14's lid was closed at 07:54, during the refit: 39 minutes instead of ~4, and
  the agent's closing message failed on a dropped connection after the bundle was
  written. `result_rossmann_r14.json` was written from the run directory afterwards.
- `evaluate.py` and `submit.py` now pass `known_past` (0095) where a bundle's stored
  past ends before the rows it predicts: the validation rows for the training-only
  fit, our test days for Kaggle "as shipped". r12 (also a trailing `Open` window) is
  scored here for the first time with `evaluate.py`.
- A general bug seen in r14: one-hot columns named `PromoInterval_Jan,Apr,Jul,Oct`
  make LightGBM refuse the table ("special JSON characters in feature name"), so the
  probe cannot score it. The agent worked around it. Fixed after r14: LightGBM and
  XGBoost models rename their input to names they accept (`_safe_names.py`).

**Kaggle (private / public):**

| | r12 | r13 | r14 |
|---|---|---|---|
| refit on all rows | **0.1233** / **0.1070** | 0.1340 / 0.1200 | 0.1245 / 0.1109 |
| as shipped, our test days as `known_past` | 0.1303 / 0.1133 | 0.1334 / 0.1251 | **0.1263** / 0.1185 |

r14 is 0.0012 behind r12 on private for the all-rows refit and 0.004 ahead as
shipped: level, as on our test days. The median is 0.1297.

## Re-ranking Rossmann r14 and Restaurant r10 (2026-10-29): does the probe pick the table a fitted model would?

The r1/r2 re-ranking covered House and Porto only. These are the two forecasting
runs on the current code (0094–0097), with 33 and 17 tables. `rerank.py` now
replays forecast lineages as the run did: validation rows read the streams cut at
their issue time and no known past; the test days read the streams cut at their
first day and the validation rows as known past. Every fit is on the work rows only,
as during the search, so test-day scores sit above what shipped; the comparison is
between tables, not against the shipped number.

**Prediction, written before the run.**

- Rank agreement (Spearman, validation rows) probe vs lgbm_es ≥ 0.7 on both runs:
  most steps here are large (calendar, store history), not near-ties.
- Where a fitted ruler picks a different table than the probe, the gain on the test
  days, under that ruler's model, is within ±0.003 and its interval includes zero.

**Decision rule.** If on either run a fitted ruler's pick beats the probe's pick by
≥ 0.003 with the store-bootstrap interval excluding zero, ranking tables with an
early-stopped fit is worth building. Otherwise the probe's ranking costs nothing
measurable on these datasets, and the next lever is elsewhere.

**Reading.** The refitted probe reproduces the journal on every scored table of both
runs (Rossmann's work table was never probed in the run; t1 there was unscorable).
Test-day scores, fits on work rows only; intervals are store-bootstrap.

| | Restaurant r10 (rmsle, 17 tables) | Rossmann r14 (rmspe, 33 tables) |
|---|---|---|
| Spearman probe vs lgbm_es / catboost_es | 0.71 / 0.54 | 0.63 / 0.81 |
| probe's pick | t11 | t29 |
| lgbm_es's pick, vs the probe's (under lgbm_es) | t14, **+0.0047 [0.0033, 0.0063]** | t24, +0.0011 [0.0002, 0.0020] |
| catboost_es's pick, vs the probe's (under catboost_es) | t12, **+0.0045 [0.0027, 0.0063]** | t10, −0.0007 [−0.0019, 0.0004] |
| table that shipped (0094's bar) | t8 | t20 |
| best table on the test days, any ruler | t3 | t21 / t22 |

- **Against the prediction.** Rank agreement ≥ 0.7 held for lgbm_es on Restaurant
  and catboost_es on Rossmann only. "Inside noise" held on Rossmann and failed on
  Restaurant, where both fitted rulers' picks beat the probe's by ~0.0046.
- **Against the decision rule.** Met on Restaurant by both fitted rulers; not on
  Rossmann (lgbm_es clear of zero but 0.001, catboost_es inside noise).
- **Why Restaurant needs a second look before building.** These are work-only fits,
  and Restaurant's test days begin about three months after the work rows end. Every
  ruler's best table on the test days is t3, ~0.02 better than anything the search
  went on to prefer on validation: history features going stale across the gap,
  which no ruler sees. The picks' difference is measured through that staleness.
  Refitting each table on training + validation rows (what ships since 0092) removes
  it.
- **Where the probe misleads most** is early in a run. On Rossmann t2–t6 the probe
  read 0.19–0.21 and lgbm_es 0.12–0.13; t7 looked like a 0.06 step to the probe and
  a 0.006 loss to lgbm_es. Once store history was in (t9 on), all three rulers sat
  within ~0.01 of each other and the tables were near-ties.
- **Cost.** lgbm_es + catboost_es took about 4 minutes per Rossmann table here,
  against ~10 seconds for the probe; an early-stopped ruler inside a 30-minute
  budget would allow far fewer tables.

### Restaurant r10 refitted as shipped (`rerank_refit.py`)

Same picks (validation, work-only fits); each table's lineage and each ruler's model
refitted on training + validation rows, as finalize does (0092), and scored on the
test days. Check: the probe on t8 should reproduce `arms_restaurant_r10.json`'s
refitted probe (0.5284).

**Prediction, written before the run.** Most of the +0.0046 was staleness: refitted,
both fitted rulers' picks beat the probe's by less than 0.003, and t3 stops being
the best table on the test days.

**Reading.** The check holds: the probe on t8 refitted reads 0.528392, as in
`arms_restaurant_r10.json`, and the work table reproduces all three of its arms.

| refitted on training + validation | probe | lgbm_es | catboost_es |
|---|---|---|---|
| t11 (probe's pick) | 0.5271 | 0.5268 | 0.5233 |
| t14 (lgbm_es's pick) | **0.5216** | **0.5219** | 0.5200 |
| t12 (catboost_es's pick) | 0.5256 | 0.5247 | 0.5223 |
| t8 (shipped) | 0.5284 | 0.5284 | 0.5243 |
| t3 | 0.5290 | 0.5263 | 0.5275 |

- **lgbm_es's pick vs the probe's: +0.0049 [0.0033, 0.0066]** under lgbm_es, the same
  as work-only (+0.0047). Staleness did not make it; my prediction was wrong here.
  t14 is the best table on the test days under the probe and lgbm_es alike; the
  probe ranked it sixth on validation, 0.0013 behind t11.
- **catboost_es's pick vs the probe's: +0.0010 [−0.0005, 0.0025]**, inside noise
  (work-only: +0.0045). Here staleness did make most of it.
- t3 is no longer the best table on the test days (prediction right): it was the
  stale fits.
- Against what shipped (t8): lgbm_es's pick +0.0065 [0.0043, 0.0088], catboost_es's
  +0.0019 [0.0005, 0.0033].

**Across the two forecasting runs**, picking by lgbm_es beat picking by the probe in
both, clear of zero: +0.0049 (Restaurant, refitted) and +0.0011 (Rossmann,
work-only). Picking by catboost_es did not, in either. Each is one choice per run,
so two runs are two draws, not a rate.

### What finalize's fitted ruler ships on the finished runs (`fitted_choice_replay.py`)

Since `6499326` finalize picks the shipped table with an early-stopped LightGBM
(lr 0.03, up to 5000 rounds) under 0094's rule. The replay copies a finished run and
calls the product's own choice (`tools._best_table_path`); no LLM calls.

| | Restaurant r10 | Rossmann r14 |
|---|---|---|
| shipped (probe's choice) | t8 | t20 |
| fitted ruler's choice | t14 | t24 |
| eligible tables | — | t3, t5, t14, t19, t24, t29 (t20 not among them) |
| minutes at finalize | 1.6 | 30.5 |

- On Rossmann t24 is the best of all 32 tables under the ruler (0.1113 vs 0.1168 for
  t20 on validation); in the work-only re-rank above it read 0.1226 on the test days
  against t20's 0.1272.
- **Cost.** 30.5 minutes for 32 Rossmann tables, as long as the search budget.
- **Ceiling.** 7 of the 32 tables stopped at or within a few rounds of 5000 (t2–t6,
  t15, t19), so their scores are likely pessimistic; t19, second, stopped at 4996.

### House r3 and Restaurant r11 (2026-10-30): first runs with `fit_encoding(method="categorical")`

Both on `c81144a`+, the agent told nothing beyond the tool's own description.

- **House r3** (minimal profile, so no fitted ruler at finalize): the agent never
  tried `categorical`; it frequency- and target-encoded, and noted `MSSubClass` "is an
  int but categorical" before frequency-encoding it. Shipped t13; on the audit rows it
  read 0.1458 under the probe against the work table's 0.1360 (gain −0.0097
  [−0.039, +0.012]); r1 0.1348, r2 0.1411.
- **Restaurant r11** (full profile, 36 tables, 17.6 min): three categorical steps.
  The six text columns as categories (t17) +0.0015 under the probe, not sign-stable;
  every later table descends from it. Store/genre/area × weekday crosses (t20)
  −0.0063, worse in all five blocks. Fitted ruler shipped t32 over the probe's t29
  (0.5368 vs 0.5372 on validation). Test days: shipped **0.5137** against r10's
  0.5248; lgbm_es 0.5150 (r10 0.5284), catboost_es 0.5250 (r10 0.5243).
- CatBoost did not gain where LightGBM did, and still carries 0.31 of r11's blend
  weight (two members). Not yet checked: the blend with and without them.
- `evaluate.py` now fits CatBoost through the product's classes, which declare
  category columns; it crashed on r11 before.
