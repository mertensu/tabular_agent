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
